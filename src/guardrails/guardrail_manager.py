"""Single entry point that orchestrates the guardrail checks for a domain.

Usage:
    manager = GuardrailManager(domain="legal")  # or "therapist"
    input_result = manager.check_input(user_message, history=prior_turns)
    if not input_result.passed and input_result.category in ("prompt_injection", "indecent_language", "off_topic"):
        ...  # block / respond with a redirect instead of calling the LLM

    output_result = manager.check_output(generated_answer, context=retrieved_context)
    text_to_return = output_result.redacted_text or generated_answer
"""
from src.guardrails.input_guardrails import check_topic_scope, run_input_guardrails
from src.guardrails.output_guardrails import run_output_guardrails
from src.guardrails.rag_guardrails import check_rag_groundedness

VALID_DOMAINS = {"legal", "therapist"}


class GuardrailManager:
    def __init__(self, domain: str = "legal"):
        if domain not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(VALID_DOMAINS)}, got {domain!r}")
        self.domain = domain

    def check_input(self, text: str, history: list = None):
        """Run all input guardrails. See input_guardrails.run_input_guardrails
        for check order and what each category means.

        For the legal domain only, and only on the first message of a
        conversation (history empty/None), also gates on check_topic_scope —
        a message with no legal or mental-health keyword gets category
        "off_topic". Skipped once history exists so a legitimate short
        follow-up ("how much time do I have?") that relies on earlier
        context isn't blocked for not repeating a keyword; skipped for the
        therapist domain since MentalHealthQAClassifier already scopes that
        conversation, and this heuristic isn't tuned for it."""
        result = run_input_guardrails(text)
        if not result.passed:
            return result

        if self.domain == "legal" and not history:
            topic_result = check_topic_scope(text)
            if not topic_result.passed:
                return topic_result

        return result

    def check_output(self, text: str, context: str = None):
        """Run output guardrails (PII, indecent language), then — only for the
        legal domain, and only if context was supplied — a RAG groundedness
        check on cited case numbers. Therapist-chat output has no "retrieved
        context to be grounded in" in the same sense, so groundedness is
        skipped there."""
        result = run_output_guardrails(text)
        if not result.passed:
            return result

        if self.domain == "legal" and context is not None:
            groundedness = check_rag_groundedness(text, context)
            if not groundedness.passed:
                return groundedness

        return result
