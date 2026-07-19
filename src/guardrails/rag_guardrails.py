"""Basic groundedness check for RAG-generated answers: flags case numbers the
generated answer cites that don't actually appear anywhere in the retrieved
context it was supposed to be grounded in — a cheap signal for hallucinated
citations, not a guarantee of full factual accuracy.
"""
import re

from src.guardrails.schemas import GuardrailResult

# Armenian case numbers look like "ԱՐԴ/0161/02/24-1" or "ԵԴ2/8262/02/24-1" —
# uppercase Armenian/Latin letters, digits, slashes, dashes, at least 6 chars.
_CASE_NUMBER_RE = re.compile(r"\b[Ա-ֆA-Z]{2,4}\d?/\d{2,6}/\d{2}/\d{2}-\d\b")


def extract_case_numbers(text: str) -> list:
    if not text:
        return []
    return list(dict.fromkeys(_CASE_NUMBER_RE.findall(text)))  # de-duped, order preserved


def check_rag_groundedness(answer: str, context: str) -> GuardrailResult:
    """Flags any case number cited in `answer` that isn't present in `context`
    (the text the RAG pipeline actually retrieved and gave the LLM to work
    from). If the answer cites no case numbers at all, there's nothing to
    check and this passes — plenty of valid answers don't cite one."""
    cited = extract_case_numbers(answer)
    if not cited:
        return GuardrailResult(passed=True)

    context = context or ""
    ungrounded = [case_number for case_number in cited if case_number not in context]
    if ungrounded:
        return GuardrailResult(
            passed=False,
            category="rag_ungrounded",
            reasons=[f"cited case number not found in retrieved context: {c}" for c in ungrounded],
        )
    return GuardrailResult(passed=True)
