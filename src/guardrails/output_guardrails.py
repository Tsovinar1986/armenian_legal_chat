"""Checks run on AI-generated output before it's returned to the user.

Purpose: catch things the LLM itself might leak or produce that input
guardrails can't — PII the model echoes back or invents, and indecent
language in a generated response (rare, but cheap to check).
"""
import os

from src.guardrails.pii_detector import detect_pii, redact_pii
from src.guardrails.schemas import GuardrailResult
from src.guardrails.validators import load_phrase_list

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_INDECENT_TERMS = load_phrase_list(os.path.join(_THIS_DIR, "indecent_terms.txt"))


def check_output_indecent_language(text: str) -> GuardrailResult:
    lowered = (text or "").lower()
    hits = [t for t in _INDECENT_TERMS if t in lowered]
    if hits:
        return GuardrailResult(passed=False, category="indecent_language", reasons=[f"matched: {h}" for h in hits])
    return GuardrailResult(passed=True)


def check_output_pii(text: str) -> GuardrailResult:
    """Flags PII in generated output and provides a redacted version — unlike
    input PII (which is just flagged), output PII is more concerning since it
    could mean the model echoed something it shouldn't have, so callers should
    generally prefer redacted_text over the raw response when this fires."""
    matches = detect_pii(text)
    if matches:
        return GuardrailResult(
            passed=False,
            category="pii",
            reasons=[f"{m.kind} detected in generated output" for m in matches],
            pii_matches=matches,
            redacted_text=redact_pii(text, matches),
        )
    return GuardrailResult(passed=True)


def run_output_guardrails(text: str) -> GuardrailResult:
    """Run all output checks; PII takes priority (it comes with a usable
    redacted_text), then indecent language."""
    pii_result = check_output_pii(text)
    if not pii_result.passed:
        return pii_result

    indecent_result = check_output_indecent_language(text)
    if not indecent_result.passed:
        return indecent_result

    return GuardrailResult(passed=True)
