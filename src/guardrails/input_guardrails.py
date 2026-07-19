"""Checks run on user input before it reaches the classifier/RAG/LLM.

NOTE ON SCOPE: the crisis-term check here is a defense-in-depth backstop, not
the primary safety mechanism. The real crisis short-circuit is
src/services/crisis_detection.detect_crisis_signal + the
MentalHealthRiskClassifier, called directly by LegalAgent.get_advice() and
api.py's /api/therapist-chat *before* input guardrails ever run — those two
checks are what actually route to CRISIS_RESPONSE_HY/EN. This module's crisis
flag exists so a caller that only invokes GuardrailManager (skipping
get_advice) still gets a signal, not to replace the tested crisis pipeline.
"""
import os

from src.guardrails.schemas import GuardrailResult
from src.guardrails.validators import is_within_length_limit, load_phrase_list

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CRISIS_TERMS = load_phrase_list(os.path.join(_THIS_DIR, "crisis_terms.txt"))
_INDECENT_TERMS = load_phrase_list(os.path.join(_THIS_DIR, "indecent_terms.txt"))
_INJECTION_PATTERNS = load_phrase_list(os.path.join(_THIS_DIR, "prompt_injection_patterns.txt"))


def check_prompt_injection(text: str) -> GuardrailResult:
    lowered = (text or "").lower()
    hits = [p for p in _INJECTION_PATTERNS if p in lowered]
    if hits:
        return GuardrailResult(passed=False, category="prompt_injection", reasons=[f"matched: {h}" for h in hits])
    return GuardrailResult(passed=True)


def check_indecent_language(text: str) -> GuardrailResult:
    lowered = (text or "").lower()
    hits = [t for t in _INDECENT_TERMS if t in lowered]
    if hits:
        return GuardrailResult(passed=False, category="indecent_language", reasons=[f"matched: {h}" for h in hits])
    return GuardrailResult(passed=True)


def check_crisis_terms(text: str) -> GuardrailResult:
    """Defense-in-depth only — see module docstring."""
    lowered = (text or "").lower()
    hits = [t for t in _CRISIS_TERMS if t in lowered]
    if hits:
        return GuardrailResult(passed=False, category="crisis", reasons=[f"matched: {h}" for h in hits])
    return GuardrailResult(passed=True)


def check_length(text: str, max_length: int = 8000) -> GuardrailResult:
    if not is_within_length_limit(text, max_length):
        return GuardrailResult(passed=False, category="length", reasons=[f"message exceeds {max_length} characters"])
    return GuardrailResult(passed=True)


def run_input_guardrails(text: str) -> GuardrailResult:
    """Run all input checks and return the first failure, or a combined pass.

    Order matters: crisis is checked first since it's the most safety-critical
    category to surface if a caller only looks at .category on failure.
    """
    from src.guardrails.pii_detector import detect_pii

    checks = [check_crisis_terms, check_prompt_injection, check_indecent_language, check_length]
    for check in checks:
        result = check(text)
        if not result.passed:
            return result

    pii_matches = detect_pii(text)
    if pii_matches:
        return GuardrailResult(
            passed=True,  # PII in input isn't blocked, just flagged — caller may choose to redact before logging/storage
            category="pii",
            reasons=[f"{m.kind} detected" for m in pii_matches],
            pii_matches=pii_matches,
        )
    return GuardrailResult(passed=True)
