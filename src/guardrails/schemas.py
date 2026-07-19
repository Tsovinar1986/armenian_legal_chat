"""Shared result types for the guardrails package."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class PIIMatch:
    kind: str  # "email", "phone", "card_number", "ssn_like", "ip_address"
    value: str
    start: int
    end: int


@dataclass
class GuardrailResult:
    """Outcome of running one or more guardrail checks over a piece of text.

    passed=False means at least one check flagged the text; category names
    *why* (e.g. "pii", "prompt_injection", "indecent_language",
    "rag_ungrounded"); reasons has one human-readable line per flagged check.
    A guardrail failing does not necessarily mean the text should be blocked
    outright — callers decide what to do per category (e.g. PII in input might
    just get redacted before use, while prompt injection in input should
    probably be blocked).
    """

    passed: bool
    category: str = ""
    reasons: List[str] = field(default_factory=list)
    pii_matches: List[PIIMatch] = field(default_factory=list)
    redacted_text: str = ""

    def __bool__(self) -> bool:
        return self.passed
