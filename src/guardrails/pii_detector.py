"""Regex-based PII (personally identifiable information) detection.

Deliberately not ML-based — same reasoning as crisis_detection.py's keyword
approach: pattern matching stays auditable and predictable. Covers the PII
categories most likely to show up in a legal/therapist chat: email addresses,
phone numbers, card numbers, and a few government-ID-shaped number patterns.
This is a heuristic net, not a certified PII scanner — it will miss unusual
formats and can false-positive on structured numbers that aren't actually PII
(e.g. a case number that happens to look like a phone number).
"""
import re

from src.guardrails.schemas import PIIMatch

_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    # Phone: sequences of 7+ digits, optionally grouped with spaces/dashes/parens,
    # optionally with a leading +. Loose on purpose to catch Armenian (+374) and
    # generic international formats.
    "phone": re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{2,4}[\s-]?\d{2,4}[\s-]?\d{2,4}(?:[\s-]?\d{2,4})?"),
    "card_number": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def _looks_like_card(candidate: str) -> bool:
    digits = re.sub(r"[^\d]", "", candidate)
    return 13 <= len(digits) <= 19


def detect_pii(text: str) -> list:
    """Return a list of PIIMatch for every PII-shaped substring found in text."""
    if not text:
        return []

    matches = []
    for kind, pattern in _PATTERNS.items():
        for m in pattern.finditer(text):
            value = m.group(0)
            if kind == "card_number" and not _looks_like_card(value):
                continue
            if kind == "phone" and len(re.sub(r"[^\d]", "", value)) < 7:
                continue
            matches.append(PIIMatch(kind=kind, value=value, start=m.start(), end=m.end()))

    # card_number and phone patterns can both match the same digit run —
    # prefer the more specific card_number match when spans overlap.
    matches.sort(key=lambda m: m.start)
    deduped = []
    for match in matches:
        if deduped and match.start < deduped[-1].end:
            if match.kind == "card_number" and deduped[-1].kind == "phone":
                deduped[-1] = match
            continue
        deduped.append(match)
    return deduped


def redact_pii(text: str, matches: list = None) -> str:
    """Replace each detected PII span with a [REDACTED_<KIND>] placeholder."""
    if not text:
        return text
    if matches is None:
        matches = detect_pii(text)
    if not matches:
        return text

    result = []
    cursor = 0
    for match in sorted(matches, key=lambda m: m.start):
        result.append(text[cursor:match.start])
        result.append(f"[REDACTED_{match.kind.upper()}]")
        cursor = match.end
    result.append(text[cursor:])
    return "".join(result)
