"""Small, shared, dependency-free validation helpers used across the
guardrails package. Kept separate from pii_detector.py's regexes since these
are generic format checks, not PII scanning."""
import re

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MAX_MESSAGE_LENGTH = 8000


def is_valid_email_format(value: str) -> bool:
    return bool(value) and bool(_EMAIL_RE.match(value.strip()))


def is_within_length_limit(text: str, max_length: int = _MAX_MESSAGE_LENGTH) -> bool:
    return bool(text) and len(text) <= max_length


def is_non_empty(text: str) -> bool:
    return bool(text) and bool(text.strip())


def load_phrase_list(path: str) -> list:
    """Load a text data file of one phrase/pattern per line, skipping blank
    lines and lines starting with '#'. Used by input/output guardrails to
    load crisis_terms.txt, indecent_terms.txt, prompt_injection_patterns.txt."""
    phrases = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    phrases.append(line.lower())
    except FileNotFoundError:
        pass
    return phrases
