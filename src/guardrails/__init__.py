"""Input/output safety guardrails for the legal and therapist AI chat domains.

Distinct from src/services/crisis_detection.py and
LegalCaseClassifier.classify_mental_health_risk (the crisis/self-harm safety
net, which stays exactly as-is and always runs first) — this package adds
complementary checks that don't exist elsewhere: PII leakage, prompt-injection
attempts, indecent language, and (for RAG output) basic groundedness against
retrieved context.

See guardrail_manager.GuardrailManager for the entry point used by callers.
"""
from src.guardrails.guardrail_manager import GuardrailManager
from src.guardrails.schemas import GuardrailResult

__all__ = ["GuardrailManager", "GuardrailResult"]
