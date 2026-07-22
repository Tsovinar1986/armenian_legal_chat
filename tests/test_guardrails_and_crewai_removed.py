"""Regression tests for the removal of src/guardrails and crewai (the
researcher+writer crews in src/agents/legal_crew.py and
src/agents/therapist_crew.py). Both were deleted outright, along with every
usage site in api.py and src/agents/legal_agent.py — these tests exist so a
future re-add of either doesn't silently reintroduce partial/inconsistent
wiring (e.g. one call site patched but another left importing a module that
no longer has a reason to exist)."""
import unittest
from unittest.mock import MagicMock

import api
from src.agents.legal_agent import LegalAgent


class RemovedModulesTests(unittest.TestCase):
    def test_guardrails_package_is_gone(self):
        with self.assertRaises(ModuleNotFoundError):
            import src.guardrails  # noqa: F401

    def test_legal_crew_module_is_gone(self):
        with self.assertRaises(ModuleNotFoundError):
            import src.agents.legal_crew  # noqa: F401

    def test_therapist_crew_module_is_gone(self):
        with self.assertRaises(ModuleNotFoundError):
            import src.agents.therapist_crew  # noqa: F401


class ApiHasNoGuardrailsOrCrewReferencesTests(unittest.TestCase):
    def test_no_guardrail_manager_symbol(self):
        self.assertFalse(hasattr(api, "GuardrailManager"))
        self.assertFalse(hasattr(api, "_therapist_guardrails"))

    def test_no_crew_model_constant(self):
        self.assertFalse(hasattr(api, "THERAPIST_CREW_MODEL"))

    def test_direct_therapist_answer_helper_exists(self):
        # Replaces run_therapist_crew as the therapist-chat LLM path.
        self.assertTrue(hasattr(api, "_direct_therapist_answer"))
        self.assertTrue(callable(api._direct_therapist_answer))


def _make_bare_agent():
    """Same construction pattern as test_legal_agent_rag_fallback.py — builds
    a LegalAgent without running __init__ (no CSV load, no real Ollama LLM)."""
    agent = LegalAgent.__new__(LegalAgent)
    agent.repo = MagicMock()
    agent.repo.db.similarity_search_with_score.return_value = []
    agent.state = MagicMock()
    agent.classifier = None
    agent.risk_classifier = None
    agent.model_name = "armenia-lawyer-router"
    agent.court_cases = []
    agent.export_service = MagicMock()
    agent.llm = MagicMock()
    return agent


class LegalAgentHasNoGuardrailsAttributeTests(unittest.TestCase):
    def test_no_guardrails_attribute_on_instance(self):
        agent = _make_bare_agent()
        self.assertFalse(hasattr(agent, "guardrails"))

    def test_previously_off_topic_message_is_no_longer_blocked(self):
        """Before removal, a message with no legal/mental-health keyword on
        the first turn of a conversation got short-circuited by
        check_topic_scope with the "off_topic_blocked" template, never
        reaching the RAG pipeline. That check no longer exists, so the same
        message should now fall all the way through to the RAG fallback's
        "no local precedents" response instead (repo.db is mocked to return
        no results above)."""
        agent = _make_bare_agent()
        result = agent.get_advice("What's a good recipe for banana bread?")
        self.assertNotIn("scope", result.lower())
        self.assertIn("չգտնվեցին", result)  # "no_local_precedents" (hy default)


if __name__ == "__main__":
    unittest.main()
