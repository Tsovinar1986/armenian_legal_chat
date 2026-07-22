import unittest
from unittest.mock import MagicMock, patch

from src.agents.legal_agent import LegalAgent


def _make_agent(classifier=None):
    """Build a LegalAgent without running __init__ (which loads a 142MB CSV and
    a real Ollama LLM) — same pattern used to verify crisis-detection earlier."""
    agent = LegalAgent.__new__(LegalAgent)
    agent.repo = MagicMock()
    agent.state = MagicMock()
    agent.classifier = classifier
    agent.model_name = "armenia-lawyer-router"
    agent.court_cases = []
    agent.export_service = MagicMock()
    agent.llm = MagicMock()  # only self.llm.invoke(...) is ever called
    return agent


class LegalAgentRagFallbackTests(unittest.TestCase):
    def test_generate_rag_response_uses_direct_llm(self):
        agent = _make_agent()
        doc = MagicMock()
        doc.page_content = "Relevant precedent text."
        agent.repo.db.similarity_search_with_score.return_value = [(doc, 0.2)]
        agent.llm.invoke.return_value = "Direct LLM-drafted Armenian answer."

        result = agent._generate_rag_response("Ինչպես կարող եմ ամուսնալուծվել")

        self.assertIn("Direct LLM-drafted Armenian answer.", result)
        agent.llm.invoke.assert_called_once()
        (prompt,), _ = agent.llm.invoke.call_args
        self.assertIn("Relevant precedent text.", prompt)

    def test_llm_failure_returns_unavailable_template(self):
        agent = _make_agent()
        doc = MagicMock()
        doc.page_content = "Relevant precedent text."
        agent.repo.db.similarity_search_with_score.return_value = [(doc, 0.2)]
        agent.llm.invoke.side_effect = RuntimeError("llm failed")

        result = agent._generate_rag_response("Ինչպես կարող եմ ամուսնալուծվել")

        self.assertIn("ժամանակավորապես անհասանելի", result)

    def test_no_context_returns_no_results_message_without_calling_llm(self):
        agent = _make_agent()
        agent.repo.db.similarity_search_with_score.return_value = []

        result = agent._generate_rag_response("Ինչպես կարող եմ ամուսնալուծվել")

        agent.llm.invoke.assert_not_called()
        self.assertIn("չգտնվեցին", result)


if __name__ == "__main__":
    unittest.main()
