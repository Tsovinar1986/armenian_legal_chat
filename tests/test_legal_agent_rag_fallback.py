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
    agent.llm = MagicMock()  # only truthiness is checked before the crew call
    return agent


class LegalAgentRagFallbackCrewTests(unittest.TestCase):
    def test_generate_rag_response_uses_legal_crew(self):
        agent = _make_agent()
        doc = MagicMock()
        doc.page_content = "Relevant precedent text."
        agent.repo.db.similarity_search_with_score.return_value = [(doc, 0.2)]

        with patch("src.agents.legal_crew.run_legal_crew", return_value="Crew-drafted Armenian answer.") as mock_crew:
            result = agent._generate_rag_response("Ինչպես կարող եմ ամուսնալուծվել")

        self.assertIn("Crew-drafted Armenian answer.", result)
        mock_crew.assert_called_once()
        _, kwargs = mock_crew.call_args
        self.assertIn("Relevant precedent text.", kwargs["context"])

    def test_crew_failure_falls_back_to_template_response(self):
        agent = _make_agent()
        doc = MagicMock()
        doc.page_content = "Relevant precedent text."
        agent.repo.db.similarity_search_with_score.return_value = [(doc, 0.2)]

        with patch("src.agents.legal_crew.run_legal_crew", side_effect=RuntimeError("crew failed")):
            result = agent._generate_rag_response("Ինչպես կարող եմ ամուսնալուծվել")

        self.assertIn("սինթեզված իրավաբանական խորհրդատվություն", result)

    def test_no_context_returns_no_results_message_without_calling_crew(self):
        agent = _make_agent()
        agent.repo.db.similarity_search_with_score.return_value = []

        with patch("src.agents.legal_crew.run_legal_crew") as mock_crew:
            result = agent._generate_rag_response("Ինչպես կարող եմ ամուսնալուծվել")

        mock_crew.assert_not_called()
        self.assertIn("չգտնվեցին", result)


if __name__ == "__main__":
    unittest.main()
