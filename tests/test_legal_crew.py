import unittest
from unittest.mock import patch

from crewai import Crew

from src.agents.legal_crew import run_legal_crew


class LegalCrewTests(unittest.TestCase):
    """Crew.kickoff is mocked so these run fast and without a live Ollama call —
    see the real end-to-end smoke test performed manually against Ollama, which
    is what caught the "model does not support tools" issue these agents are
    deliberately built to avoid (no tools=[...] on the researcher agent)."""

    def test_returns_crew_kickoff_result_as_string(self):
        with patch.object(Crew, "kickoff", return_value="Final Armenian answer.") as mock_kickoff:
            result = run_legal_crew(
                query="Ինչպես կարող եմ ամուսնալուծվել",
                context="Case A: divorce granted.",
                cases_context="Example 1: Case 123.",
                conversation_context="Սա այս զրույցի առաջին հարցն է։",
                model_name="armenia-lawyer-router",
            )
        self.assertEqual(result, "Final Armenian answer.")
        mock_kickoff.assert_called_once()

    def test_researcher_agent_has_no_tools(self):
        """The researcher agent must not be given tools — the local Ollama model
        (armenia-lawyer-router) returns a 400 "does not support tools" error if
        a crewai Agent is given any tools=[...]."""
        captured = {}

        def fake_kickoff(self):
            captured["agents"] = list(self.agents)
            return "ok"

        with patch.object(Crew, "kickoff", fake_kickoff):
            run_legal_crew(
                query="test query",
                context="ctx",
                cases_context="cases",
                conversation_context="history",
                model_name="armenia-lawyer-router",
            )
        for agent in captured["agents"]:
            self.assertFalse(agent.tools, f"Agent {agent.role} must not have tools")


if __name__ == "__main__":
    unittest.main()
