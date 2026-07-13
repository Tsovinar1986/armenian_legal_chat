import unittest
from unittest.mock import MagicMock, patch

from crewai import Crew

from src.agents.therapist_crew import run_therapist_crew


class TherapistCrewTests(unittest.TestCase):
    """Crew.kickoff is mocked so these run fast and without a live Ollama call."""

    def _make_qa_classifier(self, match=None):
        qa = MagicMock()
        qa.find_similar_answer.return_value = match
        return qa

    def test_returns_crew_kickoff_result_as_string(self):
        qa = self._make_qa_classifier({
            "question": "I feel stressed about exams",
            "answer": "That sounds tough.",
            "label": "stress",
        })
        with patch.object(Crew, "kickoff", return_value="A warm supportive reply.") as mock_kickoff:
            result = run_therapist_crew("I feel stressed", qa, "armenia-lawyer-router")
        self.assertEqual(result, "A warm supportive reply.")
        mock_kickoff.assert_called_once()
        qa.find_similar_answer.assert_called_once_with("I feel stressed")

    def test_handles_no_match_found(self):
        qa = self._make_qa_classifier(match=None)
        with patch.object(Crew, "kickoff", return_value="A generic supportive reply."):
            result = run_therapist_crew("something obscure", qa, "armenia-lawyer-router")
        self.assertEqual(result, "A generic supportive reply.")

    def test_researcher_agent_has_no_tools(self):
        """The researcher agent must not be given tools — the local Ollama model
        (armenia-lawyer-router) returns a 400 "does not support tools" error if
        a crewai Agent is given any tools=[...]."""
        qa = self._make_qa_classifier({"question": "q", "answer": "a", "label": "stress"})
        captured = {}

        def fake_kickoff(self):
            captured["agents"] = list(self.agents)
            return "ok"

        with patch.object(Crew, "kickoff", fake_kickoff):
            run_therapist_crew("test message", qa, "armenia-lawyer-router")
        for agent in captured["agents"]:
            self.assertFalse(agent.tools, f"Agent {agent.role} must not have tools")


if __name__ == "__main__":
    unittest.main()
