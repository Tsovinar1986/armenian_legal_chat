import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api
from src.db import portal_store
from src.services.crisis_detection import (
    CRISIS_RESPONSE_EN,
    CRISIS_RESPONSE_HY,
    get_crisis_response,
)


class GetCrisisResponseTests(unittest.TestCase):
    def test_hy_returns_armenian_text(self):
        self.assertEqual(get_crisis_response("hy"), CRISIS_RESPONSE_HY)

    def test_en_returns_english_text(self):
        self.assertEqual(get_crisis_response("en"), CRISIS_RESPONSE_EN)

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(get_crisis_response("fr"), CRISIS_RESPONSE_EN)

    def test_default_is_armenian(self):
        self.assertEqual(get_crisis_response(), CRISIS_RESPONSE_HY)


class ChatLanguageParamTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
        self.client = TestClient(api.app)

    def test_legal_chat_crisis_response_respects_language(self):
        with patch.object(api, "get_legal_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.get_advice.return_value = CRISIS_RESPONSE_EN
            mock_get_agent.return_value = mock_agent

            res = self.client.post("/api/chat", json={"message": "I want to kill myself", "language": "en"})
            self.assertTrue(res.json()["success"])
            mock_agent.get_advice.assert_called_once()
            _, kwargs = mock_agent.get_advice.call_args
            self.assertEqual(kwargs["language"], "en")

    def test_legal_chat_defaults_to_armenian(self):
        with patch.object(api, "get_legal_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.get_advice.return_value = "some answer"
            mock_get_agent.return_value = mock_agent

            self.client.post("/api/chat", json={"message": "Ինչպես կարող եմ ամուսնալուծվել"})
            _, kwargs = mock_agent.get_advice.call_args
            self.assertEqual(kwargs["language"], "hy")

    def test_therapist_chat_crisis_response_respects_language(self):
        res = self.client.post("/api/therapist-chat", json={"message": "I want to kill myself", "language": "hy"})
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["response"], CRISIS_RESPONSE_HY)

    def test_therapist_chat_crisis_response_defaults_to_english(self):
        res = self.client.post("/api/therapist-chat", json={"message": "I want to kill myself"})
        data = res.json()
        self.assertEqual(data["response"], CRISIS_RESPONSE_EN)

    def test_therapist_chat_passes_language_to_direct_llm(self):
        mock_qa = MagicMock()
        mock_qa.find_similar_answer.return_value = None
        mock_agent = MagicMock()
        mock_agent.risk_classifier.classify_mental_health_risk.return_value = {"is_risk": False}
        mock_agent.llm.invoke.return_value = "reply"
        with patch.object(api, "get_mental_health_qa_classifier", return_value=mock_qa), \
             patch.object(api, "get_legal_agent", return_value=mock_agent), \
             patch("api._direct_therapist_answer", return_value="reply") as mock_direct_answer:
            self.client.post("/api/therapist-chat", json={"message": "I feel stressed", "language": "hy"})
        args, _ = mock_direct_answer.call_args
        self.assertEqual(args[-1], "hy")


if __name__ == "__main__":
    unittest.main()
