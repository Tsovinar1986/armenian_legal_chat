import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api
from src.db import portal_store


class BookingProviderTypeTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
        self.client = TestClient(api.app)

    def test_booking_defaults_to_lawyer_provider_type(self):
        res = self.client.post("/api/bookings", json={
            "title": "Contract review",
            "client_name": "Alice",
            "lawyer_name": "Bob Lawyer",
            "start_time": "2026-08-01T10:00",
            "role": "individual",
        })
        self.assertTrue(res.json()["success"])
        self.assertEqual(res.json()["booking"]["provider_type"], "lawyer")

    def test_booking_with_therapist_provider_type(self):
        res = self.client.post("/api/bookings", json={
            "title": "Check-in session",
            "client_name": "Alice",
            "lawyer_name": "Dr. Carol Therapist",
            "start_time": "2026-08-02T10:00",
            "role": "individual",
            "provider_type": "therapist",
        })
        self.assertTrue(res.json()["success"])
        self.assertEqual(res.json()["booking"]["provider_type"], "therapist")


class PaymentEndpointTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
        self.client = TestClient(api.app)

    def test_create_intent_fails_cleanly_without_stripe_key(self):
        with patch.object(api, "STRIPE_SECRET_KEY", ""):
            res = self.client.post("/api/payments/create-intent", json={
                "consultation_type": "lawyer",
                "customer_name": "Alice",
                "amount_cents": 5000,
            })
        data = res.json()
        self.assertFalse(data["success"])
        self.assertIn("STRIPE_SECRET_KEY", data["message"])

    def test_create_intent_rejects_invalid_consultation_type(self):
        with patch.object(api, "STRIPE_SECRET_KEY", "sk_test_dummy"):
            res = self.client.post("/api/payments/create-intent", json={
                "consultation_type": "plumber",
                "customer_name": "Alice",
                "amount_cents": 5000,
            })
        self.assertFalse(res.json()["success"])

    def test_create_intent_success_stores_payment(self):
        fake_intent = {"id": "pi_123", "client_secret": "pi_123_secret", "status": "requires_payment_method"}
        with patch.object(api, "STRIPE_SECRET_KEY", "sk_test_dummy"), \
             patch.object(api, "STRIPE_PUBLISHABLE_KEY", "pk_test_dummy"), \
             patch.object(api.stripe.PaymentIntent, "create", return_value=fake_intent) as mock_create:
            res = self.client.post("/api/payments/create-intent", json={
                "consultation_type": "therapist",
                "customer_name": "Alice",
                "amount_cents": 5000,
                "currency": "usd",
            })
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["client_secret"], "pi_123_secret")
        self.assertEqual(data["publishable_key"], "pk_test_dummy")
        mock_create.assert_called_once()

        stored = portal_store.get_payment(data["payment_id"])
        self.assertEqual(stored["stripe_payment_intent_id"], "pi_123")
        self.assertEqual(stored["consultation_type"], "therapist")

    def test_webhook_without_secret_configured(self):
        with patch.object(api, "STRIPE_WEBHOOK_SECRET", ""):
            res = self.client.post("/api/payments/webhook", content=b"{}", headers={"stripe-signature": "x"})
        self.assertFalse(res.json()["success"])


class TherapistChatTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
        api.therapist_chat_sessions.clear()
        self.client = TestClient(api.app)

    def test_crisis_keyword_short_circuits_before_qa_classifier(self):
        with patch.object(api, "get_mental_health_qa_classifier") as mock_get_qa:
            res = self.client.post("/api/therapist-chat", json={"message": "I want to kill myself"})
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("911", data["response"])
        mock_get_qa.assert_not_called()

    def test_non_crisis_message_uses_therapist_crew(self):
        mock_qa = MagicMock()
        with patch.object(api, "get_mental_health_qa_classifier", return_value=mock_qa), \
             patch.object(api, "get_legal_agent", side_effect=RuntimeError("no ollama")), \
             patch("src.agents.therapist_crew.run_therapist_crew", return_value="A warm supportive reply.") as mock_crew:
            res = self.client.post("/api/therapist-chat", json={"message": "I feel stressed about my exams"})
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("A warm supportive reply.", data["response"])
        self.assertIn("not advice from a licensed therapist", data["response"])
        mock_crew.assert_called_once()

    def test_crew_failure_falls_back_to_direct_retrieval(self):
        mock_qa = MagicMock()
        mock_qa.find_similar_answer.return_value = {
            "question": "I feel stressed about exams",
            "answer": "That sounds tough.",
            "label": "stress",
            "similarity_score": 0.8,
        }
        with patch.object(api, "get_mental_health_qa_classifier", return_value=mock_qa), \
             patch.object(api, "get_legal_agent", side_effect=RuntimeError("no ollama")), \
             patch("src.agents.therapist_crew.run_therapist_crew", side_effect=RuntimeError("crew failed")):
            res = self.client.post("/api/therapist-chat", json={"message": "I feel stressed about my exams"})
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("That sounds tough.", data["response"])
        self.assertIn("not advice from a licensed therapist", data["response"])


if __name__ == "__main__":
    unittest.main()
