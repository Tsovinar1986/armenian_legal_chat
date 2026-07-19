import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api
from src.db import portal_store


class SessionTokenTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
        self.client = TestClient(api.app)

    def test_register_issues_a_token_that_validates(self):
        res = self.client.post("/api/auth/register", json={
            "name": "Alice", "email": "alice@example.com", "password": "pw123456", "role": "individual",
        })
        data = res.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["token"])

        me = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {data['token']}"})
        me_data = me.json()
        self.assertTrue(me_data["success"])
        self.assertEqual(me_data["user_id"], data["user"]["id"])

    def test_login_issues_a_token(self):
        self.client.post("/api/auth/register", json={
            "name": "Bob", "email": "bob@example.com", "password": "pw123456", "role": "lawyer",
        })
        res = self.client.post("/api/auth/login", json={
            "identifier": "bob@example.com", "password": "pw123456", "role": "lawyer",
        })
        self.assertTrue(res.json()["token"])

    def test_logout_invalidates_token(self):
        register = self.client.post("/api/auth/register", json={
            "name": "Carol", "email": "carol@example.com", "password": "pw123456", "role": "individual",
        })
        token = register.json()["token"]
        self.client.post("/api/auth/logout", json={"token": token})
        me = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertFalse(me.json()["success"])

    def test_missing_or_bad_token_returns_failure_not_error(self):
        res = self.client.get("/api/auth/me")
        self.assertFalse(res.json()["success"])
        res2 = self.client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
        self.assertFalse(res2.json()["success"])


class ChatPersistenceTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
        self.client = TestClient(api.app)

    def test_legal_chat_history_persists_across_requests(self):
        with patch.object(api, "get_legal_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.get_advice.return_value = "some answer"
            mock_get_agent.return_value = mock_agent

            res = self.client.post("/api/chat", json={"message": "Ինչպես կարող եմ ամուսնալուծվել"})
            session_id = res.json()["session_id"]

        history = self.client.get(f"/api/chat/{session_id}")
        messages = history.json()["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "bot")

    def test_chat_history_survives_module_reload_simulation(self):
        # Persistence means a fresh call to get_chat_messages (as if a new
        # process started) still sees prior messages — no in-memory dict involved.
        portal_store.append_chat_message("sess-1", "legal", "user", "hello")
        portal_store.append_chat_message("sess-1", "legal", "bot", "hi there")
        messages = portal_store.get_chat_messages("sess-1", "legal")
        self.assertEqual([m["text"] for m in messages], ["hello", "hi there"])

    def test_legal_and_therapist_histories_are_independent(self):
        portal_store.append_chat_message("shared-id", "legal", "user", "legal question")
        portal_store.append_chat_message("shared-id", "therapist", "user", "therapist message")
        self.assertEqual(len(portal_store.get_chat_messages("shared-id", "legal")), 1)
        self.assertEqual(len(portal_store.get_chat_messages("shared-id", "therapist")), 1)


class PaymentToBookingTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
        self.client = TestClient(api.app)

    def test_payment_without_start_time_does_not_create_booking(self):
        payment = portal_store.create_payment(
            consultation_type="lawyer", customer_name="Alice", amount_cents=5000,
            currency="usd", stripe_payment_intent_id="pi_1", status="succeeded",
            created_at="2026-01-01T00:00:00+00:00",
        )
        result = portal_store.create_booking_from_payment_if_scheduled(payment)
        self.assertIsNone(result)
        self.assertEqual(portal_store.count_bookings(), 0)

    def test_payment_with_start_time_creates_booking(self):
        payment = portal_store.create_payment(
            consultation_type="therapist", customer_name="Alice", amount_cents=5000,
            currency="usd", stripe_payment_intent_id="pi_2", status="succeeded",
            created_at="2026-01-01T00:00:00+00:00",
            provider_name="Dr. Carol", start_time="2026-08-03T10:00:00", timezone="Asia/Yerevan",
        )
        booking = portal_store.create_booking_from_payment_if_scheduled(payment)
        self.assertIsNotNone(booking)
        self.assertEqual(booking["client_name"], "Alice")
        self.assertEqual(booking["lawyer_name"], "Dr. Carol")
        self.assertEqual(booking["provider_type"], "therapist")

        updated_payment = portal_store.get_payment_by_intent("pi_2")
        self.assertEqual(updated_payment["booking_id"], booking["id"])

    def test_duplicate_webhook_does_not_double_book(self):
        payment = portal_store.create_payment(
            consultation_type="lawyer", customer_name="Alice", amount_cents=5000,
            currency="usd", stripe_payment_intent_id="pi_3", status="succeeded",
            created_at="2026-01-01T00:00:00+00:00",
            provider_name="Bob Lawyer", start_time="2026-08-03T10:00:00", timezone="UTC",
        )
        first = portal_store.create_booking_from_payment_if_scheduled(payment)
        self.assertIsNotNone(first)

        payment_again = portal_store.get_payment_by_intent("pi_3")
        second = portal_store.create_booking_from_payment_if_scheduled(payment_again)
        self.assertIsNone(second)
        self.assertEqual(portal_store.count_bookings(), 1)

    def test_webhook_endpoint_triggers_auto_booking(self):
        payment = portal_store.create_payment(
            consultation_type="lawyer", customer_name="Alice", amount_cents=5000,
            currency="usd", stripe_payment_intent_id="pi_4", status="requires_payment_method",
            created_at="2026-01-01T00:00:00+00:00",
            provider_name="Bob Lawyer", start_time="2026-08-03T10:00:00", timezone="UTC",
        )
        fake_event = {
            "type": "payment_intent.succeeded",
            "data": {"object": {"id": "pi_4"}},
        }
        with patch.object(api, "STRIPE_WEBHOOK_SECRET", "whsec_test"), \
             patch.object(api.stripe.Webhook, "construct_event", return_value=fake_event):
            res = self.client.post("/api/payments/webhook", content=b"{}", headers={"stripe-signature": "x"})
        self.assertTrue(res.json()["received"])
        self.assertEqual(portal_store.count_bookings(), 1)


class ProviderScheduleTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
        self.client = TestClient(api.app)

    def test_set_and_get_schedule(self):
        res = self.client.post("/api/providers/schedule", json={
            "provider_name": "Bob Lawyer", "weekday": 0, "start_hour": 10, "end_hour": 14, "timezone": "Asia/Yerevan",
        })
        self.assertTrue(res.json()["success"])

        listed = self.client.get("/api/providers/schedule", params={"provider_name": "Bob Lawyer"})
        schedule = listed.json()["schedule"]
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0]["start_hour"], 10)
        self.assertEqual(schedule[0]["end_hour"], 14)

    def test_availability_uses_configured_schedule_instead_of_default(self):
        # 2026-08-03 is a Monday -> weekday 0
        self.client.post("/api/providers/schedule", json={
            "provider_name": "Bob Lawyer", "weekday": 0, "start_hour": 10, "end_hour": 12, "timezone": "UTC",
        })
        res = self.client.get("/api/bookings/availability", params={
            "provider_name": "Bob Lawyer", "date": "2026-08-03", "timezone": "UTC",
        })
        data = res.json()
        self.assertTrue(data["success"])
        # 10:00-12:00 in 60-minute slots = 2 slots, not the default 9 (09:00-18:00)
        self.assertEqual(len(data["slots"]), 2)
        self.assertEqual(data["slots"][0]["local_start"][:16], "2026-08-03T10:00")

    def test_explicit_query_params_override_configured_schedule(self):
        self.client.post("/api/providers/schedule", json={
            "provider_name": "Bob Lawyer", "weekday": 0, "start_hour": 10, "end_hour": 12, "timezone": "UTC",
        })
        res = self.client.get("/api/bookings/availability", params={
            "provider_name": "Bob Lawyer", "date": "2026-08-03", "timezone": "UTC",
            "start_hour": 9, "end_hour": 18,
        })
        self.assertEqual(len(res.json()["slots"]), 9)

    def test_no_schedule_falls_back_to_default_hours(self):
        res = self.client.get("/api/bookings/availability", params={
            "provider_name": "Nobody Configured", "date": "2026-08-03", "timezone": "UTC",
        })
        self.assertEqual(len(res.json()["slots"]), 9)


if __name__ == "__main__":
    unittest.main()
