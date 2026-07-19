import unittest

from fastapi.testclient import TestClient

import api
from src.db import portal_store


class AuthFlowTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
        api.chat_sessions.clear()
        api.therapist_chat_sessions.clear()
        self.client = TestClient(api.app)

    def test_register_login_and_reset_password_for_lawyer_with_phone(self):
        register_response = self.client.post(
            "/api/auth/register",
            json={
                "name": "Lina Lawyer",
                "email": "lina@example.com",
                "phone_number": "+15551234567",
                "password": "initial123",
                "role": "lawyer",
                "license_number": "LIC-1001",
            },
        )
        self.assertEqual(register_response.status_code, 200)
        self.assertTrue(register_response.json()["success"])
        self.assertEqual(register_response.json()["user"]["license_number"], "LIC-1001")

        login_response = self.client.post(
            "/api/auth/login",
            json={
                "identifier": "+15551234567",
                "password": "initial123",
                "role": "lawyer",
            },
        )
        self.assertEqual(login_response.status_code, 200)
        self.assertTrue(login_response.json()["success"])

        forgot_response = self.client.post(
            "/api/auth/forgot-password",
            json={"identifier": "+15551234567", "channel": "phone"},
        )
        self.assertEqual(forgot_response.status_code, 200)
        self.assertTrue(forgot_response.json()["success"])
        otp = forgot_response.json()["otp"]

        reset_response = self.client.post(
            "/api/auth/reset-password",
            json={
                "identifier": "+15551234567",
                "otp": otp,
                "new_password": "newSecure456",
            },
        )
        self.assertEqual(reset_response.status_code, 200)
        self.assertTrue(reset_response.json()["success"])

        login_after_reset = self.client.post(
            "/api/auth/login",
            json={
                "identifier": "+15551234567",
                "password": "newSecure456",
                "role": "lawyer",
            },
        )
        self.assertEqual(login_after_reset.status_code, 200)
        self.assertTrue(login_after_reset.json()["success"])


if __name__ == "__main__":
    unittest.main()
