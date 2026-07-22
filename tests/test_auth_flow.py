import unittest

from fastapi.testclient import TestClient

import api
from src.db import portal_store


class AuthFlowTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
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

    def _register(self, **overrides):
        payload = {
            "name": "Test User",
            "email": "test.user@example.com",
            "phone_number": "",
            "password": "secret123",
            "role": "individual",
            "license_number": None,
        }
        payload.update(overrides)
        return self.client.post("/api/auth/register", json=payload)

    def test_login_wrong_password_fails(self):
        self._register(email="wrongpw@example.com")

        res = self.client.post(
            "/api/auth/login",
            json={"identifier": "wrongpw@example.com", "password": "not-the-password", "role": "individual"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["success"])
        self.assertIn("not found", data["message"].lower())

    def test_login_unknown_account_fails(self):
        res = self.client.post(
            "/api/auth/login",
            json={"identifier": "nobody@example.com", "password": "whatever", "role": "individual"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["success"])

    def test_register_duplicate_email_fails(self):
        self._register(email="dupe@example.com")
        second = self._register(email="dupe@example.com", name="Someone Else")
        self.assertFalse(second.json()["success"])
        self.assertIn("already exists", second.json()["message"].lower())

    def test_login_requires_matching_role(self):
        """Registering as one role and logging in with a different role should
        fail — authenticate_user() looks up by (identifier, role) together, so
        the same email under a different role is treated as a different
        account (relevant now that the frontend picks role via a toggle
        before sign-in rather than a single implicit role)."""
        self._register(email="rolecheck@example.com", role="lawyer", license_number="LIC-2")

        res = self.client.post(
            "/api/auth/login",
            json={"identifier": "rolecheck@example.com", "password": "secret123", "role": "therapist"},
        )
        self.assertFalse(res.json()["success"])

        matching_role_res = self.client.post(
            "/api/auth/login",
            json={"identifier": "rolecheck@example.com", "password": "secret123", "role": "lawyer"},
        )
        self.assertTrue(matching_role_res.json()["success"])

    def test_register_therapist_role(self):
        res = self._register(email="therapist@example.com", role="therapist", license_number=None)
        self.assertTrue(res.json()["success"])
        self.assertEqual(res.json()["user"]["role"], "therapist")

    def test_logout_invalidates_session(self):
        register_res = self._register(email="logout@example.com")
        token = register_res.json()["token"]

        me_before = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertTrue(me_before.json()["success"])

        logout_res = self.client.post("/api/auth/logout", json={"token": token})
        self.assertTrue(logout_res.json()["success"])

        me_after = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertFalse(me_after.json()["success"])

    def test_me_rejects_missing_or_invalid_token(self):
        res = self.client.get("/api/auth/me")
        self.assertFalse(res.json()["success"])

        res_bad_token = self.client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
        self.assertFalse(res_bad_token.json()["success"])


if __name__ == "__main__":
    unittest.main()
