import unittest

from fastapi.testclient import TestClient

import main
from src.db import portal_store


class StartTimeToUtcTests(unittest.TestCase):
    def test_naive_start_time_interpreted_in_given_timezone(self):
        result = portal_store.start_time_to_utc_iso("2026-08-03T10:00:00", "Asia/Yerevan")
        self.assertEqual(result, "2026-08-03T06:00:00+00:00")

    def test_start_time_with_explicit_offset_is_trusted(self):
        result = portal_store.start_time_to_utc_iso("2026-08-03T10:00:00+02:00", "Asia/Yerevan")
        self.assertEqual(result, "2026-08-03T08:00:00+00:00")

    def test_unparseable_start_time_returns_none(self):
        self.assertIsNone(portal_store.start_time_to_utc_iso("not-a-date", "UTC"))

    def test_unknown_timezone_falls_back_to_utc(self):
        result = portal_store.start_time_to_utc_iso("2026-08-03T10:00:00", "Not/ARealZone")
        self.assertEqual(result, "2026-08-03T10:00:00+00:00")


class BookingAvailabilityEndpointTests(unittest.TestCase):
    def setUp(self):
        portal_store.clear_all()
        self.client = TestClient(main.app)

    def _book(self, start_time, timezone="UTC", lawyer_name="Bob Lawyer"):
        return self.client.post("/api/bookings", json={
            "title": "Consult",
            "client_name": "Alice",
            "lawyer_name": lawyer_name,
            "start_time": start_time,
            "role": "individual",
            "timezone": timezone,
        })

    def test_booking_stores_start_time_utc(self):
        res = self._book("2026-08-03T10:00:00", timezone="Asia/Yerevan")
        self.assertEqual(res.json()["booking"]["start_time_utc"], "2026-08-03T06:00:00+00:00")

    def test_booked_slot_is_not_free(self):
        self._book("2026-08-03T10:00:00", timezone="Asia/Yerevan")
        res = self.client.get("/api/bookings/availability", params={
            "provider_name": "Bob Lawyer", "date": "2026-08-03", "timezone": "Asia/Yerevan",
        })
        data = res.json()
        self.assertTrue(data["success"])
        ten_am_slot = next(s for s in data["slots"] if s["local_start"].startswith("2026-08-03T10:00"))
        self.assertFalse(ten_am_slot["is_free"])
        nine_am_slot = next(s for s in data["slots"] if s["local_start"].startswith("2026-08-03T09:00"))
        self.assertTrue(nine_am_slot["is_free"])

    def test_slots_include_both_local_and_utc(self):
        res = self.client.get("/api/bookings/availability", params={
            "provider_name": "Nobody", "date": "2026-08-03", "timezone": "Asia/Yerevan",
        })
        data = res.json()
        first = data["slots"][0]
        self.assertEqual(first["local_start"], "2026-08-03T09:00:00+04:00")
        self.assertEqual(first["utc_start"], "2026-08-03T05:00:00+00:00")

    def test_default_business_hours_produce_nine_hourly_slots(self):
        res = self.client.get("/api/bookings/availability", params={
            "provider_name": "Nobody", "date": "2026-08-03", "timezone": "UTC",
        })
        self.assertEqual(len(res.json()["slots"]), 9)  # 09:00-18:00, 1h slots

    def test_unknown_timezone_returns_error(self):
        res = self.client.get("/api/bookings/availability", params={
            "provider_name": "Nobody", "date": "2026-08-03", "timezone": "Not/ARealZone",
        })
        self.assertFalse(res.json()["success"])

    def test_bad_date_format_returns_error(self):
        res = self.client.get("/api/bookings/availability", params={
            "provider_name": "Nobody", "date": "08/03/2026", "timezone": "UTC",
        })
        self.assertFalse(res.json()["success"])

    def test_a_booking_in_another_timezone_does_not_block_unrelated_provider(self):
        self._book("2026-08-03T10:00:00", timezone="Asia/Yerevan", lawyer_name="Someone Else")
        res = self.client.get("/api/bookings/availability", params={
            "provider_name": "Bob Lawyer", "date": "2026-08-03", "timezone": "Asia/Yerevan",
        })
        ten_am_slot = next(s for s in res.json()["slots"] if s["local_start"].startswith("2026-08-03T10:00"))
        self.assertTrue(ten_am_slot["is_free"])


if __name__ == "__main__":
    unittest.main()
