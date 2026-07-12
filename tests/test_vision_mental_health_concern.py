import unittest

from src.core.state import SystemState
from src.services.vision import LegalVisionService, MENTAL_HEALTH_CONCERN_WINDOW


class MentalHealthConcernTests(unittest.TestCase):
    """Tests for LegalVisionService's rolling-window sad/angry detection, which
    surfaces a soft therapist suggestion (not a diagnosis) via SystemState."""

    def setUp(self):
        self.state = SystemState()
        self.vision = LegalVisionService(self.state)
        self.negative_labels = {"Ծանրը", "Բարկացած"}

    def test_no_concern_before_window_is_full(self):
        for _ in range(MENTAL_HEALTH_CONCERN_WINDOW - 1):
            self.vision._check_mental_health_concern("Ծանրը", self.negative_labels)
        self.assertFalse(self.state.get_mental_health_concern())

    def test_concern_flagged_once_window_is_sustained_negative(self):
        for _ in range(MENTAL_HEALTH_CONCERN_WINDOW):
            self.vision._check_mental_health_concern("Ծանրը", self.negative_labels)
        self.assertTrue(self.state.get_mental_health_concern())
        self.assertTrue(self.state.get_mental_health_suggestion())

    def test_no_concern_for_mostly_happy_window(self):
        for _ in range(MENTAL_HEALTH_CONCERN_WINDOW):
            self.vision._check_mental_health_concern("Երջանիկ", self.negative_labels)
        self.assertFalse(self.state.get_mental_health_concern())
        self.assertEqual(self.state.get_mental_health_suggestion(), "")

    def test_concern_clears_when_mood_recovers(self):
        for _ in range(MENTAL_HEALTH_CONCERN_WINDOW):
            self.vision._check_mental_health_concern("Ծանրը", self.negative_labels)
        self.assertTrue(self.state.get_mental_health_concern())

        for _ in range(MENTAL_HEALTH_CONCERN_WINDOW):
            self.vision._check_mental_health_concern("Երջանիկ", self.negative_labels)
        self.assertFalse(self.state.get_mental_health_concern())

    def test_empty_emotion_is_ignored(self):
        self.vision._check_mental_health_concern("", self.negative_labels)
        self.vision._check_mental_health_concern(None, self.negative_labels)
        self.assertEqual(len(self.vision._recent_emotions), 0)


if __name__ == "__main__":
    unittest.main()
