import unittest

from src.services.crisis_detection import detect_crisis_signal


class CrisisDetectionTests(unittest.TestCase):
    def test_detects_armenian_suicide_phrase(self):
        self.assertTrue(detect_crisis_signal("Ես ուզում եմ մեռնել, ամեն ինչ վերջացել է"))

    def test_detects_armenian_self_harm_phrase(self):
        self.assertTrue(detect_crisis_signal("Երեկ սկսեցի մտածել ինքնավնասման մասին"))

    def test_detects_english_phrase(self):
        self.assertTrue(detect_crisis_signal("I don't know what to do, I want to kill myself"))

    def test_is_case_insensitive(self):
        self.assertTrue(detect_crisis_signal("I WANT TO DIE"))

    def test_does_not_flag_normal_legal_question(self):
        self.assertFalse(detect_crisis_signal("Ինչպես կարող եմ ամուսնալուծվել և բաժանել գույքը"))

    def test_does_not_flag_unrelated_short_text(self):
        self.assertFalse(detect_crisis_signal("Բարև, ինչպե՞ս եք"))

    def test_empty_text(self):
        self.assertFalse(detect_crisis_signal(""))
        self.assertFalse(detect_crisis_signal(None))


if __name__ == "__main__":
    unittest.main()
