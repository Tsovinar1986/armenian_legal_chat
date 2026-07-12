import unittest
from unittest.mock import MagicMock, patch

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


class ZeroShotMentalHealthRiskTests(unittest.TestCase):
    """Tests for LegalCaseClassifier.classify_mental_health_risk, with the
    HuggingFace pipeline mocked so these run fast and without a model download."""

    def _make_classifier(self):
        from src.services.classifier import LegalCaseClassifier
        clf = LegalCaseClassifier.__new__(LegalCaseClassifier)
        clf._zero_shot_classifier = None
        clf.zero_shot_model_name = "mock-model"
        return clf

    def test_empty_text_returns_none(self):
        clf = self._make_classifier()
        self.assertIsNone(clf.classify_mental_health_risk(""))
        self.assertIsNone(clf.classify_mental_health_risk(None))

    def test_flags_high_confidence_risk_label(self):
        clf = self._make_classifier()
        mock_pipeline = MagicMock(return_value={
            "labels": ["suicide or self-harm risk", "casual conversation",
                       "general legal question", "acute emotional or mental health crisis"],
            "scores": [0.91, 0.04, 0.03, 0.02],
        })
        clf._zero_shot_classifier = mock_pipeline
        result = clf.classify_mental_health_risk("some ambiguous text")
        self.assertTrue(result["is_risk"])
        self.assertEqual(result["label"], "suicide or self-harm risk")
        self.assertAlmostEqual(result["score"], 0.91)
        mock_pipeline.assert_called_once()

    def test_does_not_flag_low_confidence_risk_label(self):
        clf = self._make_classifier()
        clf._zero_shot_classifier = MagicMock(return_value={
            "labels": ["suicide or self-harm risk", "general legal question",
                       "casual conversation", "acute emotional or mental health crisis"],
            "scores": [0.3, 0.28, 0.22, 0.2],
        })
        result = clf.classify_mental_health_risk("some ambiguous text")
        self.assertFalse(result["is_risk"])

    def test_does_not_flag_non_risk_top_label(self):
        clf = self._make_classifier()
        clf._zero_shot_classifier = MagicMock(return_value={
            "labels": ["general legal question", "casual conversation",
                       "suicide or self-harm risk", "acute emotional or mental health crisis"],
            "scores": [0.8, 0.1, 0.06, 0.04],
        })
        result = clf.classify_mental_health_risk("Ինչպես կարող եմ ամուսնալուծվել")
        self.assertFalse(result["is_risk"])
        self.assertEqual(result["label"], "general legal question")

    def test_pipeline_failure_returns_none(self):
        clf = self._make_classifier()
        clf._zero_shot_classifier = MagicMock(side_effect=RuntimeError("model unavailable"))
        self.assertIsNone(clf.classify_mental_health_risk("some text"))

    def test_zero_shot_classifier_property_lazily_loads_pipeline(self):
        clf = self._make_classifier()
        fake_loader = MagicMock(return_value="pipeline-instance")
        with patch("src.services.classifier._load_zero_shot_pipeline", fake_loader):
            first = clf.zero_shot_classifier
            second = clf.zero_shot_classifier
        self.assertEqual(first, "pipeline-instance")
        self.assertIs(first, second)
        fake_loader.assert_called_once_with("mock-model")


if __name__ == "__main__":
    unittest.main()
