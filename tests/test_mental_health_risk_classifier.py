import csv
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.services.classifier import MentalHealthRiskClassifier


def _write_csv(rows):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8")
    writer = csv.writer(tmp)
    writer.writerow(["question", "answer", "label"])
    for q, a, label in rows:
        writer.writerow([q, a, label])
    tmp.close()
    return tmp.name


class MentalHealthRiskClassifierLoadingTests(unittest.TestCase):
    def test_filters_unknown_labels_and_respects_sample_size(self):
        path = _write_csv([
            ("I feel sad", "a1", "depression"),
            ("garbled row", "a2", "like a dark cloud covers me"),  # not a known label
            ("exam stress", "a3", "stress"),
            ("thinking about ending it", "a4", "suicidal thoughts"),
        ])
        self.addCleanup(os.unlink, path)
        clf = MentalHealthRiskClassifier(csv_path=path, sample_size=2)
        texts, labels = clf._load_training_sample()
        self.assertEqual(len(texts), 2)  # capped by sample_size
        self.assertTrue(all(l in {"depression", "stress", "suicidal thoughts"} for l in labels))

    def test_missing_csv_returns_empty(self):
        clf = MentalHealthRiskClassifier(csv_path="/nonexistent/path.csv")
        texts, labels = clf._load_training_sample()
        self.assertEqual(texts, [])
        self.assertEqual(labels, [])


class MentalHealthRiskClassifierInferenceTests(unittest.TestCase):
    """Injects a fake trained pipeline directly (bypassing real ML training) to
    test the is_risk decision logic in isolation."""

    def _make_trained_classifier(self, label_probs: dict):
        clf = MentalHealthRiskClassifier.__new__(MentalHealthRiskClassifier)
        clf._trained = True
        labels = list(label_probs.keys())
        probs = list(label_probs.values())

        clf._embedder = MagicMock()
        clf._embedder.encode.return_value = np.array([[0.0] * 5])
        clf._pca = MagicMock()
        clf._pca.transform.return_value = np.array([[0.0] * 3])
        clf._label_encoder = MagicMock()
        clf._label_encoder.classes_ = labels
        clf._rf = MagicMock()
        clf._rf.predict_proba.return_value = np.array([probs])
        return clf

    def test_empty_text_returns_none(self):
        clf = MentalHealthRiskClassifier.__new__(MentalHealthRiskClassifier)
        clf._trained = True
        clf._rf = None
        self.assertIsNone(clf.classify_mental_health_risk(""))
        self.assertIsNone(clf.classify_mental_health_risk(None))

    def test_untrained_classifier_returns_none(self):
        # e.g. dataset missing — _ensure_trained sets _trained=True but leaves _rf None
        clf = MentalHealthRiskClassifier.__new__(MentalHealthRiskClassifier)
        clf._trained = True
        clf._rf = None
        self.assertIsNone(clf.classify_mental_health_risk("some text"))

    def test_flags_high_probability_risk_label(self):
        clf = self._make_trained_classifier({
            "suicidal thoughts": 0.6, "depression": 0.3, "stress": 0.1,
        })
        result = clf.classify_mental_health_risk("some ambiguous text", risk_threshold=0.3)
        self.assertTrue(result["is_risk"])
        self.assertEqual(result["label"], "suicidal thoughts")

    def test_does_not_flag_low_probability_risk_label(self):
        clf = self._make_trained_classifier({
            "depression": 0.5, "stress": 0.3, "suicidal thoughts": 0.2,
        })
        result = clf.classify_mental_health_risk("some ambiguous text", risk_threshold=0.3)
        self.assertFalse(result["is_risk"])
        self.assertEqual(result["label"], "depression")

    def test_flags_risk_even_when_not_top_label(self):
        # risk_threshold is checked independently of argmax — e.g. suicidal
        # thoughts at 0.35 should still flag even though depression (0.4) wins.
        clf = self._make_trained_classifier({
            "depression": 0.4, "suicidal thoughts": 0.35, "stress": 0.25,
        })
        result = clf.classify_mental_health_risk("some ambiguous text", risk_threshold=0.3)
        self.assertTrue(result["is_risk"])
        self.assertEqual(result["label"], "depression")  # top label still reported correctly

    def test_prediction_failure_returns_none(self):
        clf = self._make_trained_classifier({"depression": 1.0})
        clf._rf.predict_proba.side_effect = RuntimeError("boom")
        self.assertIsNone(clf.classify_mental_health_risk("some text"))


class MentalHealthRiskClassifierLazyTrainingTests(unittest.TestCase):
    def test_trains_once_and_caches(self):
        path = _write_csv([
            ("I feel really sad and hopeless", "a1", "depression"),
            ("I am so stressed about my exams", "a2", "stress"),
            ("I think about ending it all", "a3", "suicidal thoughts"),
            ("I need someone to talk to", "a4", "seeking help"),
        ])
        self.addCleanup(os.unlink, path)
        clf = MentalHealthRiskClassifier(csv_path=path, sample_size=10, pca_components=2)

        fake_model = MagicMock()
        fake_model.encode.return_value = np.random.RandomState(0).rand(4, 8)
        with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_st:
            clf._ensure_trained()
            clf._ensure_trained()  # second call must be a no-op

        mock_st.assert_called_once()
        self.assertTrue(clf._trained)
        self.assertIsNotNone(clf._rf)
        self.assertEqual(set(clf._label_encoder.classes_), {"depression", "stress", "suicidal thoughts", "seeking help"})


if __name__ == "__main__":
    unittest.main()
