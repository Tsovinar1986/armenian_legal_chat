import csv
import os
import tempfile
import unittest

from src.services.classifier import MentalHealthQAClassifier


class MentalHealthQAClassifierTests(unittest.TestCase):
    """Tests against a small temp CSV (not the real 100k-row dataset) so these
    run fast and don't depend on a large data file being present."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        )
        writer = csv.writer(self._tmp)
        writer.writerow(["question", "answer", "label"])
        writer.writerow(["I feel very stressed about my exams", "That sounds tough, tell me more about the exam stress.", "stress"])
        writer.writerow(["I am so happy today, everything is going well", "That's wonderful to hear!", "positive mental state"])
        writer.writerow(["I don't know who to ask for help with my problems", "Reaching out is a great first step. What kind of help are you looking for?", "seeking help"])
        self._tmp.close()
        self.classifier = MentalHealthQAClassifier(csv_path=self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_finds_similar_question(self):
        result = self.classifier.find_similar_answer("I feel stressed about my upcoming exams")
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "stress")
        self.assertIn("exam stress", result["answer"])

    def test_lazy_loading_only_happens_once(self):
        self.assertFalse(self.classifier._loaded)
        self.classifier.find_similar_answer("I feel stressed")
        self.assertTrue(self.classifier._loaded)
        pairs_after_first_call = len(self.classifier.qa_pairs)
        self.classifier.find_similar_answer("I am happy")
        self.assertEqual(len(self.classifier.qa_pairs), pairs_after_first_call)

    def test_empty_text_returns_none(self):
        self.assertIsNone(self.classifier.find_similar_answer(""))
        self.assertIsNone(self.classifier.find_similar_answer(None))

    def test_low_similarity_returns_none(self):
        result = self.classifier.find_similar_answer("completely unrelated text about spaceships and rockets", min_similarity=0.9)
        self.assertIsNone(result)

    def test_missing_csv_file_returns_none_without_raising(self):
        clf = MentalHealthQAClassifier(csv_path="/nonexistent/path/does_not_exist.csv")
        result = clf.find_similar_answer("I feel stressed")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
