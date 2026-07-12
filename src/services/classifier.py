# src/services/classifier.py
import os
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Multilingual NLI model used for zero-shot mental-health risk screening (see
# classify_mental_health_risk below). No task-specific training data or fine-tuning
# is required — candidate labels are matched against the text via natural-language
# inference at inference time.
ZERO_SHOT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

MENTAL_HEALTH_LABELS = [
    "suicide or self-harm risk",
    "acute emotional or mental health crisis",
    "general legal question",
    "casual conversation",
]

MENTAL_HEALTH_RISK_LABELS = {
    "suicide or self-harm risk",
    "acute emotional or mental health crisis",
}


def _load_zero_shot_pipeline(model_name: str):
    """Import transformers and build the zero-shot classification pipeline.

    Split out as a standalone function (rather than inlined in the property below)
    so tests can monkeypatch this one call instead of reaching into transformers'
    internal lazy-module machinery.
    """
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=model_name)


class LegalCaseClassifier:
    def __init__(self, data_folder="data", zero_shot_model: str = None):
        self.data_folder = data_folder
        self.past_cases = []
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = None
        self.zero_shot_model_name = zero_shot_model or ZERO_SHOT_MODEL_NAME
        self._zero_shot_classifier = None

        # Ավտոմատ բեռնել տվյալները ստեղծվելու պահին
        self._load_all_prehistories()
        self._train_classifier()

    def _load_all_prehistories(self):
        """Փնտրում և բեռնում է բոլոր prehistory HTML ֆայլերը data պանակից"""
        if not os.path.exists(self.data_folder):
            print(f"⚠️ Data folder not found: {self.data_folder}")
            return

        for file_name in os.listdir(self.data_folder):
            if file_name.startswith("prehistory") and file_name.endswith(".htm"):
                file_path = os.path.join(self.data_folder, file_name)
                self._parse_html(file_path)

    def _parse_html(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        table = soup.find('table')
        if not table:
            return

        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 5:
                self.past_cases.append({
                    'unique_number': cols[0].get_text(strip=True),
                    'judicial_prehistory': cols[1].get_text(strip=True),
                    'civil_case_classifier': cols[2].get_text(strip=True),
                    'lawyer_name': cols[4].get_text(strip=True),
                    'link': f"http://www.datalex.am/?app=AppCaseSearch&case_id={cols[0].get_text(strip=True)}"
                })

    def _train_classifier(self):
        if self.past_cases:
            corpus = [case['judicial_prehistory'] for case in self.past_cases]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            print(f"✅ Classifier: Indexed {len(self.past_cases)} historical cases.")

    @property
    def zero_shot_classifier(self):
        """Lazily load the HuggingFace zero-shot classification pipeline.

        The model (~280MB) is only downloaded/loaded into memory on first actual
        use of classify_mental_health_risk, not at LegalCaseClassifier construction
        time — same lazy-loading pattern used for the vision stack in
        src/services/vision.py, so text-only sessions that never trigger a
        mental-health risk check never pay this memory/import cost.
        """
        if self._zero_shot_classifier is None:
            print(f"🧠 Loading zero-shot classification model ({self.zero_shot_model_name})...")
            self._zero_shot_classifier = _load_zero_shot_pipeline(self.zero_shot_model_name)
        return self._zero_shot_classifier

    def classify_mental_health_risk(self, text: str, risk_threshold: float = 0.55) -> dict:
        """
        Zero-shot classify whether text signals a mental-health crisis (suicide/
        self-harm risk or acute emotional distress) — no labeled training data
        needed, the model matches the text against MENTAL_HEALTH_LABELS via NLI.

        This is a second, heavier-weight signal meant to run alongside the fast
        keyword check in src/services/crisis_detection.py — it exists to catch
        phrasings the keyword list misses, not to replace it. It is not a clinical
        assessment and must never be presented as one.

        Args:
            text: The message text to screen.
            risk_threshold: Minimum confidence on the top label before treating the
                text as a risk signal.

        Returns:
            dict with keys is_risk (bool), label (str), score (float), and
            scores (dict of all candidate labels to their scores), or None if the
            text is empty or classification fails (e.g. model unavailable).
        """
        if not text or not text.strip():
            return None
        try:
            result = self.zero_shot_classifier(text, candidate_labels=MENTAL_HEALTH_LABELS)
        except Exception as e:
            print(f"⚠️ Zero-shot mental-health classification failed: {e}")
            return None

        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        return {
            "is_risk": top_label in MENTAL_HEALTH_RISK_LABELS and top_score >= risk_threshold,
            "label": top_label,
            "score": top_score,
            "scores": dict(zip(result["labels"], (float(s) for s in result["scores"]))),
        }

    def find_similar_case(self, text):
        if self.tfidf_matrix is None or not self.past_cases:
            return None
        new_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(new_vec, self.tfidf_matrix).flatten()
        idx = similarities.argmax()
        if similarities[idx] < 0.15:
            return None
        return self.past_cases[idx]
    
    def find_multiple_similar_cases(self, text, limit: int = 5):
        """
        Find multiple similar cases ranked by similarity score.
        
        Args:
            text: The query text
            limit: Maximum number of cases to return
            
        Returns:
            List of similar case dictionaries ranked by similarity
        """
        if self.tfidf_matrix is None or not self.past_cases:
            return []
        
        try:
            new_vec = self.vectorizer.transform([text])
            similarities = cosine_similarity(new_vec, self.tfidf_matrix).flatten()
            
            # Get indices sorted by similarity (descending)
            sorted_indices = similarities.argsort()[::-1]
            
            # Filter cases with similarity score > 0.1 and limit results
            similar_cases = []
            for idx in sorted_indices:
                if similarities[idx] > 0.1 and len(similar_cases) < limit:
                    case = self.past_cases[idx].copy()
                    case['similarity_score'] = float(similarities[idx])
                    case['is_approved'] = self._is_approved_case(case)
                    similar_cases.append(case)
            
            return similar_cases
        except Exception as e:
            print(f"⚠️ Error finding similar cases: {e}")
            return []
    
    def _is_approved_case(self, case: dict) -> bool:
        """
        Detect if a case is approved/successful based on keywords in the text.
        
        Args:
            case: Case dictionary
            
        Returns:
            True if case appears to be approved/successful
        """
        # Keywords that indicate an approved/successful case
        approved_keywords_hy = [
            'հաստատել', 'հաստատվել', 'հաստատեց', 'հաստատված',
            'հաճախել', 'հաճախվել', 'հաճախեց',
            'մեղադրանք', 'դատել', 'դատվել',
            'նպաստել', 'նպաստել', 'օգտար',
            'ընդունել', 'ընդունվել', 'ընդունեց',
            'հաջողել', 'հաջողվել', 'հաջողեց',
            'բավարար', 'բավարար', 'ճիշտ'
        ]
        
        approved_keywords_en = [
            'approved', 'success', 'successful', 'granted', 'upheld',
            'affirm', 'confirm', 'confirmed', 'valid', 'prevail'
        ]
        
        prehistory = (case.get('judicial_prehistory', '') or '').lower()
        classifier = (case.get('civil_case_classifier', '') or '').lower()
        
        # Check for approved keywords
        for keyword in approved_keywords_hy + approved_keywords_en:
            if keyword in prehistory or keyword in classifier:
                return True
        
        return False
    
    def find_approved_cases(self, limit: int = 10) -> list:
        """
        Find all approved/successful cases from the database.
        
        Args:
            limit: Maximum number of approved cases to return
            
        Returns:
            List of approved case dictionaries
        """
        approved_cases = []
        
        for case in self.past_cases:
            if self._is_approved_case(case):
                approved_cases.append(case)
                if len(approved_cases) >= limit:
                    break
        
        return approved_cases
    
    def find_cases_by_lawyer(self, lawyer_name: str, limit: int = 10) -> list:
        """
        Find cases handled by a specific lawyer.
        
        Args:
            lawyer_name: Name of the lawyer to search for
            limit: Maximum number of cases to return
            
        Returns:
            List of case dictionaries for the lawyer
        """
        lawyer_cases = []
        lawyer_name_lower = lawyer_name.lower()
        
        for case in self.past_cases:
            case_lawyer = (case.get('lawyer_name', '') or '').lower()
            if lawyer_name_lower in case_lawyer or case_lawyer in lawyer_name_lower:
                lawyer_cases.append(case)
                if len(lawyer_cases) >= limit:
                    break
        
        return lawyer_cases
    
    def get_top_lawyer_for_query(self, text: str, search_limit: int = 15) -> dict:
        """
        Among cases similar to the query, find the lawyer with the most approved cases.

        Args:
            text: The query text
            search_limit: How many similar cases to consider when ranking lawyers

        Returns:
            Dict with the best-ranked lawyer's name and case stats, or None if no lawyer found
        """
        similar_cases = self.find_multiple_similar_cases(text, limit=search_limit)
        if not similar_cases:
            return None

        lawyer_stats = {}
        for case in similar_cases:
            lawyer = (case.get('lawyer_name', '') or '').strip()
            if not lawyer or lawyer == "(NULL)":
                continue
            stats = lawyer_stats.setdefault(lawyer, {'total': 0, 'approved': 0, 'cases': []})
            stats['total'] += 1
            stats['cases'].append(case)
            if self._is_approved_case(case):
                stats['approved'] += 1

        if not lawyer_stats:
            return None

        best_name, best_stats = max(
            lawyer_stats.items(),
            key=lambda kv: (kv[1]['approved'], kv[1]['total'])
        )
        return {
            'lawyer_name': best_name,
            'approved_cases': best_stats['approved'],
            'total_similar_cases': best_stats['total'],
            'sample_cases': best_stats['cases'][:3]
        }

    def get_top_lawyers_by_cases(self, limit: int = 10) -> list:
        """
        Get the top lawyers by number of successful cases.
        
        Returns:
            List of tuples (lawyer_name, case_count, sample_cases)
        """
        lawyer_stats = {}
        
        for case in self.past_cases:
            lawyer = case.get('lawyer_name', 'N/A')
            if lawyer and lawyer != "(NULL)":
                if lawyer not in lawyer_stats:
                    lawyer_stats[lawyer] = {
                        'count': 0,
                        'cases': []
                    }
                lawyer_stats[lawyer]['count'] += 1
                # Keep up to 3 sample cases per lawyer
                if len(lawyer_stats[lawyer]['cases']) < 3:
                    lawyer_stats[lawyer]['cases'].append(case)
        
        # Sort by count (descending) and limit results
        sorted_lawyers = sorted(
            lawyer_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:limit]

        return sorted_lawyers


DEFAULT_MENTAL_HEALTH_QA_CSV = "src/data/student_mh_counseling_100k_with_label_column.csv"


class MentalHealthQAClassifier:
    """Retrieval-style Q&A matcher over a labeled counseling-conversation dataset
    (default: src/data/student_mh_counseling_100k_with_label_column.csv, columns
    question/answer/label — labels like "depression", "stress", "seeking help",
    "suicidal thoughts"). Mirrors LegalCaseClassifier's TF-IDF + cosine-similarity
    approach, but over counseling Q&A pairs instead of legal case prehistories.

    This is a supportive-conversation helper, not a diagnosis or a licensed
    therapist substitute — every response using it should say so. It must never
    be used in place of the crisis/safety check in src/services/crisis_detection.py
    and LegalCaseClassifier.classify_mental_health_risk: a real crisis signal
    should always short-circuit to CRISIS_RESPONSE_HY, not to a retrieved answer,
    even though this dataset itself contains "suicidal thoughts"-labeled rows.

    Loading and indexing ~100k rows is comparatively expensive, so — like the
    zero-shot risk model — it only happens lazily, on first actual use.
    """

    def __init__(self, csv_path: str = DEFAULT_MENTAL_HEALTH_QA_CSV, max_rows: int = None):
        self.csv_path = csv_path
        self.max_rows = max_rows
        self.qa_pairs = []
        self.vectorizer = TfidfVectorizer(max_features=8000, stop_words="english")
        self.tfidf_matrix = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._load_csv()
        self._train()
        self._loaded = True

    def _load_csv(self):
        import csv
        if not os.path.exists(self.csv_path):
            print(f"⚠️ Mental-health Q&A dataset not found: {self.csv_path}")
            return
        csv.field_size_limit(10 * 1024 * 1024)
        with open(self.csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if self.max_rows is not None and i >= self.max_rows:
                    break
                question = (row.get('question') or '').strip()
                answer = (row.get('answer') or '').strip()
                label = (row.get('label') or '').strip()
                if question and answer:
                    self.qa_pairs.append({'question': question, 'answer': answer, 'label': label})
        print(f"✅ Mental-health Q&A classifier: loaded {len(self.qa_pairs)} question/answer pairs.")

    def _train(self):
        if self.qa_pairs:
            corpus = [qa['question'] for qa in self.qa_pairs]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            print(f"✅ Mental-health Q&A classifier: indexed {len(self.qa_pairs)} pairs.")

    def find_similar_answer(self, text: str, min_similarity: float = 0.2) -> dict:
        """Find the closest matching question in the dataset and return its answer.

        Returns a dict with question/answer/label/similarity_score, or None if the
        dataset is unavailable or nothing clears min_similarity.
        """
        if not text or not text.strip():
            return None
        self._ensure_loaded()
        if self.tfidf_matrix is None or not self.qa_pairs:
            return None

        new_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(new_vec, self.tfidf_matrix).flatten()
        idx = similarities.argmax()
        if similarities[idx] < min_similarity:
            return None

        match = self.qa_pairs[idx].copy()
        match['similarity_score'] = float(similarities[idx])
        return match