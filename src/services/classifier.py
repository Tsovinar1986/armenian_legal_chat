# src/services/classifier.py
import os
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LegalCaseClassifier:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        self.past_cases = []
        # stop_words="english": the corpus is almost entirely Armenian, but a
        # handful of cases embed English boilerplate (foreign-party names,
        # contract text). Without this, a common English word like "and" is
        # *rare* corpus-wide and gets a high IDF weight, so a single accidental
        # token overlap with an English-language query can outweigh everything
        # else and win the argmax — e.g. "divorce ... and property issue"
        # matched a procurement case on the sole shared token "and". Stripping
        # English stopwords means English-language queries no longer get a
        # spurious high-confidence match against unrelated Armenian cases;
        # they correctly fall through to the vector-DB/RAG steps instead.
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.tfidf_matrix = None

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
                    'lawyer_name': self._clean_lawyer_names(cols[4].get_text(strip=True)),
                    'link': f"http://www.datalex.am/?app=AppCaseSearch&case_id={cols[0].get_text(strip=True)}"
                })

    @staticmethod
    def _clean_lawyer_names(raw: str) -> str:
        """The DataLex export's lawyer-name column repeats the same
        comma-separated representative list once per case party (e.g. the same
        7 names 3x in a row), so it's deduplicated here, preserving first-seen
        order, rather than displayed verbatim."""
        if not raw:
            return raw
        seen = []
        for name in raw.split(','):
            name = name.strip()
            if name and name not in seen:
                seen.append(name)
        return ", ".join(seen)

    def _train_classifier(self):
        if self.past_cases:
            corpus = [case['judicial_prehistory'] for case in self.past_cases]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            print(f"✅ Classifier: Indexed {len(self.past_cases)} historical cases.")

    def find_similar_case(self, text):
        """Gate for the deterministic "CLASSIFIER MATCH FOUND" response (Step 2
        in LegalAgent.get_advice), which returns the matched case's real details
        (classification, lawyer, case excerpt) as if they directly answer the
        question — so a weak match here reads as a confident, wrong answer, not
        an ambiguous one. 0.35 was picked empirically: unrelated queries in this
        corpus (3000 cases, heavy shared legal-procedure boilerplate that TF-IDF
        doesn't discount well) were scoring 0.27-0.32 against a arbitrary
        "closest" case and getting treated as a real match. It's still not a
        clean separation — some genuinely matching text scores below this too —
        but it's a meaningfully better bar than 0.15."""
        if self.tfidf_matrix is None or not self.past_cases:
            return None
        new_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(new_vec, self.tfidf_matrix).flatten()
        idx = similarities.argmax()
        if similarities[idx] < 0.35:
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
            
            # Filter cases with similarity score > 0.2 and limit results
            similar_cases = []
            for idx in sorted_indices:
                if similarities[idx] > 0.2 and len(similar_cases) < limit:
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


# Known labels in student_mh_counseling_100k_with_label_column.csv. Filters out
# the rare malformed row(s) where a truncated/garbled sentence ends up in the
# label column instead of a real category (e.g. a stray CSV line break).
KNOWN_MENTAL_HEALTH_LABELS = {
    "depression", "stress", "seeking help", "self-esteem issues",
    "relationship problems", "positive mental state", "coping strategies",
    "suicidal thoughts", "anxiety", "academic pressure", "grief",
    "anger management", "loneliness", "substance abuse",
    "general well-being", "eating disorders",
}

# Labels whose predicted probability counts as a risk signal. Conservative on
# purpose — mirrors what the earlier zero-shot check flagged (suicide/self-harm
# risk specifically), not general distress like "stress" or "grief".
MENTAL_HEALTH_RISK_LABELS = {"suicidal thoughts"}


class MentalHealthRiskClassifier:
    """Multi-class mental-health risk classifier: SentenceTransformer embeddings
    -> PCA -> RandomForestClassifier, the same technique used in
    notebook/Modeling.ipynb's Random Forest experiment, but trained on
    src/data/student_mh_counseling_100k_with_label_column.csv's question/label
    columns instead of the notebook's legal-verdict dataset (that model's
    learned labels — legal decision outcomes — have nothing to do with mental-
    health risk, so it can't be reused directly; only the modeling technique is
    reused here).

    Replaces the earlier zero-shot NLI-based risk check as the second, heavier
    signal alongside the fast keyword check in src/services/crisis_detection.py.
    Predicts the full label set (depression, stress, seeking help, suicidal
    thoughts, ...) rather than a plain risk/non-risk binary, then flags is_risk
    from the predicted probability of the "suicidal thoughts" class specifically
    (see MENTAL_HEALTH_RISK_LABELS) — it only ever adds coverage on top of the
    keyword check, never overrides it, and is not a clinical assessment.

    Embedding and training on the full ~100k rows is too slow to do lazily on
    first request (likely minutes on CPU), so training uses a bounded random
    subsample instead — embedded and fit once, on first actual use, and cached
    in-process for the classifier's lifetime (same lazy-loading philosophy as
    the other classifiers in this file, just with a capped training set size to
    keep the first-use delay reasonable).
    """

    def __init__(
        self,
        csv_path: str = DEFAULT_MENTAL_HEALTH_QA_CSV,
        sample_size: int = 8000,
        embedding_model_name: str = "all-mpnet-base-v2",
        pca_components: int = 20,
        random_state: int = 42,
    ):
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.embedding_model_name = embedding_model_name
        self.pca_components = pca_components
        self.random_state = random_state
        self._embedder = None
        self._pca = None
        self._label_encoder = None
        self._rf = None
        self._trained = False

    def _load_training_sample(self):
        import csv
        import random

        if not os.path.exists(self.csv_path):
            print(f"⚠️ Mental-health risk classifier: dataset not found: {self.csv_path}")
            return [], []

        csv.field_size_limit(10 * 1024 * 1024)
        rows = []
        with open(self.csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = (row.get('question') or '').strip()
                label = (row.get('label') or '').strip()
                if question and label in KNOWN_MENTAL_HEALTH_LABELS:
                    rows.append((question, label))

        if not rows:
            return [], []
        if len(rows) > self.sample_size:
            random.Random(self.random_state).shuffle(rows)
            rows = rows[:self.sample_size]

        texts, labels = zip(*rows)
        return list(texts), list(labels)

    def _ensure_trained(self):
        if self._trained:
            return
        self._trained = True  # set first so a failed/empty attempt doesn't retry every call

        texts, labels = self._load_training_sample()
        if not texts:
            return

        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        print(f"🧠 Training mental-health risk classifier on {len(texts)} sampled messages...")
        self._embedder = SentenceTransformer(self.embedding_model_name)
        embeddings = self._embedder.encode(texts, show_progress_bar=False)

        n_components = min(self.pca_components, len(texts) - 1, embeddings.shape[1])
        self._pca = PCA(n_components=n_components, random_state=self.random_state)
        reduced = self._pca.fit_transform(embeddings)

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(labels)

        self._rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=self.random_state)
        self._rf.fit(reduced, y)
        print(f"✅ Mental-health risk classifier trained: {len(texts)} messages, {len(self._label_encoder.classes_)} labels.")

    def classify_mental_health_risk(self, text: str, risk_threshold: float = 0.3) -> dict:
        """
        Classify text against the trained label set and flag a risk signal from
        the predicted probability of risk-labeled classes (see
        MENTAL_HEALTH_RISK_LABELS).

        Args:
            text: The message text to screen.
            risk_threshold: Minimum predicted probability on a risk label before
                treating the text as a risk signal (checked independently of
                whether that label is the single most likely one).

        Returns:
            dict with keys is_risk (bool), label (str, top predicted label),
            score (float, top label's probability), and scores (dict of all
            labels to their predicted probabilities), or None if the text is
            empty or the classifier has no trained model (e.g. dataset missing).
        """
        if not text or not text.strip():
            return None
        self._ensure_trained()
        if self._rf is None:
            return None

        try:
            embedding = self._embedder.encode([text])
            reduced = self._pca.transform(embedding)
            proba = self._rf.predict_proba(reduced)[0]
        except Exception as e:
            print(f"⚠️ Mental-health risk classification failed: {e}")
            return None

        label_probs = dict(zip(self._label_encoder.classes_, (float(p) for p in proba)))
        top_label = max(label_probs, key=label_probs.get)
        risk_score = max((label_probs.get(label, 0.0) for label in MENTAL_HEALTH_RISK_LABELS), default=0.0)

        return {
            "is_risk": risk_score >= risk_threshold,
            "label": top_label,
            "score": label_probs[top_label],
            "scores": label_probs,
        }