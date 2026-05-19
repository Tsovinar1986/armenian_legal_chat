# src/services/classifier.py
import os
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LegalCaseClassifier:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        self.past_cases = []
        self.vectorizer = TfidfVectorizer(max_features=5000)
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
                    'lawyer_name': cols[4].get_text(strip=True),
                    'link': f"http://www.datalex.am/?app=AppCaseSearch&case_id={cols[0].get_text(strip=True)}"
                })

    def _train_classifier(self):
        if self.past_cases:
            corpus = [case['judicial_prehistory'] for case in self.past_cases]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            print(f"✅ Classifier: Indexed {len(self.past_cases)} historical cases.")

    def find_similar_case(self, text):
        if self.tfidf_matrix is None or not self.past_cases:
            return None
        new_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(new_vec, self.tfidf_matrix).flatten()
        idx = similarities.argmax()
        if similarities[idx] < 0.15:
            return None
        return self.past_cases[idx]