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