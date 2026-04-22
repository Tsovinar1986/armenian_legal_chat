import os
import pandas as pd
from src.db.case_parser import classify_and_parse_cases

class IngestionService:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def process_file(self, file_path):
        """Detects file type and extracts Armenian legal data."""
        filename = os.path.basename(file_path)
        
        try:
            # Handle Excel Files (.xlsx)
            if filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                # Assuming the Armenian text is in a column named 'text' or 'prompt'
                texts = df.iloc[:, 0].astype(str).tolist() 
                self.vector_db.add_texts(texts=texts)
                return f"Successfully indexed {len(texts)} rows from Excel."

            # Handle Case List Text Files (.txt)
            elif filename.endswith('.txt'):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Use our custom Armenian PHP-style parser
                cases = classify_and_parse_cases(content)
                
                if not cases:
                    return "⚠️ No cases found. Check case_parser logic."
                
                texts = [c['verdict'] for c in cases]
                metadatas = [{"category": c['legal_category']} for c in cases]
                
                self.vector_db.add_texts(texts=texts, metadatas=metadatas)
                return f"Successfully indexed {len(cases)} Armenian cases."

        except Exception as e:
            return f"❌ Error: {str(e)}"