import csv
import json
import os

import pandas as pd
from src.db.case_parser import classify_and_parse_cases


class IngestionService:
    """Embeds uploaded documents into the case vector store. .xlsx/.csv/.json/
    .txt/.docx/.pdf each have a dedicated extractor; any other extension
    falls back to best-effort plain-text reading instead of being rejected
    outright, since most non-binary formats still carry readable text."""

    def __init__(self, vector_db):
        self.vector_db = vector_db

    def process_file(self, file_path):
        """Detects file type, extracts its text, and indexes it."""
        try:
            suffix = os.path.splitext(file_path)[1].lower()

            if suffix == '.xlsx':
                texts = self.extract_texts(file_path)
                self.vector_db.add_texts(texts=texts)
                return f"Successfully indexed {len(texts)} rows from Excel."

            if suffix == '.csv':
                texts = self.extract_texts(file_path)
                if not texts:
                    return "⚠️ No text found in CSV."
                self.vector_db.add_texts(texts=texts)
                return f"Successfully indexed {len(texts)} rows from CSV."

            if suffix == '.json':
                texts = self.extract_texts(file_path)
                if not texts:
                    return "⚠️ No text found in JSON."
                self.vector_db.add_texts(texts=texts)
                return f"Successfully indexed {len(texts)} entries from JSON."

            if suffix == '.txt':
                content = self.extract_text(file_path)
                # Use our custom Armenian PHP-style parser first — falls
                # back to indexing the raw text as a single document when
                # the file isn't in that specific case-list format, instead
                # of discarding it.
                cases = classify_and_parse_cases(content)
                if cases:
                    texts = [c['verdict'] for c in cases]
                    metadatas = [{"category": c['legal_category']} for c in cases]
                    self.vector_db.add_texts(texts=texts, metadatas=metadatas)
                    return f"Successfully indexed {len(cases)} Armenian cases."
                if content.strip():
                    self.vector_db.add_texts(texts=[content])
                    return "Successfully indexed 1 text document (no case-list format detected)."
                return "⚠️ No cases found. Check case_parser logic."

            # .docx, .pdf, and anything without a dedicated extractor above
            # all share the same "extract the whole document as one text
            # blob, index it as one chunk" path.
            content = self.extract_text(file_path)
            if not content.strip():
                return "⚠️ No extractable text found in this file (it may be a scanned/image-only or unsupported binary format)."
            self.vector_db.add_texts(texts=[content])
            label = {'.docx': 'Word document', '.pdf': 'PDF document'}.get(suffix, 'document')
            return f"Successfully indexed 1 {label}."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def extract_texts(self, file_path):
        """List of separate text chunks for formats that are naturally
        tabular/enumerable (xlsx rows, csv rows, json entries)."""
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix == '.xlsx':
            df = pd.read_excel(file_path)
            # Assuming the Armenian text is in the first column.
            return df.iloc[:, 0].astype(str).tolist()
        if suffix == '.csv':
            with open(file_path, newline='', encoding='utf-8-sig') as f:
                return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]
        if suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._flatten_json_texts(data)
        return [self.extract_text(file_path)]

    def extract_text(self, file_path):
        """Best-effort single-blob plain-text extraction for any file type —
        shared by both the web upload flow (embedding, above) and the CLI
        upload flow (src/main.py feeds this straight to get_advice), so
        there's one place that knows how to read each format instead of two
        independent, drifting implementations."""
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix == '.xlsx':
            return " ".join(self.extract_texts(file_path))
        if suffix in ('.csv', '.json'):
            return "\n".join(self.extract_texts(file_path))
        if suffix == '.docx':
            from docx import Document
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if suffix == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        # .txt and anything else without a dedicated reader: best-effort
        # plain-text read (errors='ignore' only for the unknown-extension
        # case, so a genuinely binary file doesn't raise here).
        errors = 'strict' if suffix == '.txt' else 'ignore'
        with open(file_path, 'r', encoding='utf-8', errors=errors) as f:
            return f.read()

    @staticmethod
    def _flatten_json_texts(data):
        """Collect every string value out of arbitrarily nested JSON (a list
        of strings, a list of objects, or a single object) as index-able
        text."""
        texts = []
        if isinstance(data, str):
            texts.append(data)
        elif isinstance(data, list):
            for item in data:
                texts.extend(IngestionService._flatten_json_texts(item))
        elif isinstance(data, dict):
            for value in data.values():
                texts.extend(IngestionService._flatten_json_texts(value))
        return [t.strip() for t in texts if isinstance(t, str) and t.strip()]
