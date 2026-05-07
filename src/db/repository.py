class CompanyLegalRepo:
    def __init__(self, vector_db):
        self.db = vector_db

    def get_classified_evidence(self, query, category=None, k=6):
        """
        Retrieves relevant legal cases with metadata.
        """
        search_args = {"k": k}
        if category:
            search_args["filter"] = {"category": category}

        results = self.db.similarity_search(query, **search_args)
        return results