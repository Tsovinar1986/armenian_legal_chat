class CompanyLegalRepo:
    def __init__(self, vector_db):
        self.db = vector_db

    def get_classified_evidence(self, query, category=None):
        """
        Retrieves the most relevant legal context.
        If a category (like 'Criminal') is provided, it filters the search.
        """
        search_args = {"k": 3}
        if category:
            search_args["filter"] = {"category": category}
            
        # Similarity search returns the actual law snippets
        results = self.db.similarity_search(query, **search_args)
        return results