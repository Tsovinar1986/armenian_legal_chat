# src/agents/legal_agent.py
import urllib.parse

class LegalAgent:
    def __init__(self, repo, state, classifier=None):
        """
        Initializes the Legal Agent.
        :param repo: Instance of CompanyLegalRepo containing Chroma vector_db
        :param state: Instance of SystemState
        :param classifier: Instance of LegalCaseClassifier
        """
        self.repo = repo
        self.state = state
        self.classifier = classifier

    def get_advice(self, user_query: str) -> str:
        """
        Processes the legal query by routing it through the classifier first, 
        then a strict vector match, and falling back to a structured RAG pipeline.
        """
        # Step 1: Interactive check / Clarification hook
        if len(user_query.split()) < 3:
            return "Խնդրում եմ, նկարագրեք ձեր իրավական խնդիրը մի փոքր ավելի մանրամասն, որպեսզի կարողանամ ճշգրիտ նախադեպեր գտնել:"

        # Step 2: Try the Classifier first (even for voice/text)
        if self.classifier:
            try:
                matched_case = self.classifier.find_similar_case(user_query)
                if matched_case:
                    lawyer = matched_case.get('lawyer_name')
                    lawyer_display = lawyer if lawyer and lawyer != "(NULL)" else "Նշված չէ"
                    
                    return (
                        f"🎯 [CLASSIFIER MATCH FOUND]\n"
                        f"🔹 Դասակարգում: {matched_case.get('civil_case_classifier')}\n"
                        f"🔹 Նմանատիպ գործ: {matched_case.get('unique_number')}\n"
                        f"🔹 Հղում: {matched_case.get('link')}\n"
                        f"🔹 Առաջարկվող փաստաբան: {lawyer_display}"
                    )
            except Exception as ex:
                print(f"⚠️ Error during case classification: {ex}")

        # Step 3: Try Exact Vector DB Precedent Search
        try:
            results = self.repo.db.similarity_search_with_score(user_query, k=1)
            if results:
                doc, score = results[0]
                # Lower scores mean higher semantic similarity
                if score < 0.45:
                    metadata = doc.metadata or {}
                    case_number = metadata.get("case_number")
                    datalex_link = metadata.get("datalex_link")

                    if not datalex_link and case_number:
                        encoded_case = urllib.parse.quote(str(case_number).strip())
                        datalex_link = f"http://www.datalex.am/?app=AppCaseSearch&caseNumber={encoded_case}"

                    if case_number or datalex_link:
                        response_text = "✨ [Համընկնում է գտնվել նախադեպային բազայում]\n"
                        response_text += "Ես համակարգի տվյալների բազայում գտա այս իրավական հարցին լիովին համապատասխանող պատմական նախադեպ։\n\n"
                        if case_number:
                            response_text += f"🔢 Դատական գործի համարը: {case_number}\n"
                        if datalex_link:
                            response_text += f"🌐 DataLex հղումը: {datalex_link}\n"
                        return response_text
        except Exception as e:
            print(f"⚠️ Վեկտորային բազայի ստուգման սխալ: {e}")

        # Step 4: Fallback to general RAG synthesis
        return self._generate_rag_response(user_query)

    def _generate_rag_response(self, query: str) -> str:
        try:
            docs = self.repo.db.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

            if not context:
                return "Համապատասխան տեղական իրավական նախադեպեր չգտնվեցին։ Խնդրում ենք համոզվել, որ ֆայլերը ճիշտ են ներբեռնված համակարգ։"

            return (
                "Նույնական պատմական դատական գործ չի գտնվել։ "
                "Տրամադրվում է սինթեզված իրավաբանական խորհրդատվություն՝ "
                "հիմնված բազայում առկա մոտակա իրավական կոնտեքստների վրա։"
            )
        except Exception as e:
            return f"Իրավական տվյալների համադրման (RAG) համակարգի սխալ: {str(e)}"