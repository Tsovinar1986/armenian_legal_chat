from langchain_ollama import OllamaLLM

class LegalAgent:
    def __init__(self, repo, state):
        self.repo = repo
        self.state = state
        self.llm = OllamaLLM(model="armenia_lawyer_router:latest", temperature=0.6)

    def get_advice(self, user_query: str):
        detected_actions = getattr(self.state, 'people_actions', [])

        # Retrieve relevant cases
        docs = self.repo.get_classified_evidence(user_query, category=None, k=6)

        context = "\n\n".join([
            f"📌 Գործ #{i+1} | {d.metadata.get('case_type', 'Ընդհանուր')}\n{d.page_content[:800]}..." 
            for i, d in enumerate(docs)
        ])

        prompt = f"""
        Դուք Հայաստանի Հանրապետության փորձառու իրավաբան եք։

        **Հարցը**: {user_query}
        **Արձանագրված գործողություններ**: {detected_actions}

        **Նմանատիպ դատական գործեր տվյալների բազայից**:
        {context}

        ---
        Պատասխանեք հայերենով՝ խիստ կառուցվածքով:

        1. **Գործի տեսակը**՝ (օրինակ՝ Ընտանեկան բռնություն, Գույքային վեճ, Աշխատանքային վեճ, Վնասի հատուցում և այլն)

        2. **Նման դեպքեր մեր բազայում**՝ 
           Մեր տվյալների բազայում այս տեսակի խնդրով կան մի քանի նախադեպեր։

        3. **Ինչպես է լուծվել դատարանում**՝ 
           - Գործ #1: ... (կարճ ամփոփում + ելքը՝ մեղադրել են / արդարացրել են / փոխհատուցում են սահմանել և այլն)

        4. **Իրավական հիմքեր**՝ ՀՀ օրենսգրքերի համապատասխան հոդվածներ

        5. **Խորհրդատվություն**՝ ինչ անել հիմա, քայլ առ քայլ

        Պատասխանը պետք է լինի հստակ, մասնագիտական և օգտակար։
        """

        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            return f"❌ Տեխնիկական խնդիր է առաջացել։ Խնդրում ենք փորձել կրկին։ ({str(e)})"