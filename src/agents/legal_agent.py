from langchain_ollama import OllamaLLM

class LegalAgent:
    def __init__(self, repo, state):
        self.repo = repo
        self.state = state
        # Initialize the Armenian LLM
        self.llm = OllamaLLM(model="armenia_lawyer_router:latest")

    def get_advice(self, user_query):
        # 1. Real-time actions from the camera state
        detected_actions = self.state.people_actions
        
        # 2. Combine actions and query for search
        search_query = f"{user_query} {' '.join(detected_actions)}"
        
        # 3. Search the Repository (filtered by what was seen)
        # We pass None for category to search everything, or filter if needed
        docs = self.repo.get_classified_evidence(search_query, category=None)
        context = "\n".join([d.page_content for d in docs])

        # 4. Your Armenian Prompt Logic
        prompt = f"""
        Դուք ՀՀ իրավաբան եք:
        Արձանագրված գործողություններ: {detected_actions}
        Իրավական նախադեպեր: {context}
        
        Հարց: {user_query}
        
        Տվեք մասնագիտական խորհրդատվություն հայերենով: Եթե արձանագրվել է բռնություն (ապտակ/հրում), 
        նշեք ՀՀ քրեական օրենսգրքի համապատասխան հոդվածները (օրինակ՝ 195):
        """
        
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"Error connecting to LLM: {e}"