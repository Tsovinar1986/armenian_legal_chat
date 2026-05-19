# src/agents/legal_agent.py
import urllib.parse

class LegalAgent:
    def __init__(self, repo, state):
        """
        Ինիցիալիզացնում է Իրավաբանական Գործակալին (Legal Agent) տվյալների բազայի ռեպոզիտորիայով և համակարգի գլոբալ ստատուսով։
        :param repo: CompanyLegalRepo-ի օրինակ, որը պարունակում է Chroma vector_db-ն
        :param state: SystemState-ի օրինակ
        """
        self.repo = repo
        self.state = state
        # Եթե օգտագործում եք LangChain/Ollama LLM, այն կարող եք կանչել այստեղ՝
        # self.llm = YourLLMInitialization()

    def get_advice(self, user_query: str) -> str:
        """
        Մշակում է օգտատիրոջ իրավաբանական հարցումը։ Սկզբում ստուգում է նախադեպային բազան՝ 
        համընկնող գործ գտնելու և ուղիղ DataLex հղում կամ գործի համար տրամադրելու համար։ 
        Համընկնում չգտնելու դեպքում անցնում է ստանդարտ RAG տարբերակին։
        """
        try:
            # 1. Հարցում vector տվյալների բազային (Chroma-ն վերադարձնում է L2 distance score՝ որքան ցածր, այնքան նման է)
            results = self.repo.vector_db.similarity_search_with_score(user_query, k=1)
            
            if results:
                doc, score = results[0]
                
                # Կարգավորեք շեմը (threshold) ըստ nomic-embed-text մոդելի արդյունավետության։
                # Սովորաբար 0.45-ից ցածր արժեքը ցույց է տալիս բարձր ճշգրտության համընկնում բազայում առկա գործի հետ։
                if score < 0.45:
                    metadata = doc.metadata or {}
                    case_number = metadata.get("case_number")
                    datalex_link = metadata.get("datalex_link")
                    
                    # Եթե հղումը չկա, բայց ունենք դատական գործի համարը (օրինակ՝ ԵԴ/1234/02/24), կառուցում ենք հղումը
                    if not datalex_link and case_number:
                        # Ապահով կերպով URL-encode ենք անում հայերեն տառերը և սիմվոլները
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
            # Խուսափում ենք համակարգի կանգից, եթե մետատվյալների ստրուկտուրան կամ վեկտորային հարցումը սխալ տան
            print(f"⚠️ Վեկտորային բազայի ստուգման սխալ (փնտրման ընթացքում): {e}")

        # 2. Եթե հստակ պատմական համընկնում չի հայտնաբերվել, անցնում ենք ստանդարտ RAG համադրմանը
        return self._generate_rag_response(user_query)

    def _generate_rag_response(self, query: str) -> str:
        """
        RAG մեթոդ, երբ տվյալների բազայում կոնկրետ դատական գործի պրոֆիլ կամ հղում չի հայտնաբերվում։
        """
        try:
            # Վերցնում ենք թոփ համապատասխանող կոնտեքստային փաստաթղթերը ռեպոզիտորիայից
            docs = self.repo.vector_db.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            if not context:
                return "Համապատասխան տեղական իրավական նախադեպեր չգտնվեցին։ Խնդրում ենք համոզվել, որ ֆայլերը ճիշտ են ներբեռնված համակարգ։"

            # Այստեղ կատարվում է Ձեր LLM-ի կանչը (օրինակ՝ Ollama կամ LangChain chain)
            # prompt = f"Կոնտեքստ: {context}\n\nՀարց: {query}\n\nՏրամադրիր իրավաբանական խորհրդատվություն ՀՀ օրենսդրության համաձայն՝"
            # return self.llm.invoke(prompt)
            
            return f"Նույնական պատմական դատական գործ չի գտնվել։ Տրամադրվում է սինթեզված իրավաբանական խորհրդատվություն՝ հիմնված բազայում առկա մոտակա իրավական կոնտեքստների վրա։"
            
        except Exception as e:
            return f"Իրավական տվյալների համադրման (RAG) համակարգի սխալ: {str(e)}"
