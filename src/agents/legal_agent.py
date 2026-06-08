# src/agents/legal_agent.py
import urllib.parse
import os
import csv
from langchain_ollama import OllamaLLM
from src.services.case_export import CaseExportService

class LegalAgent:
    def __init__(self, repo, state, classifier=None, model=None):
        """
        Initializes the Legal Agent.
        :param repo: Instance of CompanyLegalRepo containing Chroma vector_db
        :param state: Instance of SystemState
        :param classifier: Instance of LegalCaseClassifier
        :param model: Model name to use (e.g., "armenia-lawyer-router")
        """
        self.repo = repo
        self.state = state
        self.classifier = classifier
        self.model_name = model or "armenia-lawyer-router"
        self.court_cases = []
        self.export_service = CaseExportService()  # Initialize export service
        
        # Load court papers data from CSV
        self._load_court_cases()
        
        # Initialize the LLM with the specified model
        try:
            print(f"🔄 Initializing Ollama LLM with model: {self.model_name}")
            self.llm = OllamaLLM(model=self.model_name)
            print(f"✅ LLM initialized successfully with model: {self.model_name}")
        except Exception as e:
            print(f"❌ CRITICAL: Could not initialize LLM with model '{self.model_name}'")
            print(f"   Error: {e}")
            print(f"   Please ensure:")
            print(f"   1. Ollama service is running (run: ollama serve)")
            print(f"   2. Model is pulled (run: ollama pull {self.model_name})")
            print(f"   3. Check if model name is correct: {self.model_name}")
            self.llm = None
    
    def _load_court_cases(self):
        """Load court papers from CSV file for use as examples"""
        csv_path = "src/data/court_papers_full.csv"
        if not os.path.exists(csv_path):
            print(f"⚠️ Court papers CSV not found at {csv_path}")
            return
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.court_cases = list(reader)
            print(f"✅ Loaded {len(self.court_cases)} court cases from CSV")
        except Exception as e:
            print(f"⚠️ Error loading court cases: {e}")
            self.court_cases = []
    
    def _find_relevant_cases(self, query: str, limit: int = 3) -> list:
        """Find relevant court cases from CSV based on query keywords"""
        if not self.court_cases:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score each case based on keyword matching
        scored_cases = []
        for case in self.court_cases:
            case_text = f"{case.get('Category', '')} {case.get('Verdict_Text', '')}".lower()
            case_words = set(case_text.split())
            
            # Calculate similarity score
            matching_words = len(query_words.intersection(case_words))
            if matching_words > 0:
                scored_cases.append((matching_words, case))
        
        # Sort by score and return top cases
        scored_cases.sort(reverse=True, key=lambda x: x[0])
        return [case for _, case in scored_cases[:limit]]

    def _truncate_text(self, text: str, max_chars: int = 900) -> str:
        if not text:
            return "N/A"
        if len(text) <= max_chars:
            return text.strip()
        truncated = text.strip()[:max_chars]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        return f"{truncated.strip()}..."

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
                    case_excerpt = self._truncate_text(matched_case.get('judicial_prehistory', ''), max_chars=1200)
                    return (
                        f"🎯 [CLASSIFIER MATCH FOUND]\n"
                        f"🔹 Դասակարգում: {matched_case.get('civil_case_classifier')}\n"
                        f"🔹 Նմանատիպ գործ: {matched_case.get('unique_number')}\n"
                        f"🔹 Հղում: {matched_case.get('link')}\n"
                        f"🔹 Առաջարկվող փաստաբան: {lawyer_display}\n\n"
                        f"📄 Գործի նախապատմություն / բովանդակության օրինակ:\n{case_excerpt}\n"
                        f"\nՓոխարենը կարող էք բացել հղումը՝ ամբողջ գործը ընթերցելու համար։"
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
                        response_text += "Ես համակարգի տվյալների բազայում գտա այս իրավական հարցին համապատասխանող պատմական գործ։\n\n"
                        if case_number:
                            response_text += f"🔢 Դատական գործի համարը: {case_number}\n"
                        if datalex_link:
                            response_text += f"🌐 DataLex հղումը: {datalex_link}\n"
                        response_text += "\n📄 Գործի բովանդակության օրինակ:\n"
                        response_text += self._truncate_text(doc.page_content, max_chars=1200)
                        response_text += "\n\nՄանրամասների համար բացեք հղումը կամ երկարացրեք որոնումը։"
                        return response_text
        except Exception as e:
            print(f"⚠️ Վեկտորային բազայի ստուգման սխալ: {e}")

        # Step 4: Fallback to general RAG synthesis
        return self._generate_rag_response(user_query)

    def _generate_rag_response(self, query: str) -> str:
        try:
            print(f"🔍 Starting RAG response generation for query: {query[:50]}...")
            docs = self.repo.db.similarity_search(query, k=3)
            print(f"📚 Found {len(docs)} documents in vector DB")
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Find relevant court cases from CSV
            relevant_cases = self._find_relevant_cases(query, limit=2)
            cases_context = ""
            if relevant_cases:
                print(f"📋 Found {len(relevant_cases)} relevant court cases from database")
                cases_context = "\n\nՀԱՄՀ case examples FROM REAL COURT DECISIONS:\n"
                for i, case in enumerate(relevant_cases, 1):
                    case_num = case.get('Case_Number', 'N/A')
                    category = case.get('Category', 'N/A')
                    judge = case.get('Judge', 'N/A')
                    verdict = case.get('Verdict_Text', 'N/A')[:300] + "..."
                    cases_context += f"\n📌 Example {i}: Case {case_num}\n"
                    cases_context += f"   Category: {category}\n"
                    cases_context += f"   Judge: {judge}\n"
                    cases_context += f"   Verdict Summary: {verdict}\n"

            if not context:
                return "Համապատասխան տեղական իրավական նախադեպեր չգտնվեցին։ Խնդրում ենք համոզվել, որ ֆայլերը ճիշտ են ներբեռնված համակարգ։"

            # If LLM is available, use it to generate a proper response
            if self.llm:
                try:
                    print(f"📝 Generating response using model: {self.model_name}")
                    # Create a prompt that instructs the model to answer based on the context
                    system_prompt = """Դուք մասնագետ իրավաբան ես, որը փակ համակարգում գործում ես:
Հաճախորդի հարցին պատասխանիր հետևյալ համատեքստի և իրական դատական գործերի հիման վրա:
Պատասխան տուր կոնկրետ, կառուցված, հասկանալի և հղում կատարիր նմանատիպ դատական գործերի:

ՀԱՄԱՏԵՔՍՏ ԱՎԵԼԱՑՅԱԼ ՏԵՂԵԿԱՏՎՈՒԹՅՈՒՆ:
{context}

ԻՐԱԿԱՆ ԴԱՏԱԿԱՆ ԳՈՐԾԵՐԻ ՕՐԻՆԱԿՆԵՐ:
{cases_context}

ՀԱՃԱԽՈՐԴԻ ՀԱՐՑ: {query}

ՊԱՏԱՍԽԱՆ (հայերեն, մասնագետական և հասկանալի):"""

                    prompt = system_prompt.format(context=context, cases_context=cases_context, query=query)
                    
                    # Call the LLM to generate response
                    print(f"⏳ Waiting for model response...")
                    response = self.llm.invoke(prompt)
                    print(f"✅ Model response received ({len(response)} characters)")
                    return response
                except Exception as llm_error:
                    print(f"❌ LLM generation error: {llm_error}")
                    print(f"   Model: {self.model_name}")
                    # Fall back to template response if LLM fails
                    return (
                        "Նույնական պատմական դատական գործ չի գտնվել։ "
                        "Տրամադրվում է սինթեզված իրավաբանական խորհրդատվություն՝ "
                        "հիմնված բազայում առկա մոտակա իրավական կոնտեքստների վրա։"
                    )
            else:
                # If LLM is not available, return template response
                print(f"⚠️ LLM not available, using fallback response")
                return (
                    "Նույնական պատմական դատական գործ չի գտնվել։ "
                    "Տրամադրվում է սինթեզված իրավաբանական խորհրդատվություն՝ "
                    "հիմնված բազայում առկա մոտակա իրավական կոնտեքստների վրա։"
                )
        except Exception as e:
            print(f"❌ RAG Response generation failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Իրավական տվյալների համադրման (RAG) համակարգի սխալ: {str(e)}"
    
    def get_similar_cases(self, query: str, limit: int = 5) -> list:
        """
        Get similar cases based on the query and export them to a text file.
        
        Args:
            query: The search query/case description
            limit: Maximum number of similar cases to return
            
        Returns:
            List of similar case dictionaries
        """
        if not self.classifier:
            print("⚠️ Classifier not available for similar case search")
            return []
        
        try:
            similar_cases = self.classifier.find_multiple_similar_cases(query, limit=limit)
            
            if similar_cases:
                print(f"\n✅ Found {len(similar_cases)} similar cases")
                
                # Export cases to file
                export_path = self.export_service.export_similar_cases(similar_cases, query)
                
                if export_path:
                    print(f"📁 Cases exported to: {export_path}")
                    return similar_cases
            else:
                print("⚠️ No similar cases found")
                return []
                
        except Exception as e:
            print(f"❌ Error getting similar cases: {e}")
            return []
    
    def get_approved_cases_with_lawyers(self, limit: int = 10) -> dict:
        """
        Get approved/successful cases with lawyer information.
        
        Args:
            limit: Maximum number of approved cases to return
            
        Returns:
            Dictionary containing approved cases and lawyer statistics
        """
        if not self.classifier:
            print("⚠️ Classifier not available for case search")
            return {}
        
        try:
            # Get approved cases
            approved_cases = self.classifier.find_approved_cases(limit=limit)
            
            if not approved_cases:
                print("⚠️ No approved cases found in the database")
                return {
                    'approved_cases': [],
                    'total_count': 0,
                    'lawyers': []
                }
            
            print(f"\n✅ Found {len(approved_cases)} approved cases")
            
            # Export approved cases to file
            export_path = self.export_service.export_approved_cases(approved_cases)
            
            # Get top lawyers by cases
            top_lawyers = self.classifier.get_top_lawyers_by_cases(limit=10)
            
            # Format lawyer information
            lawyer_info = []
            for lawyer_name, stats in top_lawyers:
                lawyer_info.append({
                    'name': lawyer_name,
                    'case_count': stats['count'],
                    'sample_cases': stats['cases'][:2]  # Include up to 2 sample cases
                })
            
            return {
                'approved_cases': approved_cases,
                'total_count': len(approved_cases),
                'lawyers': lawyer_info,
                'export_file': export_path
            }
            
        except Exception as e:
            print(f"❌ Error getting approved cases: {e}")
            return {}
    
    def get_lawyer_cases(self, lawyer_name: str, limit: int = 10) -> list:
        """
        Get all cases handled by a specific lawyer.
        
        Args:
            lawyer_name: Name of the lawyer to search for
            limit: Maximum number of cases to return
            
        Returns:
            List of case dictionaries for the lawyer
        """
        if not self.classifier:
            print("⚠️ Classifier not available")
            return []
        
        try:
            cases = self.classifier.find_cases_by_lawyer(lawyer_name, limit=limit)
            
            if cases:
                print(f"\n✅ Found {len(cases)} cases for lawyer: {lawyer_name}")
                # Export to file
                export_path = self.export_service.export_similar_cases(cases, f"Cases by {lawyer_name}")
                return cases
            else:
                print(f"⚠️ No cases found for lawyer: {lawyer_name}")
                return []
                
        except Exception as e:
            print(f"❌ Error getting lawyer cases: {e}")
            return []
    
    def format_similar_cases_response(self, similar_cases: list) -> str:
        """
        Format similar cases into a readable response.
        
        Args:
            similar_cases: List of similar case dictionaries
            
        Returns:
            Formatted string response
        """
        if not similar_cases:
            return "❌ Նման գործեր չ գտնվեցին։"
        
        response = f"✨ Գտնվել է {len(similar_cases)} նման դատական գործ\n"
        response += "=" * 70 + "\n\n"
        
        for idx, case in enumerate(similar_cases, 1):
            similarity = case.get('similarity_score', 0)
            similarity_pct = int(similarity * 100)
            
            response += f"📌 ԳՈՐԾ #{idx} (համընկնում {similarity_pct}%)\n"
            response += f"   🔢 Համար: {case.get('unique_number', 'N/A')}\n"
            response += f"   ⚖️ Փաստաբան: {case.get('lawyer_name', 'Նշված չէ')}\n"
            response += f"   📋 Դասակարգում: {case.get('civil_case_classifier', 'N/A')}\n"
            response += f"   🌐 Հղում: {case.get('link', 'N/A')}\n"
            response += "\n"
        
        response += "=" * 70 + "\n"
        response += f"📁 Բոլոր գործերը արտահանվել են text ֆայලի մեջ / All cases exported to text file\n"
        
        return response
    
    def format_approved_cases_response(self, result: dict) -> str:
        """
        Format approved cases into a readable response.
        
        Args:
            result: Dictionary from get_approved_cases_with_lawyers
            
        Returns:
            Formatted string response
        """
        if not result.get('approved_cases'):
            return "❌ Հաստատված գործեր չ գտնվեցին։"
        
        approved_cases = result.get('approved_cases', [])
        lawyers = result.get('lawyers', [])
        
        response = f"✅ ՀԱՍՏԱՏՎԱԾ/ՀԱՋՈՂՎԱԾ ԴԱՏԱԿԱՆ ԳՈՐԾԵՐ\n"
        response += f"Ընդամենը հաստատված գործեր: {result.get('total_count', 0)}\n\n"
        
        if lawyers:
            response += f"👨‍⚖️ TOP ՓԱՍՏԱԲԱՆՆԵՐ ({len(lawyers)})\n"
            response += "=" * 70 + "\n"
            for idx, lawyer in enumerate(lawyers[:5], 1):  # Show top 5 lawyers
                response += f"\n#{idx} ⭐ {lawyer['name']}\n"
                response += f"    Հաջողված գործեր: {lawyer['case_count']}\n"
        
        response += "\n" + "=" * 70 + "\n"
        response += f"📄 Մանրամասն հաշվետվություն արտահանվել է text ֆայլի մեջ\n"
        response += f"📁 Ֆայլի վայրը: {result.get('export_file', 'exports/')}\n"
        
        return response