# src/agents/legal_agent.py
import urllib.parse
import os
import csv
from langchain_ollama import OllamaLLM
from src.services.case_export import CaseExportService
from src.services.classifier import MentalHealthRiskClassifier
from src.services.crisis_detection import detect_crisis_signal, get_crisis_response, DEFAULT_CRISIS_LANGUAGE

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
        # Second, heavier crisis-screening signal (see Step 0b in get_advice) —
        # cheap to construct, only trains its Random Forest lazily on first use.
        self.risk_classifier = MentalHealthRiskClassifier()

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
            csv.field_size_limit(10 * 1024 * 1024)
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

    def _build_search_query(self, user_query: str, history: list = None, max_turns: int = 3) -> str:
        """Fold recent user turns into the query so follow-up questions (e.g. pronouns,
        'what about the property?') retrieve relevant cases instead of only the latest sentence."""
        if not history:
            return user_query
        recent_user_turns = [h.get('text', '') for h in history if h.get('role') == 'user'][-max_turns:]
        recent_user_turns = [t for t in recent_user_turns if t]
        if not recent_user_turns:
            return user_query
        return " ".join(recent_user_turns + [user_query])

    def _format_history_for_prompt(self, history: list, max_turns: int = 6) -> str:
        if not history:
            return "Սա այս զրույցի առաջին հարցն է։"
        trimmed = history[-max_turns:]
        lines = []
        for turn in trimmed:
            speaker = "Հաճախորդ" if turn.get('role') == 'user' else "Օգնական"
            text = (turn.get('text') or '').strip()
            if text:
                lines.append(f"{speaker}: {text}")
        return "\n".join(lines) if lines else "Սա այս զրույցի առաջին հարցն է։"

    def _build_similar_cases_block(self, search_query: str, exclude_unique_number: str = None, limit: int = 3) -> str:
        """Render a compact list of similar cases (lawyer name + approved marker) so the
        chat answer surfaces useful precedents inline, without a separate search step."""
        if not self.classifier:
            return ""
        try:
            similar_cases = self.classifier.find_multiple_similar_cases(search_query, limit=limit + 1)
        except Exception as ex:
            print(f"⚠️ Error finding similar cases for inline block: {ex}")
            return ""

        if exclude_unique_number:
            similar_cases = [c for c in similar_cases if c.get('unique_number') != exclude_unique_number]
        similar_cases = similar_cases[:limit]
        if not similar_cases:
            return ""

        lines = ["\n📚 Նմանատիպ գործեր (օգտակար օրինակներ).\n"]
        for idx, case in enumerate(similar_cases, 1):
            lawyer_name = case.get('lawyer_name') or 'Նշված չէ'
            approved_mark = " ✅ Հաստատված" if case.get('is_approved') else ""
            lines.append(
                f"   {idx}. {case.get('unique_number', 'N/A')} — {case.get('civil_case_classifier', 'N/A')}\n"
                f"      Փաստաբան: {lawyer_name}{approved_mark}\n"
            )
        return "".join(lines)

    def get_advice(self, user_query: str, history: list = None, language: str = DEFAULT_CRISIS_LANGUAGE) -> str:
        """
        Processes the legal query by routing it through the classifier first,
        then a strict vector match, and falling back to a structured RAG pipeline.

        :param history: optional list of {"role": "user"|"bot", "text": str} prior turns
            in this conversation, used to resolve follow-up questions and give the LLM
            fallback path real conversational context.
        :param language: short language code ("hy", "en", ...) for the crisis
            response and the LLM-drafted RAG-fallback answer (see
            src/agents/legal_crew.py). The classifier-match/vector-match
            templates below (Steps 2-3) are fixed Armenian text regardless of
            this value — only the free-text LLM-drafted answer and the crisis
            response are actually localized right now.
        """
        history = history or []

        # Step 0: Crisis/safety check — takes priority over everything else below,
        # including the short-query clarification check, since a crisis message can
        # be very short ("ուզում եմ մեռնել"). See src/services/crisis_detection.py.
        if detect_crisis_signal(user_query):
            return get_crisis_response(language)

        # Step 0b: Random Forest mental-health risk screen — a second, heavier
        # signal that catches phrasings the fixed keyword list in Step 0 misses.
        # Only widens coverage (adds a positive), never overrides a keyword hit;
        # if training data is unavailable or classification fails,
        # classify_mental_health_risk returns None and we fall through to
        # normal legal-advice handling below. See MentalHealthRiskClassifier in
        # src/services/classifier.py.
        if self.risk_classifier:
            risk = self.risk_classifier.classify_mental_health_risk(user_query)
            if risk and risk["is_risk"]:
                return get_crisis_response(language)

        # Step 1: Interactive check / Clarification hook
        if len(user_query.split()) < 3:
            return "Խնդրում եմ, նկարագրեք ձեր իրավական խնդիրը մի փոքր ավելի մանրամասն, որպեսզի կարողանամ ճշգրիտ նախադեպեր գտնել:"

        search_query = self._build_search_query(user_query, history)

        # Step 2: Try the Classifier first (even for voice/text)
        if self.classifier:
            try:
                matched_case = self.classifier.find_similar_case(search_query)
                if matched_case:
                    lawyer = matched_case.get('lawyer_name')
                    lawyer_display = lawyer if lawyer and lawyer != "(NULL)" else "Նշված չէ"
                    case_excerpt = self._truncate_text(matched_case.get('judicial_prehistory', ''), max_chars=1200)

                    top_lawyer_block = ""
                    top_lawyer = self.classifier.get_top_lawyer_for_query(search_query)
                    if top_lawyer and top_lawyer['approved_cases'] > 0:
                        top_lawyer_block = (
                            f"\n🏆 Ամենահաջողակ փաստաբանը նմանատիպ գործերում: {top_lawyer['lawyer_name']}\n"
                            f"   Հաստատված գործեր: {top_lawyer['approved_cases']} "
                            f"(ընդհանուր {top_lawyer['total_similar_cases']} նմանատիպ գործից)\n"
                        )

                    similar_cases_block = self._build_similar_cases_block(
                        search_query, exclude_unique_number=matched_case.get('unique_number')
                    )

                    return (
                        f"🎯 [CLASSIFIER MATCH FOUND]\n"
                        f"🔹 Դասակարգում: {matched_case.get('civil_case_classifier')}\n"
                        f"🔹 Նմանատիպ գործ: {matched_case.get('unique_number')}\n"
                        f"🔹 Հղում: {matched_case.get('link')}\n"
                        f"🔹 Առաջարկվող փաստաբան: {lawyer_display}\n"
                        f"{top_lawyer_block}"
                        f"{similar_cases_block}\n"
                        f"📄 Գործի նախապատմություն / բովանդակության օրինակ:\n{case_excerpt}\n"
                        f"\nՓոխարենը կարող էք բացել հղումը՝ ամբողջ գործը ընթերցելու համար։"
                    )
            except Exception as ex:
                print(f"⚠️ Error during case classification: {ex}")

        # Step 3: Try Exact Vector DB Precedent Search
        try:
            results = self.repo.db.similarity_search_with_score(search_query, k=1)
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
        return self._generate_rag_response(user_query, history=history, search_query=search_query, language=language)

    def _generate_rag_response(self, query: str, history: list = None, search_query: str = None, language: str = DEFAULT_CRISIS_LANGUAGE) -> str:
        search_query = search_query or query
        try:
            print(f"🔍 Starting RAG response generation for query: {query[:50]}...")
            results = self.repo.db.similarity_search_with_score(search_query, k=3)
            relevant_docs = [doc for doc, score in results if score < 0.8]
            print(f"📚 Found {len(results)} documents in vector DB, {len(relevant_docs)} above relevance threshold")
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Find relevant court cases from CSV
            relevant_cases = self._find_relevant_cases(search_query, limit=2)
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

            # If LLM is available, use a researcher+writer crew to generate a proper response
            if self.llm:
                try:
                    print(f"📝 Generating response via legal crew using model: {self.model_name}")
                    conversation_context = self._format_history_for_prompt(history)

                    from src.agents.legal_crew import run_legal_crew
                    print("⏳ Waiting for crew response...")
                    response = run_legal_crew(
                        query=query,
                        context=context,
                        cases_context=cases_context,
                        conversation_context=conversation_context,
                        model_name=self.model_name,
                        language=language,
                    )
                    print(f"✅ Crew response received ({len(response)} characters)")
                    similar_cases_block = self._build_similar_cases_block(search_query)
                    return response + similar_cases_block if similar_cases_block else response
                except Exception as llm_error:
                    print(f"❌ Legal crew generation error: {llm_error}")
                    print(f"   Model: {self.model_name}")
                    # Fall back to template response if the crew fails
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
                
                # Export cases (same case type/links) to a Word document
                export_path = self.export_service.export_similar_cases_docx(similar_cases, query)
                
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

            # Get top lawyers by cases (ranked overall, most approved cases first)
            top_lawyers = self.classifier.get_top_lawyers_by_cases(limit=10)

            # Export approved cases + lawyer ranking to a Word document
            export_path = self.export_service.export_approved_cases_docx(approved_cases, top_lawyers)
            
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
                # Export to a Word document
                export_path = self.export_service.export_similar_cases_docx(cases, f"Cases by {lawyer_name}")
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