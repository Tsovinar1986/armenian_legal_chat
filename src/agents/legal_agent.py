# src/agents/legal_agent.py
import urllib.parse
import os
import csv
from langchain_ollama import OllamaLLM
from src.services.case_export import CaseExportService
from src.services.classifier import MentalHealthRiskClassifier
from src.services.crisis_detection import detect_crisis_signal, get_crisis_response, DEFAULT_CRISIS_LANGUAGE
# src/agents/legal_agent.py -> src/agents -> src -> project root. Used to
# anchor _load_court_cases' CSV path so it resolves regardless of the
# process's cwd at launch -- see the matching _PROJECT_ROOT comment in
# src/services/classifier.py for why this matters (a cwd mismatch here
# silently empties court_cases, which starves the RAG fallback and makes
# every query return "no local precedents found").
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Display names for the direct-LLM-answer prompt's language instruction. Any
# code not listed here is passed through as-is (e.g. "fr") — the local Ollama
# model's actual fluency in languages beyond Armenian/English is unverified,
# so this is a best-effort instruction, not a guarantee of quality.
_LANGUAGE_NAMES = {
    "hy": "Armenian",
    "en": "English",
    "ru": "Russian",
}

# Fixed template strings for the deterministic classifier-match/vector-match
# responses (Steps 2-3 in get_advice, plus the Step 1 short messages) — these
# aren't LLM-generated, so they can't "just respond in the requested
# language" the way the drafted answer does. hy/en/ru have real
# translations; any other requested code falls back to English, same
# convention as get_crisis_response in crisis_detection.py.
_TEMPLATE_TEXT = {
    "hy": {
        "clarify_short_query": "Խնդրում եմ, նկարագրեք ձեր իրավական խնդիրը մի փոքր ավելի մանրամասն, որպեսզի կարողանամ ճշգրիտ նախադեպեր գտնել:",
        "not_specified": "Նշված չէ",
        "classifier_match_header": "🎯 [CLASSIFIER MATCH FOUND]",
        "classification_label": "Դասակարգում",
        "similar_case_label": "Նմանատիպ գործ",
        "link_label": "Հղում",
        "recommended_lawyer_label": "Առաջարկվող փաստաբան",
        "top_lawyer_header": "🏆 Ամենահաջողակ փաստաբանը նմանատիպ գործերում",
        "approved_cases_label": "Հաստատված գործեր",
        "of_total_similar": "ընդհանուր {total} նմանատիպ գործից",
        "case_history_label": "📄 Գործի նախապատմություն / բովանդակության օրինակ",
        "open_link_footer": "Փոխարենը կարող էք բացել հղումը՝ ամբողջ գործը ընթերցելու համար։",
        "similar_cases_header": "📚 Նմանատիպ գործեր (օգտակար օրինակներ).",
        "lawyer_label": "Փաստաբան",
        "approved_marker": " ✅ Հաստատված",
        "vector_match_header": "✨ [Համընկնում է գտնվել նախադեպային բազայում]",
        "vector_match_intro": "Ես համակարգի տվյալների բազայում գտա այս իրավական հարցին համապատասխանող պատմական գործ։",
        "case_number_label": "🔢 Դատական գործի համարը",
        "datalex_link_label": "🌐 DataLex հղումը",
        "case_content_example_label": "📄 Գործի բովանդակության օրինակ",
        "vector_match_footer": "Մանրամասների համար բացեք հղումը կամ երկարացրեք որոնումը։",
        "no_local_precedents": "Համապատասխան տեղական իրավական նախադեպեր չգտնվեցին։ Խնդրում ենք համոզվել, որ ֆայլերը ճիշտ են ներբեռնված համակարգ։",
        "llm_unavailable": "Ինտելեկտուալ պատասխանի ծառայությունը ժամանակավորապես անհասանելի է։ Խնդրում ենք փորձել մի փոքր ուշ։",
    },
    "en": {
        "clarify_short_query": "Please describe your legal issue in a bit more detail so I can find accurate precedents.",
        "not_specified": "Not specified",
        "classifier_match_header": "🎯 [CLASSIFIER MATCH FOUND]",
        "classification_label": "Classification",
        "similar_case_label": "Similar case",
        "link_label": "Link",
        "recommended_lawyer_label": "Recommended lawyer",
        "top_lawyer_header": "🏆 Most successful lawyer in similar cases",
        "approved_cases_label": "Approved cases",
        "of_total_similar": "out of {total} similar cases",
        "case_history_label": "📄 Case history / content excerpt",
        "open_link_footer": "You can also open the link to read the full case.",
        "similar_cases_header": "📚 Similar cases (useful examples).",
        "lawyer_label": "Lawyer",
        "approved_marker": " ✅ Approved",
        "vector_match_header": "✨ [Match found in the precedent database]",
        "vector_match_intro": "I found a historical case in the system's database matching this legal question.",
        "case_number_label": "🔢 Court case number",
        "datalex_link_label": "🌐 DataLex link",
        "case_content_example_label": "📄 Case content excerpt",
        "vector_match_footer": "Open the link for details, or refine your search.",
        "no_local_precedents": "No relevant local legal precedents were found. Please make sure the files are correctly uploaded to the system.",
        "llm_unavailable": "The AI answer service is temporarily unavailable. Please try again in a moment.",
    },
    "ru": {
        "clarify_short_query": "Пожалуйста, опишите Вашу юридическую проблему немного подробнее, чтобы я мог найти точные прецеденты.",
        "not_specified": "Не указано",
        "classifier_match_header": "🎯 [НАЙДЕНО СОВПАДЕНИЕ КЛАССИФИКАТОРА]",
        "classification_label": "Классификация",
        "similar_case_label": "Похожее дело",
        "link_label": "Ссылка",
        "recommended_lawyer_label": "Рекомендуемый адвокат",
        "top_lawyer_header": "🏆 Самый успешный адвокат по похожим делам",
        "approved_cases_label": "Выигранные дела",
        "of_total_similar": "из {total} похожих дел",
        "case_history_label": "📄 История дела / пример содержания",
        "open_link_footer": "Вы также можете открыть ссылку, чтобы прочитать дело полностью.",
        "similar_cases_header": "📚 Похожие дела (полезные примеры).",
        "lawyer_label": "Адвокат",
        "approved_marker": " ✅ Выиграно",
        "vector_match_header": "✨ [Найдено совпадение в базе прецедентов]",
        "vector_match_intro": "Я нашёл в базе данных системы историческое дело, соответствующее этому юридическому вопросу.",
        "case_number_label": "🔢 Номер судебного дела",
        "datalex_link_label": "🌐 Ссылка DataLex",
        "case_content_example_label": "📄 Пример содержания дела",
        "vector_match_footer": "Откройте ссылку для подробностей или уточните поиск.",
        "no_local_precedents": "Соответствующие местные юридические прецеденты не найдены. Пожалуйста, убедитесь, что файлы правильно загружены в систему.",
        "llm_unavailable": "Сервис ответов ИИ временно недоступен. Пожалуйста, попробуйте ещё раз через некоторое время.",
    },
}


def _t(key: str, language: str) -> str:
    lang_dict = _TEMPLATE_TEXT.get(language) or _TEMPLATE_TEXT["en"]
    return lang_dict.get(key) or _TEMPLATE_TEXT["en"].get(key, key)

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
        csv_path = os.path.join(_PROJECT_ROOT, "src", "data", "court_papers_full.csv")
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

    def _build_similar_cases_block(self, search_query: str, exclude_unique_number: str = None, limit: int = 3, language: str = DEFAULT_CRISIS_LANGUAGE) -> str:
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

        lines = [f"\n{_t('similar_cases_header', language)}\n"]
        for idx, case in enumerate(similar_cases, 1):
            lawyer_name = case.get('lawyer_name') or _t('not_specified', language)
            approved_mark = _t('approved_marker', language) if case.get('is_approved') else ""
            lines.append(
                f"   {idx}. {case.get('unique_number', 'N/A')} — {case.get('civil_case_classifier', 'N/A')}\n"
                f"      {_t('lawyer_label', language)}: {lawyer_name}{approved_mark}\n"
            )
        return "".join(lines)

    def get_advice(self, user_query: str, history: list = None, language: str = DEFAULT_CRISIS_LANGUAGE) -> str:
        """
        Processes the legal query by routing it through the classifier first,
        then a strict vector match, and falling back to a structured RAG pipeline.

        :param history: optional list of {"role": "user"|"bot", "text": str} prior turns
            in this conversation, used to resolve follow-up questions and give the LLM
            fallback path real conversational context.
        :param language: short language code ("hy", "en", ...) for every
            response path: the crisis response, the LLM-drafted RAG-fallback
            answer (see _direct_llm_answer), and the deterministic
            classifier-match/vector-match templates (Steps 2-3, via the
            module-level _TEMPLATE_TEXT/_t helper). hy/en have real
            translations for the templates; any other code falls back to
            English, same convention as get_crisis_response.
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
        # if len(user_query.split()) < 3:
        #     return _t('clarify_short_query', language)

        search_query = self._build_search_query(user_query, history)

        # Step 2: Try the Classifier first (even for voice/text)
        #
        # Deliberately matched on user_query alone, NOT the history-folded
        # search_query: this step is a deterministic short-circuit that returns
        # one specific case's real details as the entire answer. Matching it
        # against several turns of concatenated history meant that once a case
        # matched on turn 1, every later reply in the session kept re-matching
        # the same case — a genuinely different follow-up question ("how much
        # time do I have to file?") would come back with the identical
        # first-turn case block, because the old turns' text still dominated
        # the TF-IDF vector. search_query stays reserved for steps 3/4 below,
        # where blending in context to widen retrieval is actually correct.
        if self.classifier:
            try:
                matched_case = self.classifier.find_similar_case(user_query)
                if matched_case:
                    lawyer = matched_case.get('lawyer_name')
                    lawyer_display = lawyer if lawyer and lawyer != "(NULL)" else _t('not_specified', language)
                    case_excerpt = self._truncate_text(matched_case.get('judicial_prehistory', ''), max_chars=1200)

                    top_lawyer_block = ""
                    top_lawyer = self.classifier.get_top_lawyer_for_query(user_query)
                    if top_lawyer and top_lawyer['approved_cases'] > 0:
                        top_lawyer_block = (
                            f"\n{_t('top_lawyer_header', language)}: {top_lawyer['lawyer_name']}\n"
                            f"   {_t('approved_cases_label', language)}: {top_lawyer['approved_cases']} "
                            f"({_t('of_total_similar', language).format(total=top_lawyer['total_similar_cases'])})\n"
                        )

                    similar_cases_block = self._build_similar_cases_block(
                        user_query, exclude_unique_number=matched_case.get('unique_number'), language=language
                    )

                    return (
                        f"{_t('classifier_match_header', language)}\n"
                        f"🔹 {_t('classification_label', language)}: {matched_case.get('civil_case_classifier')}\n"
                        f"🔹 {_t('similar_case_label', language)}: {matched_case.get('unique_number')}\n"
                        f"🔹 {_t('link_label', language)}: {matched_case.get('link')}\n"
                        f"🔹 {_t('recommended_lawyer_label', language)}: {lawyer_display}\n"
                        f"{top_lawyer_block}"
                        f"{similar_cases_block}\n"
                        f"{_t('case_history_label', language)}:\n{case_excerpt}\n"
                        f"\n{_t('open_link_footer', language)}"
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
                        response_text = f"{_t('vector_match_header', language)}\n"
                        response_text += f"{_t('vector_match_intro', language)}\n\n"
                        if case_number:
                            response_text += f"{_t('case_number_label', language)}: {case_number}\n"
                        if datalex_link:
                            response_text += f"{_t('datalex_link_label', language)}: {datalex_link}\n"
                        response_text += f"\n{_t('case_content_example_label', language)}:\n"
                        response_text += self._truncate_text(doc.page_content, max_chars=1200)
                        response_text += f"\n\n{_t('vector_match_footer', language)}"
                        return response_text
        except Exception as e:
            print(f"⚠️ Վեկտորային բազայի ստուգման սխալ: {e}")

        # Step 4: Fallback to general RAG synthesis
        return self._generate_rag_response(user_query, history=history, search_query=search_query, language=language)

    def _direct_llm_answer(self, query: str, context: str, cases_context: str, conversation_context: str, language: str) -> str:
        """Single direct LLM call that drafts the RAG-fallback answer.

        Previously this ran behind a crewai researcher+writer crew, with this
        as the fallback for when that crew failed. The crew's two-task chain
        (research, then write) was confirmed unreliable for real (long)
        prompts against this model — it consistently returned "Invalid
        response from LLM call - None or empty", most likely because
        armenia-lawyer-router's context_length is 8192 tokens and the crew's
        two calls effectively doubled prompt usage against that budget where
        a single call doesn't. crewai has since been removed entirely; this
        is now the only path.
        """
        language_name = _LANGUAGE_NAMES.get(language, language)
        prompt = (
            f"You are a senior Armenian legal consultant. Answer the client's question in "
            f"{language_name}, grounded only in the context below — do not invent case numbers, "
            "facts, or citations that aren't present here. If the context doesn't actually cover "
            "the question, say so plainly and give general legal guidance instead.\n\n"
            f"Conversation so far:\n{conversation_context}\n\n"
            f"Client's question: {query}\n\n"
            f"Retrieved precedent context:\n{self._truncate_text(context, max_chars=1500)}\n\n"
            f"Real court case examples:\n"
            f"{self._truncate_text(cases_context, max_chars=1000) if cases_context else 'None found.'}\n\n"
            f"Write a clear, structured answer in {language_name}."
        )
        return self.llm.invoke(prompt).strip()

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

            # Only bail out when NEITHER source found anything — the vector
            # store (chroma_legal_data) currently holds far fewer documents
            # than the court_papers_full.csv corpus _find_relevant_cases
            # searches (150 vs. ~2000), so gating on vector context alone
            # was discarding perfectly good CSV-sourced case examples and
            # returning "no precedents found" when cases_context had real
            # matches the whole time.
            if not context and not cases_context:
                return _t('no_local_precedents', language)

            # If LLM is available, generate a response with a single direct call
            conversation_context = self._format_history_for_prompt(history)
            if self.llm:
                try:
                    print(f"📝 Generating response via direct LLM call using model: {self.model_name}")
                    response = self._direct_llm_answer(query, context, cases_context, conversation_context, language)
                    print(f"✅ LLM response received ({len(response)} characters)")

                    similar_cases_block = self._build_similar_cases_block(search_query, language=language)
                    return response + similar_cases_block if similar_cases_block else response
                except Exception as llm_error:
                    print(f"❌ LLM generation error: {llm_error}")
                    print(f"   Model: {self.model_name}")
                    return _t('llm_unavailable', language)
            else:
                # If LLM is not available, return template response
                print(f"⚠️ LLM not available, using fallback response")
                return _t('llm_unavailable', language)
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