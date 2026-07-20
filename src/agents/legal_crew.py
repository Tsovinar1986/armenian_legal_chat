"""CrewAI-based researcher + writer crew for the legal RAG fallback path.

Replaces the single direct LLM call that used to live in
LegalAgent._generate_rag_response with two agents: a Researcher that organizes
context already retrieved by the caller (vector-DB precedent search and real
court-case examples, computed in Python before this crew runs), and a Writer
that drafts the final Armenian answer from what the Researcher produced.

The researcher agent is deliberately NOT given a callable tool to fetch this
context itself — the local Ollama model in use (armenia-lawyer-router) does
not support tool/function calling (confirmed: Ollama returns a 400,
"does not support tools", when a crewai Agent is given any `tools=[...]`).
Retrieval therefore stays exactly where it already was (in Python, before
run_legal_crew is called) and is simply handed to the researcher as task
input — this still yields two independent LLM calls/roles, just without
function-calling.

Crisis/safety checks (src/services/crisis_detection.py and
LegalCaseClassifier.classify_mental_health_risk) run in LegalAgent.get_advice()
BEFORE this crew is ever invoked (Step 0/0b), and this module has no path back
into that logic — a multi-agent crew must never sit in the crisis-response path.

Requires crewai to be installed (see requirements.txt for the two-step install
— crewai is intentionally not a normal pip-resolved dependency there).
"""
from crewai import Agent, Crew, LLM, Process, Task

# Display names for the Writer agent's language instruction. Any code not
# listed here is passed through as-is (e.g. "fr") — the local Ollama model's
# actual fluency in languages beyond Armenian/English is unverified, so this
# is a best-effort instruction, not a guarantee of quality.
LANGUAGE_NAMES = {
    "hy": "Armenian",
    "en": "English",
    "ru": "Russian",
}


def run_legal_crew(
    query: str,
    context: str,
    cases_context: str,
    conversation_context: str,
    model_name: str,
    language: str = "hy",
) -> str:
    """Run the legal researcher+writer crew and return the final answer text.

    :param query: the client's current question
    :param context: precomputed vector-DB precedent text (already retrieved by
        the caller — the researcher agent's tool just surfaces it)
    :param cases_context: precomputed real-court-case example text
    :param conversation_context: formatted recent conversation history
    :param model_name: the local Ollama model name (e.g. "armenia-lawyer-router")
    :param language: short language code for the drafted answer (default "hy"
        for backward compatibility with existing Armenian-only callers)
    """
    language_name = LANGUAGE_NAMES.get(language, language)
    llm = LLM(model=f"ollama/{model_name}")

    researcher = Agent(
        role="Legal Case Researcher",
        goal=(
            "Find and organize the most relevant Armenian legal precedents and real "
            "court-case examples for the client's question."
        ),
        backstory=(
            "An experienced Armenian legal researcher who reads through case law and prior "
            "verdicts to find precedents relevant to a new client question, always noting how "
            "each precedent actually relates to the question asked."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    writer = Agent(
        role="Legal Advisor",
        goal=(
            f"Write a clear, professional answer in {language_name} to the client's legal "
            "question, grounded only in the researcher's findings and the conversation so far."
        ),
        backstory=(
            f"A senior legal advisor who explains Armenian legal matters to clients in plain, "
            f"precise {language_name}, referencing relevant precedents without inventing facts "
            "not found in the research."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            f"Client question: {query}\n\n"
            f"Precedent context retrieved from the case database:\n{context}\n\n"
            f"Real court case examples:\n{cases_context or 'None found.'}\n\n"
            "Review the material above and summarize the precedents and case examples that are "
            "actually relevant to the client's question, in a few bullet points. Discard anything "
            "irrelevant."
        ),
        expected_output="A short bullet-point summary of the relevant precedents and case examples found.",
        agent=researcher,
    )

    writing_task = Task(
        description=(
            f"Conversation so far:\n{conversation_context}\n\n"
            f"Client's new question: {query}\n\n"
            f"Using the researcher's findings as your only factual grounding, write the final "
            f"answer in {language_name}: concrete, structured, understandable, and referencing "
            "similar court cases where relevant. Do not invent case numbers or facts not present "
            "in the research."
        ),
        expected_output=f"A complete, professional answer in {language_name} to the client's question.",
        agent=writer,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    return str(result)
