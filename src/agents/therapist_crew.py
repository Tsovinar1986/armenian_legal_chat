"""CrewAI-based researcher + writer crew for the therapist supportive-chat path.

Mirrors src/agents/legal_crew.py for the therapist/mental-health side: a
Researcher agent organizes a relevant past Q&A example (retrieved in Python via
the existing, unchanged MentalHealthQAClassifier.find_similar_answer, before
this crew runs), and a Writer agent drafts the final supportive response from
what the Researcher produced.

The researcher agent is deliberately NOT given a callable tool to fetch this
example itself — see src/agents/legal_crew.py's module docstring: the local
Ollama model in use (armenia-lawyer-router) does not support tool/function
calling. Retrieval stays in Python and is handed to the researcher as task
input instead.

Crisis/safety checks (src/services/crisis_detection.py and
LegalCaseClassifier.classify_mental_health_risk) run in main.py's
/api/therapist-chat handler BEFORE this crew is ever invoked, and this module
has no path back into that logic. That ordering matters more here than on the
legal side: the underlying counseling dataset itself contains "suicidal
thoughts"-labeled rows, so a crew-drafted response must never be allowed to
substitute for the real crisis response.

Requires crewai to be installed (see requirements.txt for the two-step install
— crewai is intentionally not a normal pip-resolved dependency there).
"""
from crewai import Agent, Crew, LLM, Process, Task

# Display names for the Writer agent's language instruction. Any code not
# listed here is passed through as-is — the local Ollama model's actual
# fluency beyond Armenian/English is unverified, so this is best-effort.
LANGUAGE_NAMES = {
    "hy": "Armenian",
    "en": "English",
}


def run_therapist_crew(message: str, qa_classifier, model_name: str, language: str = "en") -> str:
    """Run the therapist researcher+writer crew and return the final response text.

    :param message: the user's current message
    :param qa_classifier: a MentalHealthQAClassifier instance (or anything with
        a compatible find_similar_answer(text) method)
    :param model_name: the local Ollama model name (e.g. "armenia-lawyer-router")
    :param language: short language code for the drafted response (default "en"
        since the underlying counseling dataset is English-language)
    """
    language_name = LANGUAGE_NAMES.get(language, language)
    match = qa_classifier.find_similar_answer(message)
    if match:
        found_example = (
            f"Similar past question (topic: {match.get('label', 'unknown')}): {match['question']}\n"
            f"Past supportive answer: {match['answer']}"
        )
    else:
        found_example = "No sufficiently similar past conversation was found."

    llm = LLM(model=f"ollama/{model_name}")

    researcher = Agent(
        role="Peer Support Researcher",
        goal="Find the most relevant past supportive-counseling conversation for what the person just said.",
        backstory=(
            "Reads through a large archive of past supportive counseling conversations to find "
            "the example most relevant to what someone is going through right now."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    writer = Agent(
        role="Supportive Conversation Writer",
        goal=(
            f"Write a warm, brief, non-clinical supportive response in {language_name}, grounded "
            "in the researcher's findings, that always makes clear this is not a licensed therapist."
        ),
        backstory=(
            f"A compassionate peer-support writer who writes in {language_name}, never diagnoses, "
            "never claims to be a licensed therapist, and always gently points toward booking a "
            "real session for anything beyond a supportive conversation."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            f"The person's message: {message}\n\n"
            f"Retrieved past conversation:\n{found_example}\n\n"
            "Review the retrieved past conversation above and report whether it's actually "
            "relevant to the person's message, and if so, what makes it relevant."
        ),
        expected_output="The similar past question/answer found (or a note that none was found).",
        agent=researcher,
    )

    writing_task = Task(
        description=(
            f"The person just said: {message}\n\n"
            "Using the researcher's findings as inspiration (not a script to copy verbatim), "
            f"write a short, warm, supportive response in {language_name}. Do not diagnose. Do "
            "not claim to be a licensed therapist. End by gently mentioning that /therapist can "
            "be used to book a real session for anything beyond this conversation."
        ),
        expected_output=f"A short, warm, supportive response in {language_name} that is clearly not a diagnosis or licensed-therapist advice.",
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
