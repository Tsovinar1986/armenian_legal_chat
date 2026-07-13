"""Heuristic crisis-signal detection for self-harm / suicide risk language.

This is a conservative keyword safety net, not a clinical assessment tool. It
can both miss real risk (false negatives) and over-trigger on unrelated text
that happens to contain a matching phrase (false positives). Its only job is
to interrupt the normal legal-advice flow and point the user toward real
emergency/crisis resources instead of continuing to search case law when
someone may be in danger. It must never be treated as, or presented as, a
clinical risk assessment.

IMPORTANT: CRISIS_RESPONSE_HY below includes emergency contact numbers for
Armenia (911 unified emergency line; 102 police; 103 ambulance, per publicly
known numbers). Verify these are current and correct for your deployment
before relying on this in production, and consider adding a dedicated local
mental-health crisis line if/when you have a verified one to include.
"""

_CRISIS_KEYWORDS = [
    # Armenian
    "ինքնասպանություն", "ինքնասպան", "ինքնավնաս",
    "ուզում եմ մեռնել", "ուզում եմ ինձ սպանել", "ուզում եմ մահանալ",
    "կյանքս չարժե", "ապրելու իմաստ չկա", "վերջ տալ կյանքիս",
    "ինձ սպանել", "ինձ վնասել եմ ուզում",
    # English
    "suicide", "suicidal", "kill myself", "want to die", "end my life",
    "self harm", "self-harm", "hurt myself", "no reason to live",
]


def detect_crisis_signal(text: str) -> bool:
    """Return True if text contains a self-harm/suicide risk keyword or phrase.

    Simple, explainable substring matching — deliberately not an ML model, so
    behavior stays auditable and easy to extend with new keywords.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in _CRISIS_KEYWORDS)


CRISIS_RESPONSE_HY = (
    "⚠️ Ձեր հաղորդագրության մեջ նկատվում են արտահայտություններ, որոնք կարող են վկայել "
    "ծանր հուզական ճգնաժամի կամ ինքնավնասման ռիսկի մասին։\n\n"
    "Եթե Դուք անմիջական վտանգի տակ եք կամ մտածում եք ինքնասպանության մասին, "
    "խնդրում ենք անհապաղ դիմել՝\n"
    "🚨 Շտապ օգնության միասնական ծառայություն՝ 911\n"
    "👮 Ոստիկանություն՝ 102   🚑 Շտապ բժշկական օգնություն՝ 103\n\n"
    "Խնդրում ենք նաև անհապաղ կապվել վստահելի մարդու հետ (ընտանիքի անդամ, ընկեր) "
    "կամ դիմել որակավորված հոգեբանի/թերապևտի։\n\n"
    "Այս համակարգը իրավաբանական խորհրդատվության գործիք է և ՉԻ կարող փոխարինել "
    "շտապ բժշկական կամ հոգեբանական օգնությանը։"
)

CRISIS_RESPONSE_EN = (
    "⚠️ Your message contains language that may indicate a serious emotional crisis "
    "or risk of self-harm.\n\n"
    "If you are in immediate danger or thinking about suicide, please reach out right away:\n"
    "🚨 Armenia unified emergency line: 911\n"
    "👮 Police: 102   🚑 Ambulance: 103\n\n"
    "If you are outside Armenia, please contact your local emergency number instead.\n\n"
    "Please also reach out immediately to someone you trust (a family member or friend), "
    "or a qualified psychologist/therapist.\n\n"
    "This system is a legal-advice tool and CANNOT substitute for emergency medical or "
    "psychological help."
)

# Keyed by IANA-ish short language code. Only hy/en are actually translated —
# add more entries here as real translations become available. Any other
# requested code falls back to English in get_crisis_response below, since
# that reaches more people than defaulting to Armenian-only.
CRISIS_RESPONSES = {
    "hy": CRISIS_RESPONSE_HY,
    "en": CRISIS_RESPONSE_EN,
}

DEFAULT_CRISIS_LANGUAGE = "hy"


def get_crisis_response(language: str = DEFAULT_CRISIS_LANGUAGE) -> str:
    """Return the crisis response text for the given language code.

    Falls back to English for any code without a real translation (see
    CRISIS_RESPONSES) — a generic English safety message is more useful to an
    unsupported-language user than an Armenian-only one they may not read.
    """
    return CRISIS_RESPONSES.get(language, CRISIS_RESPONSE_EN)
