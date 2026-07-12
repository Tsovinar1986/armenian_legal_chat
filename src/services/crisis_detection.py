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
