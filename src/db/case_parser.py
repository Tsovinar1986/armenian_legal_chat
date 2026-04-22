import re

def classify_and_parse_cases(content):
    """
    Parses PHP-style array strings from caseList files into Python dictionaries.
    """
    cases = []
    # This regex finds the Case ID and the Verdict text
    pattern = re.compile(r"\[(\d+)\]\s*=>\s*\"(.*?)\"", re.DOTALL)
    matches = pattern.findall(content)

    for case_id, verdict in matches:
        # Simple logic to determine category based on keywords
        category = "General"
        v_lower = verdict.lower()
        if "ապտակ" in v_lower or "բռնություն" in v_lower:
            category = "Criminal (Ֆիզիկական բռնություն)"
        elif "գումար" in v_lower or "պարտք" in v_lower:
            category = "Civil (Գումարի պահանջ)"
        
        cases.append({
            "case_id": case_id,
            "verdict": verdict.strip(),
            "legal_category": category
        })
    return cases