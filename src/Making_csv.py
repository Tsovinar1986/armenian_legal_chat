import pandas as pd
import re
import os

# List of your uploaded files
files = [
    'src/data/caseList1_extracted.txt',
    'src/data/caseList2_extracted.txt',
    'src/data/caseList3_extracted.txt',
    'src/data/caseList4_extracted.txt',
    'src/data/caseList51_extracted.txt'
]

data = []

def parse_case(text):
    # 1. Extract Case Number (e.g., ԵԴ2/3094/02/24)
    case_num = "N/A"
    cn_match = re.search(r'Գործի համարը:\s*([^\n\r]+)', text)
    if cn_match:
        case_num = cn_match.group(1).strip()
    
    # 2. Extract Parties (looking for 'ընդդեմ' which means 'against')
    parties = "N/A"
    party_match = re.search(r'([Ա-Ֆ][ա-ֆ]+\s+[Ա-Ֆ][ա-ֆ]+\s+[Ա-Ֆ][ա-ֆ]+).*?ընդդեմ\s+([^\n՝]+)', text)
    if party_match:
        parties = f"{party_match.group(1).strip()} vs {party_match.group(2).strip()}"
    
    # 3. Extract Claim (looking for 'պահանջի մասին' which means 'about the claim')
    claim = "N/A"
    claim_match = re.search(r'՝\s*([^\n\.]+պահանջի մասին)', text)
    if claim_match:
        claim = claim_match.group(1).strip()
    
    return {
        "Case_Number": case_num,
        "Parties": parties,
        "Claim_Type": claim,
        "Full_Document_Text": text.strip()  # This preserves the original text for your analysis
    }

# Process each file and store the results
parsed_results = []
for f_name in files:
    if os.path.exists(f_name):
        with open(f_name, 'r', encoding='utf-8') as f:
            content = f.read()
            parsed_results.append(parse_case(content))

# Create the DataFrame and export to CSV
df = pd.DataFrame(parsed_results)
df.to_csv('src/data/legal_analysis_full_text.csv', index=False, encoding='utf-8-sig')

print("CSV file created successfully!")