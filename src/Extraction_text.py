import re
import pandas as pd
import os

files = ['/Users/tsovinarbabakhanyan/Desktop/Armenian Chat part/src/data/caseList1.txt', '/Users/tsovinarbabakhanyan/Desktop/Armenian Chat part/src/data/caseList2.txt', '/Users/tsovinarbabakhanyan/Desktop/Armenian Chat part/src/data/caseList3.txt', '/Users/tsovinarbabakhanyan/Desktop/Armenian Chat part/src/data/caseList4.txt', '/Users/tsovinarbabakhanyan/Desktop/Armenian Chat part/src/data/caseList51.txt']
all_data = []

def extract_court_papers(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Step 1: Identify Judges
    judges = list(re.finditer(r'^\s{4}\[(.*?)\] => Array', content, re.MULTILINE))
    results = []

    for i in range(len(judges)):
        judge_name = judges[i].group(1)
        start = judges[i].end()
        end = judges[i+1].start() if i + 1 < len(judges) else len(content)
        judge_block = content[start:end]
        
        # Step 2: Identify Case Categories
        categories = list(re.finditer(r'^\s{12}\[(.*?)\] => Array', judge_block, re.MULTILINE))
        for j in range(len(categories)):
            category_name = categories[j].group(1)
            cat_start = categories[j].end()
            cat_end = categories[j+1].start() if j + 1 < len(categories) else len(judge_block)
            category_block = judge_block[cat_start:cat_end]
            
            # Step 3: Extract every unique court paper entry
            entries = list(re.finditer(r'^\s{20}\[\d+\] => Array', category_block, re.MULTILINE))
            for k in range(len(entries)):
                e_start = entries[k].end()
                e_end = entries[k+1].start() if k + 1 < len(entries) else len(category_block)
                entry_block = category_block[e_start:e_end]
                
                # Extract the Case Number and Verdict
                num_match = re.search(r'\[unique_number\] => (.*?)$', entry_block, re.MULTILINE)
                txt_match = re.search(r'\[verdict_text\] => (.*?)(?=\n\s{24}\)\n|\n\s{20}\)\n|\n\s{16}\)\n|$)', entry_block, re.DOTALL)
                
                results.append({
                    'Judge': judge_name,
                    'Category': category_name,
                    'Case_Number': num_match.group(1).strip() if num_match else "N/A",
                    'Verdict_Text': txt_match.group(1).strip() if txt_match else "N/A"
                })
    return results

# Process all and save
for f in files:
    if os.path.exists(f):
        all_data.extend(extract_court_papers(f))

pd.DataFrame(all_data).to_csv('court_papers_full.csv', index=False, encoding='utf-8')