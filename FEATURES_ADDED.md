# New Features Added to Armenian Legal AI

## Overview
Added functionality to find similar legal cases and display approved cases with lawyer information. Cases are automatically exported to text files for easy review.

---

## New Features

### 1. **Similar Cases Search** [Press `s`]
Find legal cases similar to a user's query and export them as text files.

**Features:**
- Searches for multiple similar cases (up to 5)
- Shows similarity score for each case
- Exports all results to a timestamped text file
- Includes case number, lawyer name, classification, and link

**How to use:**
1. Press `s` in the main menu
2. Enter your case description or legal issue
3. Press ENTER twice to search
4. Cases are automatically exported to `exports/similar_cases_YYYYMMDD_HHMMSS.txt`

---

### 2. **Approved Cases & Top Lawyers** [Press `a`]
Display approved/successful legal cases and ranking of top lawyers by successful case count.

**Features:**
- Lists all approved/successful cases from the database
- Shows top 10 lawyers ranked by number of successful cases
- Includes case statistics and sample cases for each lawyer
- Full report exported to `exports/approved_cases_YYYYMMDD_HHMMSS.txt`

**How to use:**
1. Press `a` in the main menu
2. System automatically searches approved cases
3. Displays top lawyers with their case counts
4. Detailed report exported to text file

**Detection Logic:**
Cases are marked as "approved" if they contain keywords like:
- Armenian: հաստատել, հաստատվել, հաճախել, մեղադրանք, դատել
- English: approved, success, successful, granted, upheld, affirm, confirm

---

### 3. **Automatic Top Lawyer for Typed/Spoken Questions**
When a typed (`t`) or spoken (`m`) question matches a known case via the classifier, the answer now automatically includes the lawyer with the strongest approved-case track record among similar cases — no separate lookup step required.

**Features:**
- Ranks lawyers among cases similar to the query, not just the single closest match
- Shows approved-case count out of total similar cases considered
- Appears inline in the classifier-match response, alongside the originally matched case's recommended lawyer

**How to use:**
1. Press `t` and type your legal question (or `m` to speak it)
2. If a classifier match is found, the response includes a "🏆 Ամենահաջողակ փաստաբանը" (most successful lawyer) block

The standalone "search cases by lawyer" control (previously `l`) has been removed since this information now surfaces automatically as part of the answer.

---

## File Structure

### New Files Created:
1. **`src/services/case_export.py`** - CaseExportService class
   - `export_similar_cases()` - Export similar cases to text
   - `export_approved_cases()` - Export approved cases to text
   - `get_export_directory()` - Get exports folder path
   - `list_exports()` - List all exported files

### Modified Files:
1. **`src/services/classifier.py`** - Enhanced LegalCaseClassifier
   - `find_multiple_similar_cases(text, limit)` - Get multiple similar cases
   - `_is_approved_case(case)` - Detect approved cases
   - `find_approved_cases(limit)` - Get all approved cases
   - `find_cases_by_lawyer(name, limit)` - Get cases by lawyer
   - `get_top_lawyers_by_cases(limit)` - Get top lawyers ranking
   - `get_top_lawyer_for_query(text, search_limit)` - Rank lawyers by approved cases among cases similar to a query

2. **`src/agents/legal_agent.py`** - Enhanced LegalAgent
   - Added `CaseExportService` initialization
   - `get_similar_cases(query, limit)` - Find and export similar cases
   - `get_approved_cases_with_lawyers(limit)` - Get approved cases with lawyer stats
   - `get_lawyer_cases(name, limit)` - Find cases by lawyer
   - `format_similar_cases_response(cases)` - Format similar cases for display
   - `format_approved_cases_response(result)` - Format approved cases for display
   - `get_advice(user_query)` - Classifier-match responses now include the top lawyer for similar cases automatically

3. **`src/main.py`** - Updated LegalAIController
   - `handle_similar_cases()` - Handler for similar cases search
   - `handle_approved_cases()` - Handler for approved cases display
   - Updated keyboard controls and menu (standalone lawyer-search control removed)
   - Updated action handler in main loop

---

## Export Files

All cases are automatically exported to the `exports/` directory with timestamps:

### Similar Cases Export:
- **Filename:** `similar_cases_YYYYMMDD_HHMMSS.txt`
- **Content:**
  - Original search query
  - Number of cases found
  - For each case:
    - Case number
    - Lawyer name
    - Classification
    - DataLex link
    - Judicial prehistory (if available)

### Approved Cases Export:
- **Filename:** `approved_cases_YYYYMMDD_HHMMSS.txt`
- **Content:**
  - Total approved cases count
  - Number of unique lawyers
  - For each case:
    - Case number
    - Lawyer name (emphasized)
    - Lawyer department
    - Classification
    - Verdict type
    - Case description

---

## Usage Examples

### Example 1: Find Similar Cases
```
Press: s
Input: "Տեղավճար վեճ բնակելի տանը"
Output: 
  ✨ Գտնվել է 3 նման դատական գործ
  📌 ԳՈՐԾ #1 (համընկնում 85%)
     🔢 Համար: ՏԱ/0845/12/19
     ⚖️ Փաստաբան: Հայկ Վարդանյան
     📋 Դասակարգում: Քաղաքացիական
     ...
```

### Example 2: View Approved Cases
```
Press: a
Output:
  ✅ ՀԱՍՏԱՏՎԱԾ/ՀԱՋՈՂՎԱԾ ԴԱՏԱԿԱՆ ԳՈՐԾԵՐ
  Ընդամենը հաստատված գործեր: 42
  
  👨‍⚖️ TOP ՓԱՍՏԱԲԱՆՆԵՐ (10)
  #1 ⭐ Հայկ Վարդանյան
      Հաջողված գործեր: 8
  #2 ⭐ Արամ Հայրապետյան
      Հաջողված գործեր: 7
  ...
```

### Example 3: Typed Question with Automatic Top Lawyer
```
Press: t
Input: "Ժառանգություն ընդունելու վերաբերյալ գործ"
Output:
  🎯 [CLASSIFIER MATCH FOUND]
  🔹 Դասակարգում: 8.7.5 Ժառանգությունն ընդունելու և ժառանգ...
  🔹 Առաջարկվող փաստաբան: Հայկ Վարդանյան

  🏆 Ամենահաջողակ փաստաբանը նմանատիպ գործերում: Հայկ Վարդանյան
     Հաստատված գործեր: 5 (ընդհանուր 5 նմանատիպ գործից)
  ...
```

---

## Keyboard Shortcuts Summary

| Key | Action |
|-----|--------|
| **m** | Speak via microphone |
| **t** | Type your legal question |
| **u** | Upload a legal document |
| **s** | **[NEW]** Find similar cases |
| **a** | **[NEW]** Show approved cases & top lawyers |
| **q** | Quit the application |

Typed (`t`) and spoken (`m`) questions that match a known case now automatically include the top lawyer for similar cases in the answer — no separate control needed.

---

## Technical Details

### Similarity Scoring
- Uses TF-IDF vectorization from `sklearn`
- Cosine similarity threshold: > 0.1
- Returns cases sorted by similarity score (highest first)

### Approval Detection
- Scans case text for predefined keywords in Armenian and English
- Keywords include: approval terms, verdict types, case outcomes
- Can be easily extended by adding more keywords

### Export Formatting
- UTF-8 encoding with Armenian text support
- Organized sections with clear visual separators
- Includes metadata and timestamps
- Easy to read in any text editor

---

## Integration Notes

The new features are fully integrated with:
- ✅ Existing classifier system
- ✅ Vector database (ChromaDB)
- ✅ LLM (Ollama)
- ✅ Export directory management
- ✅ Keyboard event handling
- ✅ Unicode normalization for Armenian text

---

## Future Enhancements

Possible improvements:
1. Add filtering by case category/classification
2. Add filtering by verdict type (won/lost/pending)
3. Add statistical analysis of lawyer success rates
4. Export to PDF format
5. Add database search by date range
6. Add case similarity visualization
7. Email export functionality

---

## Troubleshooting

**Issue:** "No similar cases found"
- **Solution:** Ensure data files are loaded in the `data/` folder with proper HTML format

**Issue:** "Classifier not available"
- **Solution:** Check that `data/prehistory*.htm` files exist and are properly formatted

**Issue:** Export file not created
- **Solution:** Ensure `exports/` directory exists or can be created (permissions check)

---

## Support

For issues or feature requests, check:
1. Console output for error messages
2. Data folder structure and file formats
3. Ollama service status
4. ChromaDB connection status
