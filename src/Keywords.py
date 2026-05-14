import pandas as pd
import re
from collections import Counter

# 1. Load the dataset
# Ensure the path matches your file location
file_path = '/Users/tsovinarbabakhanyan/Desktop/Armenian Chat part/src/data/legal_analysis_labeled.csv'
df = pd.read_csv(file_path)

def get_bigram_keywords(text, num_keywords=5):
    """
    Function to extract the top N most frequent bigrams from text.
    """
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return ""
    
    # Text Cleaning: remove punctuation and make lowercase
    clean_text = re.sub(r'[^\w\s]', '', text).lower()
    words = clean_text.split()
    
    # Create Bigrams (e.g., "word1 word2", "word2 word3")
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    
    # Count occurrences and take the most common ones
    counts = Counter(bigrams)
    most_common = [bg for bg, count in counts.most_common(num_keywords)]
    
    return ", ".join(most_common)

# 2. Add the keywords column
# This operation preserves all existing columns and adds 'keywords' as a new one
print("Processing text to extract bigram keywords...")
df['keywords'] = df['Verdict_Text'].apply(get_bigram_keywords)

# 3. Save the full updated dataframe to a new file
output_file = '/Users/tsovinarbabakhanyan/Desktop/Armenian Chat part/src/data/final_data_with_keywords.xlsx'
df.to_excel(output_file, index=False)

# 4. Display confirmation
print(f"Success! The file has been saved as: {output_file}")
print("\nFinal Columns:", df.columns.tolist())
print("\nPreview of the data:")
print(df.head())
