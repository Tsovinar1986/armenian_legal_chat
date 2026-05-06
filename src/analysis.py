import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median


DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "legal_analysis_full_text.csv"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent / "data" / "legal_analysis_summary.json"
)


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int(q * (len(sorted_values) - 1)))
    return sorted_values[idx]


def normalize_col(name: str) -> str:
    return name.lstrip("\ufeff").strip()


def tokenize_armenian_text(text: str) -> list[str]:
    # Keep Armenian words and drop very short tokens.
    tokens = re.findall(r"[Ա-Ֆա-ֆև]+", text.lower())
    return [tok for tok in tokens if len(tok) >= 3]


def main() -> None:
    csv.field_size_limit(sys.maxsize)

    case_numbers: list[str] = []
    parties_list: list[str] = []
    claim_types: list[str] = []
    text_lengths: list[int] = []
    word_counts: list[int] = []
    token_counter: Counter[str] = Counter()
    missing_counter: Counter[str] = Counter()

    with DATA_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [normalize_col(col) for col in (reader.fieldnames or [])]

        for row in reader:
            normalized_row = {normalize_col(k): (v or "") for k, v in row.items()}

            case_number = normalized_row.get("Case_Number", "").strip()
            parties = normalized_row.get("Parties", "").strip()
            claim_type = normalized_row.get("Claim_Type", "").strip()
            full_text = normalized_row.get("Full_Document_Text", "")

            if not case_number:
                missing_counter["Case_Number"] += 1
            if not parties:
                missing_counter["Parties"] += 1
            if not claim_type:
                missing_counter["Claim_Type"] += 1
            if not full_text:
                missing_counter["Full_Document_Text"] += 1

            case_numbers.append(case_number)
            parties_list.append(parties)
            claim_types.append(claim_type)
            text_lengths.append(len(full_text))

            words = full_text.split()
            word_counts.append(len(words))
            token_counter.update(tokenize_armenian_text(full_text))

    claim_counter = Counter([c for c in claim_types if c])

    sorted_text_lengths = sorted(text_lengths)
    sorted_word_counts = sorted(word_counts)

    summary = {
        "dataset_path": str(DATA_PATH),
        "record_count": len(case_numbers),
        "unique_case_numbers": len(set([c for c in case_numbers if c])),
        "unique_claim_types": len(claim_counter),
        "missing_values": {
            "Case_Number": missing_counter.get("Case_Number", 0),
            "Parties": missing_counter.get("Parties", 0),
            "Claim_Type": missing_counter.get("Claim_Type", 0),
            "Full_Document_Text": missing_counter.get("Full_Document_Text", 0),
        },
        "text_character_length_stats": {
            "min": min(sorted_text_lengths) if sorted_text_lengths else 0,
            "max": max(sorted_text_lengths) if sorted_text_lengths else 0,
            "mean": round(mean(sorted_text_lengths), 2) if sorted_text_lengths else 0,
            "median": median(sorted_text_lengths) if sorted_text_lengths else 0,
            "p90": percentile(sorted_text_lengths, 0.90),
            "p99": percentile(sorted_text_lengths, 0.99),
        },
        "text_word_count_stats": {
            "min": min(sorted_word_counts) if sorted_word_counts else 0,
            "max": max(sorted_word_counts) if sorted_word_counts else 0,
            "mean": round(mean(sorted_word_counts), 2) if sorted_word_counts else 0,
            "median": median(sorted_word_counts) if sorted_word_counts else 0,
            "p90": percentile(sorted_word_counts, 0.90),
            "p99": percentile(sorted_word_counts, 0.99),
        },
        "top_claim_types": claim_counter.most_common(10),
        "top_armenian_tokens": token_counter.most_common(30),
    }

    OUTPUT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Analysis complete.")
    print(f"Records: {summary['record_count']}")
    print(f"Unique claim types: {summary['unique_claim_types']}")
    print(f"Summary file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
