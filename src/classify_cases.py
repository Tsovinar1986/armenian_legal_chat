import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


RULE_LABELS = {
    "DEBT_COLLECTION": ["գումարի բռնագանձման", "բռնագանձել"],
    "DIVORCE_FAMILY": ["ամուսնալուծ", "ամուսնությունը լուծ", "խնամակալ"],
    "PROPERTY_DISPUTE": ["անշարժ գույք", "սեփականության իրավունք"],
    "CAPACITY_CASE": ["անգործունակ", "դատահոգեբուժական"],
    "SETTLEMENT_APPROVAL": ["հաշտության համաձայնություն", "հաշտարար"],
}


def weak_label(text: str) -> str:
    low = text.lower()
    for label, keywords in RULE_LABELS.items():
        if any(k in low for k in keywords):
            return label
    return "OTHER"


def configure_csv_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit = limit // 10


def main():
    parser = argparse.ArgumentParser(description="Create rule-based classification labels for case texts.")
    parser.add_argument("--input-csv", required=True, help="Combined CSV path.")
    parser.add_argument("--model-out", required=True, help="Output JSON rules/model path.")
    parser.add_argument("--labeled-out", required=True, help="Output CSV with assigned labels.")
    args = parser.parse_args()

    configure_csv_limit()
    rows = []
    with Path(args.input_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    labels = []
    for row in rows:
        labels.append(weak_label(row["text"]))

    label_counts = Counter(labels)
    print("Assigned label counts:")
    for label, count in label_counts.most_common():
        print(f"  - {label}: {count}")

    labeled_path = Path(args.labeled_out)
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    with labeled_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_number", "source_file", "label", "text"]
        )
        writer.writeheader()
        for row, label in zip(rows, labels):
            writer.writerow(
                {
                    "case_number": row["case_number"],
                    "source_file": row["source_file"],
                    "label": label,
                    "text": row["text"],
                }
            )

    print(f"Labeled data saved to {labeled_path}")

    model_payload = {"type": "keyword_rules", "rules": RULE_LABELS, "default_label": "OTHER"}
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("w", encoding="utf-8") as f:
        json.dump(model_payload, f, ensure_ascii=False, indent=2)
    print(f"Rule model saved to {model_path}")


if __name__ == "__main__":
    main()
