import argparse
import csv
import json
import sys
from pathlib import Path


def configure_csv_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit = limit // 10


def main():
    parser = argparse.ArgumentParser(description="Predict labels for cases using keyword rule model.")
    parser.add_argument("--model", required=True, help="JSON rules model path.")
    parser.add_argument("--input-csv", required=True, help="Combined CSV path.")
    parser.add_argument("--output-csv", required=True, help="Predictions output CSV path.")
    args = parser.parse_args()

    with Path(args.model).open("r", encoding="utf-8") as f:
        model = json.load(f)
    rules = model["rules"]
    default_label = model.get("default_label", "OTHER")

    configure_csv_limit()
    rows = []
    with Path(args.input_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    preds = []
    for row in rows:
        low = row["text"].lower()
        pred = default_label
        for label, keywords in rules.items():
            if any(k in low for k in keywords):
                pred = label
                break
        preds.append(pred)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_number", "source_file", "predicted_label"]
        )
        writer.writeheader()
        for row, pred in zip(rows, preds):
            writer.writerow(
                {
                    "case_number": row["case_number"],
                    "source_file": row["source_file"],
                    "predicted_label": pred,
                }
            )

    print(f"Saved predictions for {len(rows)} cases to {output_path}")


if __name__ == "__main__":
    main()
