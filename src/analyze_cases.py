import argparse
import csv
import sys
from collections import Counter
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
    parser = argparse.ArgumentParser(description="Basic analysis for combined Armenian case dataset.")
    parser.add_argument("--input-csv", required=True, help="Combined CSV path.")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    configure_csv_limit()
    rows = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    source_counts = Counter(r["source_file"] for r in rows)
    lengths = [len(r["text"]) for r in rows]
    word_counts = [len(r["text"].split()) for r in rows]

    print(f"Total cases: {total}")
    print(f"Unique sources: {len(source_counts)}")
    print("\nCases per source file:")
    for name, count in source_counts.items():
        print(f"  - {name}: {count}")

    if total:
        print("\nText length stats (characters):")
        print(f"  - min: {min(lengths)}")
        print(f"  - avg: {sum(lengths) / total:.2f}")
        print(f"  - max: {max(lengths)}")

        print("\nToken count stats (space-split):")
        print(f"  - min: {min(word_counts)}")
        print(f"  - avg: {sum(word_counts) / total:.2f}")
        print(f"  - max: {max(word_counts)}")


if __name__ == "__main__":
    main()
