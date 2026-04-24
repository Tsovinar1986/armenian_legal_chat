import argparse
import csv
import re
from pathlib import Path


CASE_HEADER_RE = re.compile(r"^Գործի համարը:\s*(.+)$", flags=re.MULTILINE)


def split_cases(text: str):
    matches = list(CASE_HEADER_RE.finditer(text))
    cases = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        case_number = match.group(1).strip()
        case_text = text[start:end].strip()
        if case_number and case_text:
            cases.append((case_number, case_text))
    return cases


def main():
    parser = argparse.ArgumentParser(description="Combine extracted case text files into one CSV.")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_extracted.txt files.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("caseList*_extracted.txt"))
    if not input_files:
        raise FileNotFoundError(f"No matching files found in {input_dir}")

    rows = []
    for file_path in input_files:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        for case_number, case_text in split_cases(text):
            rows.append(
                {
                    "case_number": case_number,
                    "source_file": file_path.name,
                    "text": case_text,
                }
            )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_number", "source_file", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {output_csv} with {len(rows)} cases.")


if __name__ == "__main__":
    main()
