from pathlib import Path
import re

def extract_verdict_texts(input_path: str, output_path: str) -> None:
    text = Path(input_path).read_text(encoding="utf-8", errors="ignore")

    unique_numbers = re.findall(r"\[unique_number\]\s*=>\s*(.+)", text)
    verdicts = re.findall(
        r"\[verdict_text\]\s*=>\s*(.*?)(?=\n\s*\[[^\]]+\]\s*=>|\n\s*\)\s*\n\s*\)|\Z)",
        text,
        flags=re.S,
    )

    out_lines = []
    for i, verdict in enumerate(verdicts):
        case_no = unique_numbers[i].strip() if i < len(unique_numbers) else f"CASE_{i+1}"
        out_lines.append(f"Գործի համարը: {case_no}")
        out_lines.append(verdict.strip())
        out_lines.append("")
        out_lines.append("-" * 80)
        out_lines.append("")

    Path(output_path).write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Extracted {len(verdicts)} cases -> {output_path}")
extract_verdict_texts("caseList1.txt", "caseList1_extracted.txt")
extract_verdict_texts("caseList2.txt", "caseList2_extracted.txt")
extract_verdict_texts("caseList3.txt", "caseList3_extracted.txt")
extract_verdict_texts("caseList4.txt", "caseList4_extracted.txt")
extract_verdict_texts("caseList51.txt", "caseList51_extracted.txt")
