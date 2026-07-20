"""Builds a fine-tuning dataset from src/data/court_papers_full.csv (2073 real
Armenian court decisions already in this repo) in a NEW target format:
conversational legal answers grounded in the real verdict text, instead of
the old armenia_lawyer_routing_dataset_150.xlsx's terse routing-triple format
("Recommended lawyer: X / Constitution articles: Y / Routing hint: Z").

Why this exists: the deployed armenia-lawyer-router model was fine-tuned only
on that 150-row routing-triple dataset, so it learned to always answer in
that shape — no amount of system-prompt engineering fully overrides it (see
Modelfile's PARAMETER stop "Routing hint:"/"->", which are patches around
this, not a fix). Retraining on examples in the actual desired answer shape
is the real fix; this script produces those examples from real case data
already in the repo, so nothing here is invented/hallucinated legal content
— only the instruction (a client question) is templated, and it's templated
from each case's own real Category label.

Usage:
    ./law/bin/python notebook/build_finetune_dataset.py
Writes src/data/finetune_dataset.csv (query, answer, source_case, source_judge)
— alongside its source, court_papers_full.csv, rather than under notebook/,
since it's a data file, not a notebook. Upload it as a Kaggle dataset input
for fine-tune-to-gguf.ipynb.
"""
import csv
import os
import re

csv.field_size_limit(10 * 1024 * 1024)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "data")
SRC_PATH = os.path.join(_DATA_DIR, "court_papers_full.csv")
OUT_PATH = os.path.join(_DATA_DIR, "finetune_dataset.csv")

MAX_EXCERPT_CHARS = 550  # keeps the combined instruction+response comfortably under
                          # the notebook's max_seq_length=1024 tokens even for
                          # Armenian, which tokenizes less densely than English


def clean_category(raw: str) -> str:
    """'11.1 Գումարի պահանջի մասին' -> 'Գումարի պահանջի մասին'"""
    return re.sub(r"^\s*\d+(\.\d+)*\s*", "", (raw or "").strip())


def clean_excerpt(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated + "..."


def build_example(row: dict) -> dict:
    category = clean_category(row.get("Category", ""))
    case_number = (row.get("Case_Number") or "").strip()
    judge = (row.get("Judge") or "").strip()
    excerpt = clean_excerpt(row.get("Verdict_Text", ""), MAX_EXCERPT_CHARS)

    query = (
        f"Հաճախորդի հարցում իրավական խորհրդատվության համար, կապված հետևյալ խնդրի հետ. "
        f"{category}: Ի՞նչ պետք է իմանամ և ինչպե՞ս գործեմ:"
    )

    answer = (
        f"Ձեր հարցը վերաբերում է հետևյալ թեմային. {category}:\n\n"
        f"Իրական դատական գործից քաղված համապատասխան նախադեպ (գործ {case_number}"
        f"{f', դատավոր՝ {judge}' if judge else ''}).\n{excerpt}\n\n"
        f"Սա ընդհանուր իրավական կողմնորոշում է՝ հիմնված իրական դատական որոշման վրա, "
        f"ոչ թե վերջնական իրավաբանական եզրակացություն. Ձեր կոնկրետ դեպքի մանրամասների "
        f"համար խորհուրդ ենք տալիս խորհրդակցել որակավորված փաստաբանի հետ:"
    )

    return {"query": query, "answer": answer, "source_case": case_number, "source_judge": judge}


def main():
    with open(SRC_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    examples = [build_example(r) for r in rows if (r.get("Verdict_Text") or "").strip()]

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "answer", "source_case", "source_judge"])
        writer.writeheader()
        writer.writerows(examples)

    print(f"Wrote {len(examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
