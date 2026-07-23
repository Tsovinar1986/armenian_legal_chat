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

The excerpt is the legal-basis + ruling section of the verdict (see
extract_legal_excerpt), not the first N characters of the document. An
earlier version took the first 550 characters, which for these documents is
always procedural boilerplate — party names, filing dates, the
"Պ Ա Ր Զ Ե Ց" section header — never the actual law. LLaMA-family models
train better on short, substantive answers than on walls of that boilerplate,
so this targets the part a legal-advice chatbot actually needs: the articles
the court relied on and what it decided.

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

REASONING_CHARS = 180  # legal-basis text right before the ruling (article citations)
OPERATIVE_CHARS = 260  # the ruling itself (numbered outcome)
# ~440 chars of substance per example, down from the old 550-char excerpt —
# and unlike the old one it's the part that's actually useful to a
# legal-advice chatbot, not procedural preamble.

# Armenian civil verdicts follow a fixed shape: procedural history ->
# reasoning (cites the articles relied on) -> operative ruling, introduced by
# a "ՎՃՌԵՑ" ("RULED") or "ՈՐՈՇԵՑ" ("DECIDED") header — present in 91% of this
# corpus (1896/2073 rows). Matched with internal whitespace tolerance since
# these headers are often OCR'd with a space between every letter ("Վ Ճ Ռ Ե
# Ց"); restricted to uppercase Armenian so it doesn't also match ordinary
# lowercase mentions of these words in the narrative text above.
OPERATIVE_MARKER_RE = re.compile(r"(?:Վ\s*Ճ\s*Ռ\s*Ե\s*Ց|Ո\s*Ր\s*Ո\s*Շ\s*Ե\s*Ց)")
SPACED_CAPS_RE = re.compile(r"(?:[Ա-Ֆ]\s){2,}[Ա-Ֆ]")  # "Պ Ա Ր Զ Ե Ց" -> "ՊԱՐԶԵՑ"
FOOTER_RE = re.compile(r"Սույն\s+փաստաթղթի\s+իսկությունը.*", re.DOTALL)  # e-gov.am verification footer
SIGNATURE_RE = re.compile(r"ԴԱՏԱՎՈՐ\b.*", re.DOTALL)  # judge signature line


def clean_category(raw: str) -> str:
    """'11.1 Գումարի պահանջի մասին' -> 'Գումարի պահանջի մասին'"""
    return re.sub(r"^\s*\d+(\.\d+)*\s*", "", (raw or "").strip())


def _normalize(text: str) -> str:
    text = FOOTER_RE.sub("", text)
    text = SIGNATURE_RE.sub("", text)
    text = SPACED_CAPS_RE.sub(lambda m: m.group(0).replace(" ", ""), text)
    return re.sub(r"\s+", " ", text).strip()


def _trim_start(text: str, max_chars: int) -> str:
    """Cuts a slice taken from the *middle* of a document down to max_chars,
    dropping the partial word at the front so it reads as a clean fragment
    instead of starting mid-word."""
    if len(text) <= max_chars:
        return text
    trimmed = text[-max_chars:]
    if " " in trimmed:
        trimmed = trimmed.split(" ", 1)[1]
    return trimmed


def _trim_end(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed + "..."


def extract_legal_excerpt(verdict_text: str) -> str:
    """Pulls the legal-basis + ruling section instead of the procedural
    preamble at the top of the document — see OPERATIVE_MARKER_RE."""
    text = verdict_text or ""
    matches = list(OPERATIVE_MARKER_RE.finditer(text))
    if not matches:
        # No recognizable operative marker (~9% of this corpus, usually
        # atypical/truncated filings) -- the ruling is still always near the
        # end, so take the tail instead of the old first-N-chars approach.
        tail_chars = REASONING_CHARS + OPERATIVE_CHARS
        return _normalize(text)[-tail_chars:].strip()

    marker = matches[-1]
    reasoning = _trim_start(_normalize(text[: marker.start()]), REASONING_CHARS)
    operative = _trim_end(_normalize(text[marker.end() :]), OPERATIVE_CHARS)
    return f"{reasoning}\n\n{operative}".strip()


def build_example(row: dict) -> dict:
    category = clean_category(row.get("Category", ""))
    case_number = (row.get("Case_Number") or "").strip()
    judge = (row.get("Judge") or "").strip()
    excerpt = extract_legal_excerpt(row.get("Verdict_Text", ""))

    query = (
        f"Հաճախորդի հարցում իրավական խորհրդատվության համար, կապված հետևյալ խնդրի հետ. "
        f"{category}: Ի՞նչ պետք է իմանամ և ինչպե՞ս գործեմ:"
    )

    answer = (
        f"Ձեր հարցը վերաբերում է հետևյալ թեմային. {category}:\n\n"
        f"Իրական դատական գործից քաղված իրավական հիմքն ու վճիռը (գործ {case_number}"
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
