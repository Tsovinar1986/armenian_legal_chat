"""Converts src/data/finetune_dataset.csv (2073 rows: query, answer,
source_case, source_judge, category, excerpt) into the
{"prompt": ..., "completion": ...} JSONL format mlx-lm's LoRA trainer uses
for prompt-masked training (mlx_lm/tuner/datasets.py: CompletionsDataset).

Split into prompt/completion (rather than one flat "text" field) so training
can run with --mask-prompt: loss is computed only on the answer tokens, not
on the instruction/context preamble (see run_mlx_lora_cpu.py's
CompletionsDataset monkeypatch, which keeps this exact raw-text template
instead of mlx-lm's default of re-wrapping prompt/completion through the
base model's chat template).

The prompt built here deliberately MIRRORS the exact prompt
legal_agent.py's LegalAgent._direct_llm_answer() sends the deployed model in
production (same fixed English meta-instructions, same
Conversation/Client's question/Retrieved precedent context/Real court case
examples section layout) — not the old bare "### Instruction / Query /
### Response" shape this file used to build.

Why that change matters: the old prompt gave the model nothing but a
category label and expected the completion to include one specific real
case's excerpt, case number, and judge — content with zero signal in the
input pointing to which of 2073 cases to produce. That's not a task a model
can learn to generalize; at best it half-memorizes the fixed wrapper
sentences (which is exactly what the deployed armenia-lawyer-router does —
opens with a coherent boilerplate sentence, then degrades into noise where
it has to invent the rest). It also meant training and production used two
totally different prompt shapes, so even what little it did learn didn't
transfer to how it's actually invoked.

This version instead gives the model the SAME excerpt/case info as input
context (framed exactly like retrieved RAG context, because that's what it
actually is in production) and asks it to write a fluent answer grounded in
that given context — a genuinely learnable, generalizable skill, and the
exact shape it will see at inference time.

Usage:
    ./finetune_env/bin/python notebook/prepare_mlx_dataset.py
Writes data/mlx_finetune/{train,valid,test}.jsonl (90/5/5 split, seed=42).
"""
import json
import os
import random

import pandas as pd

_ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC_PATH = os.path.join(_ROOT, "src", "data", "finetune_dataset.csv")
OUT_DIR = os.path.join(_ROOT, "data", "mlx_finetune")

NO_PRIOR_TURNS = "Սա այս զրույցի առաջին հարցն է։"  # legal_agent.py's own no-history string


def _truncate(text, max_chars):
    """Mirrors LegalAgent._truncate_text exactly, so training sees the same
    truncation behavior production applies to real retrieved context."""
    text = (text or "").strip()
    if not text:
        return "N/A"
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return f"{truncated.strip()}..."


def _cases_context(category, case_number, judge, excerpt):
    """Mirrors the "ՀԱՄՀ case examples FROM REAL COURT DECISIONS" block
    _generate_rag_response builds from _find_relevant_cases results."""
    verdict_summary = excerpt[:300] + "..." if len(excerpt) > 300 else excerpt
    return (
        "\n\nՀԱՄՀ case examples FROM REAL COURT DECISIONS:\n"
        f"\n📌 Example 1: Case {case_number}\n"
        f"   Category: {category}\n"
        f"   Judge: {judge}\n"
        f"   Verdict Summary: {verdict_summary}\n"
    )


def build_prompt(query, category, case_number, judge, excerpt):
    """Byte-for-byte the same template as LegalAgent._direct_llm_answer's
    `prompt` (language_name="Armenian", first-turn conversation_context) —
    see this file's module docstring for why matching it matters."""
    cases_context = _cases_context(category, case_number, judge, excerpt)
    return (
        "You are a senior Armenian legal consultant. Answer the client's question in "
        "Armenian, grounded only in the context below — do not invent case numbers, "
        "facts, or citations that aren't present here. If the context doesn't actually cover "
        "the question, say so plainly and give general legal guidance instead.\n\n"
        f"Conversation so far:\n{NO_PRIOR_TURNS}\n\n"
        f"Client's question: {query}\n\n"
        f"Retrieved precedent context:\n{_truncate(excerpt, 1500)}\n\n"
        f"Real court case examples:\n{_truncate(cases_context, 1000)}\n\n"
        "Write a clear, structured answer in Armenian."
    )


def main():
    df = pd.read_csv(SRC_PATH)
    rows = [
        {
            "prompt": build_prompt(r["query"], r["category"], r["source_case"], r["source_judge"], r["excerpt"]),
            "completion": f"{r['answer']}\n",
        }
        for _, r in df.iterrows()
    ]

    random.Random(42).shuffle(rows)
    n = len(rows)
    n_valid = max(1, int(n * 0.05))
    n_test = max(1, int(n * 0.05))
    valid = rows[:n_valid]
    test = rows[n_valid:n_valid + n_test]
    train = rows[n_valid + n_test:]

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, split in (("train", train), ("valid", valid), ("test", test)):
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in split:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} rows -> {path}")


if __name__ == "__main__":
    main()
