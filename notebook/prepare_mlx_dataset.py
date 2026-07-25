"""Converts src/data/finetune_dataset.csv (16,201 rows: query, answer,
source_case, source_judge) into the {"prompt": ..., "completion": ...} JSONL
format mlx-lm's LoRA trainer uses for prompt-masked training
(mlx_lm/tuner/datasets.py: CompletionsDataset).

Split into prompt/completion (rather than one flat "text" field) so training
can run with --mask-prompt: loss is computed only on the answer tokens, not
on the instruction/query preamble, which is near-identical across all 1,867
rows and wastes gradient signal if left unmasked (see run_mlx_lora_cpu.py's
CompletionsDataset monkeypatch, which keeps this exact raw-text template
instead of mlx-lm's default of re-wrapping prompt/completion through the
base model's chat template).

Uses the same Instruction/Query/Response template as fine-tune-to-gguf.ipynb
(format_example) so the locally-trained model stays consistent with what the
deployed Modelfile's system prompt already expects.

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

PROMPT_TEMPLATE = """### Instruction:
Դուք հայ իրավաբան եք (legal consultant)։ Պատասխանեք հաճախորդի հարցին հստակ, մարդամոտ և իրավական հիմքով։

Query: {query}

### Response:
"""


def main():
    df = pd.read_csv(SRC_PATH)
    rows = [
        {
            "prompt": PROMPT_TEMPLATE.format(query=r["query"]),
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
