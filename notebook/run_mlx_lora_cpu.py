"""Runs mlx-lm's LoRA fine-tuning fully on CPU -- no Metal/GPU.

mlx-lm's CLI (`python -m mlx_lm lora ...`) has no --device flag, and MLX
defaults to the GPU (Metal) device on Apple Silicon. This wrapper forces
mx.set_default_device(mx.cpu) before mlx_lm loads any model or data, then
delegates to the normal CLI argument parsing, so it accepts the exact same
flags as `mlx_lm.lora`.

Usage (same flags as `python -m mlx_lm lora`, plus --mask-prompt to use the
prompt-masked training this wrapper enables):
    ./finetune_env/bin/python notebook/run_mlx_lora_cpu.py \\
        --model mlx-community/gemma-2-2b-it-4bit \\
        --train --data data/mlx_finetune --mask-prompt \\
        --adapter-path adapters/armenia_lawyer_router_v2 \\
        --iters 500 --batch-size 1 --max-seq-length 1024

Also patches CompletionsDataset (mlx_lm/tuner/datasets.py) to tokenize our
data/mlx_finetune/{split}.jsonl's raw "prompt"/"completion" strings directly
instead of mlx-lm's default of re-wrapping them through the base model's
chat template. Our prompt/completion text IS already a complete template
(see notebook/prepare_mlx_dataset.py's PROMPT_TEMPLATE) matching what
fine-tune-to-gguf.ipynb trained on and what the deployed Modelfile expects
-- wrapping it a second time in Gemma's <start_of_turn> chat template would
train on a format the production Modelfile never uses.
"""
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx_lm.tuner.datasets import CompletionsDataset  # noqa: E402


def _process_raw_completion(self, d):
    prompt = d[self.prompt_key]
    completion = d[self.completion_key]
    prompt_tokens = self.tokenizer.encode(prompt)
    full_tokens = self.tokenizer.encode(prompt + completion)
    if full_tokens[-1] != self.tokenizer.eos_token_id:
        full_tokens.append(self.tokenizer.eos_token_id)
    offset = len(prompt_tokens) if self.mask_prompt else 0
    return (full_tokens, offset)


CompletionsDataset.process = _process_raw_completion

from mlx_lm.lora import main  # noqa: E402 (import after forcing CPU device + patch)

if __name__ == "__main__":
    main()
