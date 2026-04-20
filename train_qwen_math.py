#!/usr/bin/env python3
"""
SFT fine-tuning script for Qwen Math model on denoised math samples.

Usage:
    python train_qwen_math.py \
        --train-file dataset/denoised_samples_train.jsonl \
        --test-file  dataset/denoised_samples_test.jsonl

The JSONL files must have at minimum:
    - "question": str
    - "raw": str  (full reasoning + "#### <answer>")
    - "answer": str
"""

import argparse
import json
import os
import random
import re
import sys

import torch
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        print("Device: CUDA")
        return "cuda"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        print("Device: MPS")
        return "mps"
    print("Device: CPU")
    return "cpu"

DEVICE = get_device()

# ── Prompt formatting ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise math solver. "
    "Identify and ignore any irrelevant information in the problem. "
    "Show your reasoning step by step, then give the final answer after ####."
)

def format_example(question: str, raw_response: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{raw_response}<|im_end|>"
    )

# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: File not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping malformed JSON on line {line_no}: {e}")
    if not records:
        sys.exit(f"ERROR: No valid records found in {path}")
    return records


def build_hf_dataset(records: list[dict], split_name: str) -> Dataset:
    texts, skipped = [], 0
    for entry in records:
        question = (entry.get("question") or "").strip()
        raw = (entry.get("raw") or "").strip()
        if not question or not raw:
            skipped += 1
            continue
        texts.append({"text": format_example(question, raw)})
    if skipped:
        print(f"  [{split_name}] Skipped {skipped} entries (missing question/raw).")
    if not texts:
        sys.exit(f"ERROR: [{split_name}] No usable examples after filtering.")
    print(f"  [{split_name}] {len(texts)} examples ready.")
    return Dataset.from_list(texts)

# ── Answer extraction ──────────────────────────────────────────────────────────

_ANSWER_RE = re.compile(r"####\s*([\d,.\-]+)")

def extract_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None

# ── Math accuracy callback ─────────────────────────────────────────────────────

class MathAccuracyCallback(TrainerCallback):
    def __init__(self, test_records: list[dict], tokenizer, model, max_new_tokens: int = 512):
        self.test_records = test_records
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def _generate_answer(self, question: str) -> tuple[str, str | None]:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return generated, extract_answer(generated)

    def on_evaluate(self, args, state, control, **kwargs):
        print(f"\n[MathAccuracy] Evaluating at step {state.global_step} "
              f"on {len(self.test_records)} examples...")

        self.model.eval()
        correct = 0
        rows = []

        for i, entry in enumerate(self.test_records):
            question = (entry.get("question") or "").strip()
            answer_ref = str(entry.get("answer") or "").strip()
            if not question:
                continue

            try:
                response, predicted = self._generate_answer(question)
            except Exception as e:
                print(f"  WARNING: inference failed for example {i}: {e}")
                response, predicted = "", None

            predicted_str = (predicted or "").strip()
            is_correct = predicted_str == answer_ref
            if is_correct:
                correct += 1

            rows.append({
                "step": state.global_step,
                "index": i,
                "question": question,
                "answer_ref": answer_ref,
                "predicted": predicted_str,
                "correct": is_correct,
                "response": response,
            })

        total = len(rows)
        accuracy = correct / total if total else 0.0
        print(f"[MathAccuracy] {correct}/{total} correct — accuracy={accuracy:.4f}")

        wandb.log({
            "eval/math_accuracy": accuracy,
            "eval/math_correct": correct,
            "eval/math_total": total,
            "global_step": state.global_step,
        })

        table = wandb.Table(columns=["step", "index", "question", "answer_ref", "predicted", "correct", "response"])
        for r in rows:
            table.add_data(r["step"], r["index"], r["question"],
                           r["answer_ref"], r["predicted"], r["correct"], r["response"])
        wandb.log({"eval/per_example_results": table})

# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT fine-tuning of Qwen Math on denoised samples.")
    parser.add_argument("--train-file", required=True, help="Path to training .jsonl file.")
    parser.add_argument("--test-file",  required=True, help="Path to test/eval .jsonl file.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--output-dir", default="./qwen_math_sft_output",
                        help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch-size",    type=int,   default=4)
    parser.add_argument("--grad-accum",    type=int,   default=2,
                        help="Gradient accumulation steps.")
    parser.add_argument("--max-seq-length",type=int,   default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio",  type=float, default=0.1,
                        help="Fraction of total steps used for LR warmup (default 10%%).")
    parser.add_argument("--weight-decay",  type=float, default=0.01)
    parser.add_argument("--logging-steps",  type=int,   default=1)
    parser.add_argument("--eval-interval",  type=float, default=0.1,
                        help="Eval frequency as a fraction of total training steps (default 0.1 = every 10%%).")
    parser.add_argument("--wandb-project", default="qwen-math-sft")
    parser.add_argument("--wandb-run-name",default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── WandB ──────────────────────────────────────────────────────────────────
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )

    # ── Reproducibility ────────────────────────────────────────────────────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_name}")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE != "cuda":
        model = model.to(DEVICE)

    model.config.use_cache = False
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    print("\nLoading train data...")
    train_records = load_jsonl(args.train_file)
    train_dataset = build_hf_dataset(train_records, "train")

    print("Loading test data...")
    test_records = load_jsonl(args.test_file)
    test_dataset = build_hf_dataset(test_records, "test")

    # ── Compute eval_steps / save_steps from total training steps ─────────────
    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    eval_steps = max(1, round(total_steps * args.eval_interval))
    print(f"  steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
          f"eval_steps={eval_steps} (every {args.eval_interval*100:.0f}%)")

    # ── SFTConfig ──────────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_seq_length,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=args.logging_steps,
        bf16=use_bf16,
        fp16=not use_bf16 and DEVICE == "cuda",
        dataset_text_field="text",
        report_to="wandb",
        seed=args.seed,
        dataloader_num_workers=0,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    math_cb = MathAccuracyCallback(test_records, tokenizer, model)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        callbacks=[math_cb],
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────────
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()


"""
Example usage:
    python train_qwen_math.py \
        --train-file dataset/denoised_samples_train.jsonl \
        --test-file  dataset/denoised_samples_test.jsonl \
        --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
        --epochs 3 \
        --batch-size 4 \
        --grad-accum 2 \
        --max-seq-length 1024 \
        --learning-rate 2e-5 \
        --wandb-project qwen-math-sft

    # To split denoised_samples.jsonl into train/test first, run:
    python split_jsonl_train_test.py
"""
