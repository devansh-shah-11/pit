#!/usr/bin/env python3

import argparse
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from utils.defaults import DEVICE

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME = "kmseong/Llama3.2-3B-gsm8k-fullft-atfter-ssft"

# ── Prompt formatting ──────────────────────────────────────────────────────────

def format_prompt(question: str, response_ref: str) -> str:
    prompt = (
        f"Solve this math problem step by step:\n"
        f"{question}\n"
        f"Provide your final answer in the format:\n"
        f"[reasoning steps]\n"
        f"####\n"
        f"[final answer (just the number)]"
    )
    return f"{prompt}\n{response_ref}"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def build_dataset(records: list[dict]) -> Dataset:
    texts = []
    skipped = 0

    for entry in records:
        question = entry.get("modified_question") or entry.get("original_question")
        response_ref = entry.get("response_ref")

        if not question or not response_ref:
            skipped += 1
            continue

        texts.append({"text": format_prompt(question, response_ref)})

    if skipped:
        print(f"  Skipped {skipped} entries (missing question or response_ref).")

    return Dataset.from_list(texts)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning on math problems using modified_question -> response_ref."
    )
    parser.add_argument("--train-file", required=True, help="Path to training .json file.")
    parser.add_argument("--test-file", required=True, help="Path to test/eval .json file.")
    parser.add_argument("--output-dir", default="./sft_output", help="Directory to save the model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--max-seq-length", type=int, default=128, help="Maximum token sequence length.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    args = parser.parse_args()

    # ── Load model & tokenizer ─────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(DEVICE)

    # ── Build datasets ─────────────────────────────────────────────────────────
    print("Loading train data...")
    train_records = load_records(args.train_file)
    train_dataset = build_dataset(train_records)
    print(f"  Train examples: {len(train_dataset)}")

    print("Loading test data...")
    test_records = load_records(args.test_file)
    test_dataset = build_dataset(test_records)
    print(f"  Test examples:  {len(test_dataset)}")

    # ── SFT training config ────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=torch.cuda.is_available(),
        dataset_text_field="text",
        report_to="none",
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()