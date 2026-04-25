#!/usr/bin/env python3

import argparse
import json
import os

import torch
import wandb
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig

from utils.defaults import DEVICE
from utils.create_adverserial_dataset_test import ask_a_math_question

os.environ["WANDB_API_KEY"] = "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"
wandb.login(key=os.environ["WANDB_API_KEY"])

# ── Constants ──────────────────────────────────────────────────────────────────


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
        # question = entry.get("modified_question") or entry.get("original_question")
        # response_ref = entry.get("response_ref")
        question = entry.get("question")
        response_ref = str(entry.get("raw", "")).strip()

        print("question", question)
        print("response ref", response_ref)

        if not question or not response_ref:
            skipped += 1
            continue

        texts.append({"text": format_prompt(question, response_ref)})

    if skipped:
        print(f"  Skipped {skipped} entries (missing question or response_ref).")

    return Dataset.from_list(texts)


# ── Math accuracy callback ─────────────────────────────────────────────────────

class MathAccuracyCallback(TrainerCallback):
    def __init__(self, test_records: list[dict], model):
        self.test_records = test_records
        self.model = model

    def on_evaluate(self, args, state, control, **kwargs):
        print(f"\n[MathAccuracyCallback] Running math accuracy eval at step {state.global_step}...")

        correct = 0
        per_example_results = []

        for i, entry in tqdm(enumerate(self.test_records)):
            # question = entry.get("modified_question") or entry.get("original_question")
            # answer_ref = str(entry.get("answer_ref", "")).strip()

            question = entry.get("question")
            answer_ref = str(entry.get("answer", "")).strip()

            print("question", question)
            print("answer ref", answer_ref)

            try:
                response, predicted = ask_a_math_question(question, self.model)
            except Exception as e:
                print(f"  [Warning] Example {i} failed inference: {e}")
                response, predicted = "", None

            predicted_str = str(predicted).strip() if predicted is not None else ""
            is_correct = predicted_str == answer_ref

            if is_correct:
                correct += 1

            per_example_results.append({
                "step": state.global_step,
                "index": i,
                "question": question,
                "answer_ref": answer_ref,
                "predicted": predicted_str,
                "correct": is_correct,
                "response": response,
            })

        accuracy = correct / len(self.test_records) if self.test_records else 0.0
        print(f"[MathAccuracyCallback] Accuracy: {correct}/{len(self.test_records)} = {accuracy:.4f}")

        # Log scalar accuracy
        wandb.log({
            "eval/math_accuracy": accuracy,
            "eval/math_correct": correct,
            "eval/math_total": len(self.test_records),
        }, step=state.global_step)

        # Log per-example results as a W&B Table
        table = wandb.Table(
            columns=["step", "index", "question", "answer_ref", "predicted", "correct", "response"]
        )
        for r in per_example_results:
            table.add_data(
                r["step"], r["index"], r["question"],
                r["answer_ref"], r["predicted"], r["correct"], r["response"]
            )
        wandb.log({"eval/per_example_results": table}, step=state.global_step)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning on math problems using modified_question -> response_ref."
    )
    parser.add_argument("--train-file", required=True, help="Path to training .json file.")
    parser.add_argument("--test-file", required=True, help="Path to test/eval .json file.")
    parser.add_argument("--output-dir", default="./sft_output_clean_trained", help="Directory to save the model.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Directory to get the model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device training batch size.")
    parser.add_argument("--max-seq-length", type=int, default=128, help="Maximum token sequence length.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100, help="Run eval every N steps.")
    args = parser.parse_args()

    # ── Load model & tokenizer ─────────────────────────────────────────────────
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
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
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=torch.cuda.is_available(),
        dataset_text_field="text",
        report_to="wandb",
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        callbacks=[MathAccuracyCallback(test_records, model)],
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    wandb.init(project="sft-math")
    main()
    wandb.finish()


"""
python sft_loop.py --train-file usable_dataset/gsm8k_processed_train_adversarial_augmented_train.json --test-file usable_dataset/gsm8k_processed_train_adversarial_augmented_test.json --batch-size 2
"""