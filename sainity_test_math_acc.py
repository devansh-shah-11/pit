#!/usr/bin/env python3

import argparse
import json
import os

import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from utils.create_adverserial_dataset_test import ask_a_math_question
from utils.defaults import DEVICE, MODEL_NAME

os.environ["WANDB_API_KEY"] = "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"
wandb.login(key=os.environ["WANDB_API_KEY"])



# ── Data loading ───────────────────────────────────────────────────────────────

def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


# ── Evaluation ─────────────────────────────────────────────────────────────────

def run_eval(test_records: list[dict], model) -> None:
    correct = 0
    per_example_results = []

    total = 0

    for i, entry in tqdm(enumerate(test_records), total=len(test_records), desc="Evaluating"):
        # question = entry.get("modified_question") or entry.get("original_question")
        # answer_ref = str(entry.get("answer_ref", "")).strip()

        question = entry.get("question")
        answer_ref = str(entry.get("answer", "")).strip()

        print("question", question)
        print("answer ref", answer_ref)

        try:
            response, predicted = ask_a_math_question(question, model)
        except Exception as e:
            print(f"  [Warning] Example {i} failed inference: {e}")
            response, predicted = "", None

        predicted_str = str(predicted).strip() if predicted is not None else ""
        is_correct = predicted_str == answer_ref

        if is_correct:
            correct += 1

        total += 1

        per_example_results.append({
            "index": i,
            "question": question,
            "answer_ref": answer_ref,
            "predicted": predicted_str,
            "correct": is_correct,
            "response": response,
        })

        accuracy_partial = correct / total if total else 0.0
        print(f"\n>>>>>>>>>> Accuracy: {correct}/{total} = {accuracy_partial:.4f}")

    accuracy = correct / len(test_records) if test_records else 0.0
    print(f"\nAccuracy: {correct}/{len(test_records)} = {accuracy:.4f}")

    # Log scalar accuracy
    wandb.log({
        "eval/math_accuracy": accuracy,
        "eval/math_correct": correct,
        "eval/math_total": len(test_records),
    })

    # Log per-example results as a W&B Table
    table = wandb.Table(
        columns=["index", "question", "answer_ref", "predicted", "correct", "response"]
    )
    for r in per_example_results:
        table.add_data(
            r["index"], r["question"],
            r["answer_ref"], r["predicted"], r["correct"], r["response"]
        )
    wandb.log({"eval/per_example_results": table})


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate math accuracy on a test set without any training."
    )
    parser.add_argument("--test-file", required=True, help="Path to test .json file.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="HuggingFace model name or local path.")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (optional).")
    args = parser.parse_args()

    print("Loading Model", args.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(DEVICE)

    print("Loading test data...")
    test_records = load_records(args.test_file)
    print(f"  Test examples: {len(test_records)}")

    print(f"Running eval...")
    run_eval(test_records, model=model)  # None = core.py uses its default model


if __name__ == "__main__":
    wandb.init(project="sft-math-eval")
    main()
    wandb.finish()


"""
python eval_accuracy.py --test-file usable_dataset/gsm8k_processed_train_adversarial_augmented_test.json
"""