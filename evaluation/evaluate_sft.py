"""
Evaluation Script — Noise Robust SFT
======================================
Compares three models on a test JSONL file:
  1. Base / frozen SFT model (before training)
  2. Noise-robust fine-tuned model (after training)

For each (Q, Q', answer) triplet in the test set, runs inference on:
  - Clean Q  with base model
  - Noisy Q' with base model
  - Clean Q  with trained model
  - Noisy Q' with trained model

Metrics reported:
  - Accuracy on clean Q  (base vs trained)
  - Accuracy on noisy Q' (base vs trained)
  - Noise robustness gap : acc(clean) - acc(noisy)  — lower is better
  - Recovery rate        : how many noisy failures by base model are fixed by trained model

Usage:
    python eval_sft_trained.py \
        --test-jsonl  dataset/gsm8k_adv_test.jsonl \
        --base-model  jahyungu/Qwen2.5-1.5B-Instruct_gsm8k \
        --trained-model checkpoints_noise_robust/checkpoint_epoch3_step4449 \
        --output-dir  eval_results \
        --n-samples   500
"""

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer



def build_prompt(question: str) -> str:
    return (
        f"Solve this math problem step by step:\n\n"
        f"{question}\n\n"
        f"Provide your final answer in the format:\n"
        f"[reasoning steps]\n####\n[final answer (just the number)]"
    )


def build_chat_prompt(question: str, tokenizer) -> str:
    """Wrap build_prompt in the model's chat format so instruct models respond correctly."""
    messages = [{"role": "user", "content": build_prompt(question)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



def safe_float(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def extract_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from model response."""
    if "####" in text:
        parts = text.split("####")
        answer_part = parts[-1].strip()
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", answer_part)
        if numbers:
            return numbers[0].replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def answers_match(pred: Optional[str], ref) -> bool:
    pred_f = safe_float(pred)
    ref_f  = safe_float(ref)
    if pred_f is None or ref_f is None:
        return False
    return math.isclose(pred_f, ref_f, rel_tol=1e-4)



@torch.no_grad()
def run_inference(model, tokenizer, question: str, device: str,
                  max_new_tokens: int = 256):
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    answer = extract_answer(response)
    return response, answer



def load_test_triplets(jsonl_path: str, n_samples: Optional[int] = None,
                       max_variants: int = 1, seed: int = 42):
    """
    Load (clean_q, noisy_q, ref_answer) triplets from test JSONL.
    max_variants: how many noisy variants to use per question (default: 1 to keep eval fast)
    """
    random.seed(seed)
    triplets = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            clean_q    = record.get("original_question")
            ref_answer = record.get("original_answer")
            adverserials = record.get("modified_questions", {}).get("adverserials", [])

            if not clean_q or ref_answer is None or not adverserials:
                continue

            for noisy_q in adverserials[:max_variants]:
                if noisy_q:
                    triplets.append((clean_q, noisy_q, ref_answer))

    if n_samples and len(triplets) > n_samples:
        triplets = random.sample(triplets, n_samples)

    print(f"Loaded {len(triplets)} test triplets from {jsonl_path}")
    return triplets




def evaluate_model(model, tokenizer, triplets, device, label: str,
                   max_new_tokens: int = 256):
    """
    Run inference on all triplets for both clean Q and noisy Q'.
    Returns dict with per-sample results and aggregate metrics.
    """
    results = []

    for clean_q, noisy_q, ref_answer in tqdm(triplets, desc=f"Evaluating {label}"):
        # Clean Q
        clean_resp, clean_pred = run_inference(
            model, tokenizer, clean_q, device, max_new_tokens)
        clean_correct = answers_match(clean_pred, ref_answer)

        # Noisy Q'
        noisy_resp, noisy_pred = run_inference(
            model, tokenizer, noisy_q, device, max_new_tokens)
        noisy_correct = answers_match(noisy_pred, ref_answer)

        results.append({
            "clean_q":       clean_q,
            "noisy_q":       noisy_q,
            "ref_answer":    str(ref_answer),
            "clean_pred":    clean_pred,
            "noisy_pred":    noisy_pred,
            "clean_correct": clean_correct,
            "noisy_correct": noisy_correct,
            "clean_response": clean_resp,
            "noisy_response": noisy_resp,
        })

    n = len(results)
    clean_acc = sum(r["clean_correct"] for r in results) / n
    noisy_acc = sum(r["noisy_correct"] for r in results) / n
    gap       = clean_acc - noisy_acc   # lower = more robust

    # Recovery: cases where base fails on noisy but this model succeeds
    # (computed in compare step — placeholder here)
    metrics = {
        "model":        label,
        "n_samples":    n,
        "clean_acc":    round(clean_acc, 4),
        "noisy_acc":    round(noisy_acc, 4),
        "noise_gap":    round(gap, 4),
    }

    print(f"\n{'─'*50}")
    print(f"Model : {label}")
    print(f"Clean Q  accuracy : {clean_acc:.2%}")
    print(f"Noisy Q' accuracy : {noisy_acc:.2%}")
    print(f"Noise gap (↓ better): {gap:.2%}")
    print(f"{'─'*50}")

    return results, metrics




def compare_results(base_results, trained_results):
    """
    Compute recovery rate and regression rate between base and trained model.
    Recovery : noisy wrong in base → noisy correct in trained  (improvement)
    Regression: noisy correct in base → noisy wrong in trained (degradation)
    """
    assert len(base_results) == len(trained_results)

    recovery   = 0
    regression = 0
    both_wrong = 0
    both_right = 0

    for b, t in zip(base_results, trained_results):
        base_noisy_ok    = b["noisy_correct"]
        trained_noisy_ok = t["noisy_correct"]

        if not base_noisy_ok and trained_noisy_ok:
            recovery += 1
        elif base_noisy_ok and not trained_noisy_ok:
            regression += 1
        elif base_noisy_ok and trained_noisy_ok:
            both_right += 1
        else:
            both_wrong += 1

    n = len(base_results)
    comparison = {
        "n_samples":         n,
        "recovery_rate":     round(recovery / n, 4),    # base fails, trained fixes
        "regression_rate":   round(regression / n, 4),  # base ok, trained breaks
        "both_correct_rate": round(both_right / n, 4),
        "both_wrong_rate":   round(both_wrong / n, 4),
        "net_improvement":   round((recovery - regression) / n, 4),
    }

    print(f"\n{'='*50}")
    print(f"Comparison: Base vs Trained (on noisy Q')")
    print(f"  Recovery   (base✗ → trained✓) : {recovery}/{n} = {recovery/n:.2%}")
    print(f"  Regression (base✓ → trained✗) : {regression}/{n} = {regression/n:.2%}")
    print(f"  Both correct                  : {both_right}/{n} = {both_right/n:.2%}")
    print(f"  Both wrong                    : {both_wrong}/{n} = {both_wrong/n:.2%}")
    print(f"  Net improvement               : {(recovery-regression)/n:+.2%}")
    print(f"{'='*50}")

    return comparison



def plot_results(base_metrics, trained_metrics, comparison, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))


    ax = axes[0]
    x       = np.arange(2)
    width   = 0.3
    labels  = ["Clean Q", "Noisy Q'"]
    base_vals    = [base_metrics["clean_acc"],    base_metrics["noisy_acc"]]
    trained_vals = [trained_metrics["clean_acc"], trained_metrics["noisy_acc"]]

    bars1 = ax.bar(x - width/2, base_vals,    width, label="Base model",    color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, trained_vals, width, label="Trained model", color="seagreen",  alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy: Clean Q vs Noisy Q'\nBase vs Noise-Robust Trained Model")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)

    # Plot 2: Recovery / Regression breakdown 
    ax = axes[1]
    cats   = ["Recovery\n(base (fail) -> trained (worked post))", "Regression\n(base (worked)  -> trained (failed))",
              "Both\ncorrect", "Both\nwrong"]
    values = [
        comparison["recovery_rate"],
        comparison["regression_rate"],
        comparison["both_correct_rate"],
        comparison["both_wrong_rate"],
    ]
    colors = ["seagreen", "tomato", "steelblue", "lightgrey"]
    bars = ax.bar(cats, values, color=colors, alpha=0.85)
    ax.set_ylabel("Fraction of test samples")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title(f"Noisy Q' Outcome Breakdown\nNet improvement: {comparison['net_improvement']:+.2%}")
    ax.grid(True, axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = Path(output_dir) / "eval_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")
    plt.close()



def save_results(base_results, trained_results, base_metrics,
                 trained_metrics, comparison, output_dir):
    out = {
        "base_model_metrics":    base_metrics,
        "trained_model_metrics": trained_metrics,
        "comparison":            comparison,
    }
    summary_path = Path(output_dir) / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # Full per-sample results
    detail_path = Path(output_dir) / "eval_detail.jsonl"
    with open(detail_path, "w") as f:
        for b, t in zip(base_results, trained_results):
            f.write(json.dumps({
                "clean_q":              b["clean_q"],
                "noisy_q":              b["noisy_q"],
                "ref_answer":           b["ref_answer"],
                "base_clean_pred":      b["clean_pred"],
                "base_noisy_pred":      b["noisy_pred"],
                "base_clean_correct":   b["clean_correct"],
                "base_noisy_correct":   b["noisy_correct"],
                "trained_clean_pred":   t["clean_pred"],
                "trained_noisy_pred":   t["noisy_pred"],
                "trained_clean_correct": t["clean_correct"],
                "trained_noisy_correct": t["noisy_correct"],
            }) + "\n")
    print(f"Detailed results saved: {detail_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl",     type=str, required=True,
                        help="Path to test adversarial JSONL file")
    parser.add_argument("--base-model",     type=str,
                        default="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k",
                        help="Base / frozen SFT model (before noise-robust training)")
    parser.add_argument("--trained-model",  type=str, required=True,
                        help="Path to noise-robust fine-tuned checkpoint directory")
    parser.add_argument("--output-dir",     type=str, default="eval_results")
    parser.add_argument("--n-samples",      type=int, default=500,
                        help="Max test triplets to evaluate (default: 500)")
    parser.add_argument("--max-variants",   type=int, default=1,
                        help="Noisy variants per question to test (default: 1)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    triplets = load_test_triplets(
        args.test_jsonl, n_samples=args.n_samples,
        max_variants=args.max_variants, seed=args.seed
    )
    if not triplets:
        raise ValueError("No test triplets loaded — check JSONL format.")

    print(f"Loading tokenizer from base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    print(f"\nLoading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    base_model.eval()

    base_results, base_metrics = evaluate_model(
        base_model, tokenizer, triplets, device,
        label="base", max_new_tokens=args.max_new_tokens
    )

    # Free base model memory before loading trained model
    del base_model
    torch.cuda.empty_cache()


    print(f"\nLoading trained model: {args.trained_model}")
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.trained_model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    trained_model.eval()

    trained_results, trained_metrics = evaluate_model(
        trained_model, tokenizer, triplets, device,
        label="trained", max_new_tokens=args.max_new_tokens
    )

    del trained_model
    torch.cuda.empty_cache()


    comparison = compare_results(base_results, trained_results)


    save_results(base_results, trained_results, base_metrics,
                 trained_metrics, comparison, args.output_dir)
    plot_results(base_metrics, trained_metrics, comparison, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()