#!/usr/bin/env python3
"""
Evaluate a fine-tuned Qwen Math model.

Modes:
  1. Interactive  — type a question, see the model's response
  2. Batch        — evaluate on a .jsonl file, report accuracy

Usage:
  # Interactive
  python eval_model.py --model-path ./qwen_math_sft_output

  # Batch eval on a file
  python eval_model.py --model-path ./qwen_math_sft_output \
                       --eval-file  dataset/denoised_samples_test.jsonl
"""

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "backends") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise math solver. "
    "Identify and ignore any irrelevant information in the problem. "
    "Show your reasoning step by step, then give the final answer after ####."
)

def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# ── Answer extraction ──────────────────────────────────────────────────────────

_ANSWER_RE = re.compile(r"####\s*([\d,.\-]+)")

def extract_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    return m.group(1).replace(",", "").strip() if m else None

# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_from_prompt(prompt: str, model, tokenizer, max_new_tokens: int = 512) -> tuple[str, str | None]:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return response, extract_answer(response)


def generate(question: str, model, tokenizer, max_new_tokens: int = 512) -> tuple[str, str | None]:
    return generate_from_prompt(build_prompt(question), model, tokenizer, max_new_tokens)

# ── Batch eval ─────────────────────────────────────────────────────────────────

def batch_eval(eval_file: str, model, tokenizer, max_new_tokens: int, verbose: bool):
    if not os.path.exists(eval_file):
        sys.exit(f"ERROR: File not found: {eval_file}")

    records = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        sys.exit("ERROR: No records found in eval file.")

    correct = 0
    results = []

    for i, entry in enumerate(records):
        question   = (entry.get("question") or "").strip()
        answer_ref = str(entry.get("answer") or "").strip()
        entry_type = entry.get("type", "unknown")

        if not question:
            print(f"  [{i}] SKIP — missing question")
            continue

        response, predicted = generate(question, model, tokenizer, max_new_tokens)
        predicted_str = (predicted or "").strip()
        is_correct = predicted_str == answer_ref
        if is_correct:
            correct += 1

        results.append({
            "index": i,
            "type": entry_type,
            "question": question,
            "answer_ref": answer_ref,
            "predicted": predicted_str,
            "correct": is_correct,
        })

        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"\n[{i}] {status}  type={entry_type}")
            print(f"  Q: {question[:120]}{'...' if len(question) > 120 else ''}")
            print(f"  Expected: {answer_ref}  |  Got: {predicted_str}")
            if not is_correct:
                print(f"  Response: {response[:300]}{'...' if len(response) > 300 else ''}")

    total = len(results)
    accuracy = correct / total if total else 0.0

    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")

    # Break down by type (clean vs adversarial)
    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r["correct"])
    for t, scores in sorted(by_type.items()):
        print(f"  {t:20s}: {sum(scores)}/{len(scores)} = {sum(scores)/len(scores):.2%}")

    print(f"{'='*50}")

# ── Interactive modes ──────────────────────────────────────────────────────────

def interactive_single(model, tokenizer, max_new_tokens: int):
    """One question at a time — no memory of prior turns."""
    print("\nSingle-turn mode — each question is independent.")
    print("Commands: 'quit' to exit\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        response, predicted = generate(question, model, tokenizer, max_new_tokens)
        print(f"\nResponse:\n{response}")
        print(f"\nExtracted answer: {predicted if predicted is not None else '(none found)'}\n")


def interactive_chat(model, tokenizer, max_new_tokens: int):
    """Multi-turn chat — the model sees full conversation history each turn."""
    print("\nChat mode — conversation history is preserved across turns.")
    print("Commands: 'reset' to start a new conversation, 'quit' to exit\n")

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if user_input.lower() == "reset":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("  [Conversation reset]\n")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        # Use tokenizer's chat template to build the prompt
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        response, predicted = generate_from_prompt(prompt, model, tokenizer, max_new_tokens)
        history.append({"role": "assistant", "content": response})

        print(f"\nAssistant: {response}")
        if predicted is not None:
            print(f"[Extracted answer: {predicted}]")
        print()

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Qwen Math model.")
    parser.add_argument("--model-path",    required=True,
                        help="Path to the fine-tuned model directory (or HF model ID).")
    parser.add_argument("--eval-file",     default=None,
                        help="JSONL file to batch-evaluate (omit for interactive mode).")
    parser.add_argument("--chat",          action="store_true",
                        help="Start a multi-turn chat session (default: single-turn).")
    parser.add_argument("--max-new-tokens",type=int, default=512)
    parser.add_argument("--verbose",       action="store_true",
                        help="Print per-example results in batch mode.")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Model  : {args.model_path}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    model.eval()

    if args.eval_file:
        batch_eval(args.eval_file, model, tokenizer, args.max_new_tokens, args.verbose)
    elif args.chat:
        interactive_chat(model, tokenizer, args.max_new_tokens)
    else:
        interactive_single(model, tokenizer, args.max_new_tokens)


if __name__ == "__main__":
    main()


"""
Examples:

  # Single-turn interactive (default)
  python eval_model.py --model-path ./qwen_math_sft_output

  # Multi-turn chat
  python eval_model.py --model-path ./qwen_math_sft_output --chat

  # Batch (quiet — just accuracy summary)
  python eval_model.py --model-path ./qwen_math_sft_output \
                       --eval-file dataset/denoised_samples_test.jsonl

  # Batch with per-example output
  python eval_model.py --model-path ./qwen_math_sft_output \
                       --eval-file dataset/denoised_samples_test.jsonl \
                       --verbose

  # Evaluate the base model (before fine-tuning) for comparison
  python eval_model.py --model-path Qwen/Qwen2.5-Math-1.5B-Instruct \
                       --eval-file dataset/denoised_samples_test.jsonl \
                       --verbose
"""
