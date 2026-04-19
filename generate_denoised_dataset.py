#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import requests

from utils.defaults import EXTERNAL_LLM
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DENOISING_PROMPT = """\
You are a math reasoning expert. You will be given:
1. An original clean math question
2. The correct step-by-step solution to the original question
3. An adversarial version of the same question that contains irrelevant noise and distractors

Your task: Produce the correct step-by-step reasoning for the adversarial question, \
identifying and ignoring all irrelevant information. Focus only on the mathematically \
relevant facts. The answer must be the same as the original.

Format your response EXACTLY like this (step-by-step reasoning, then a line with #### and the number):
<reasoning steps>
#### <final numerical answer>

The correct final answer is: {original_answer}

--- Original Question ---
{original_question}

--- Correct Solution ---
{original_raw}

--- Adversarial Question (contains noise to ignore) ---
{adversarial_question}

Provide the correct reasoning now:\
"""


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_raw_response(raw: str):
    """Split a raw response on #### into (reasoning, answer)."""
    if "####" in raw:
        parts = raw.split("####", 1)
        reasoning = parts[0].strip().replace("\n", " ")
        answer = parts[1].strip().split()[0] if parts[1].strip() else ""
    else:
        reasoning = raw.strip().replace("\n", " ")
        answer = ""
    return reasoning, answer


def normalize_answer(ans: str) -> str:
    """Strip currency, commas, whitespace; normalise to bare number string."""
    ans = ans.strip().replace("$", "").replace(",", "").replace("%", "")
    ans = ans.strip()
    try:
        # Normalise floats: 10.0 -> 10, 10.60 -> 10.6
        f = float(ans)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return ans.lower()


def build_combined(reasoning: str, answer: str) -> str:
    reasoning = normalize_text(reasoning)
    answer = normalize_text(answer)
    return f"<reasoning>{reasoning}</reasoning><answer>{answer}</answer>"


def call_llm(prompt: str, api_key: str, model: str, delay: float) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    time.sleep(delay)
    return content


def make_clean_sample(entry: dict) -> dict:
    original_question = entry["original_question"]
    original_answer = entry["original_answer"]
    original_raw = entry["original_raw"]

    reasoning, answer = parse_raw_response(original_raw)
    combined = build_combined(reasoning, answer or original_answer)

    return {
        "question": original_question,
        "original_question": original_question,
        "answer": original_answer,
        "raw": original_raw,
        "reasoning": reasoning,
        "combined": combined,
        "type": "clean",
        "answer_match": True,
    }


def make_adversarial_sample(
    entry: dict,
    adversarial_question: str,
    api_key: str,
    model: str,
    delay: float,
) -> dict:
    original_question = entry["original_question"]
    original_answer = entry["original_answer"]
    original_raw = entry["original_raw"]

    prompt = DENOISING_PROMPT.format(
        original_answer=original_answer,
        original_question=original_question,
        original_raw=original_raw,
        adversarial_question=adversarial_question,
    )

    llm_raw = call_llm(prompt, api_key, model, delay)

    reasoning, extracted_answer = parse_raw_response(llm_raw)

    answer_match = normalize_answer(extracted_answer) == normalize_answer(original_answer)

    # Use ground-truth answer in combined so training target is always correct
    combined = build_combined(reasoning, original_answer)
    # Reconstruct raw with clean #### line
    clean_raw = f"{reasoning}\n#### {original_answer}"

    return {
        "question": adversarial_question,
        "original_question": original_question,
        "answer": original_answer,
        "raw": clean_raw,
        "reasoning": reasoning,
        "combined": combined,
        "type": "adversarial",
        "answer_match": answer_match,
        "llm_raw_response": llm_raw,
    }


def generate(args):
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No API key provided. Set OPENROUTER_API_KEY env var or pass --api-key."
        )

    model = args.model or EXTERNAL_LLM
    rng = random.Random(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all entries first so we can report totals
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    total_clean = 0
    total_adversarial = 0
    total_match = 0
    total_mismatch = 0

    print(f"Processing {len(entries)} entries from {input_path}")
    print(f"Model: {model}")
    print(f"Adversarials per entry: {args.num_adversarials}  |  delay: {args.delay}s\n")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, entry in enumerate(entries, start=1):
            adversarials = entry["modified_questions"]["adverserials"]

            # --- Clean sample (always included) ---
            clean_sample = make_clean_sample(entry)
            f_out.write(json.dumps(clean_sample) + "\n")
            total_clean += 1

            # --- Adversarial samples ---
            # Randomly select num_adversarials of the available adversarials
            n = min(args.num_adversarials, len(adversarials))
            chosen = rng.sample(adversarials, n)

            for adv_q in chosen:
                try:
                    adv_sample = make_adversarial_sample(
                        entry, adv_q, api_key, model, args.delay
                    )
                    f_out.write(json.dumps(adv_sample) + "\n")
                    total_adversarial += 1
                    if adv_sample["answer_match"]:
                        total_match += 1
                    else:
                        total_mismatch += 1
                        print(
                            f"  [MISMATCH] entry {idx} | "
                            f"expected={entry['original_answer']} | "
                            f"llm_raw={adv_sample['llm_raw_response'][:120]!r}"
                        )
                except Exception as exc:
                    print(f"  [ERROR] entry {idx}, adversarial skipped: {exc}")

            total = total_clean + total_adversarial
            clean_pct = 100.0 * total_clean / total if total else 0
            print(
                f"[{idx}/{len(entries)}] samples={total}  clean={clean_pct:.1f}%  "
                f"adv_match={total_match}/{total_adversarial}"
            )

    total = total_clean + total_adversarial
    print(f"\nDone. Written to {output_path}")
    print(f"Total samples : {total}")
    print(f"Clean         : {total_clean}  ({100*total_clean/total:.1f}%)")
    print(f"Adversarial   : {total_adversarial}  ({100*total_adversarial/total:.1f}%)")
    if total_adversarial:
        print(
            f"Answer match  : {total_match}/{total_adversarial}  "
            f"({100*total_match/total_adversarial:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate denoised training data using an external LLM trainer."
    )
    parser.add_argument(
        "--input",
        default="test.jsonl",
        help="Path to input .jsonl file (default: test.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="dataset/denoised_training.jsonl",
        help="Path to output .jsonl file (default: dataset/denoised_training.jsonl)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key (falls back to OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model to use (default: EXTERNAL_LLM from defaults.py = {EXTERNAL_LLM})",
    )
    parser.add_argument(
        "--num-adversarials",
        type=int,
        default=2,
        help="Number of adversarial variants to process per entry (default: 2 → ~33%% clean)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=10,
        help="Seconds to wait between API calls (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for adversarial selection (default: 42)",
    )
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()