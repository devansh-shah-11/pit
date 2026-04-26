#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests

from utils.defaults import EXTERNAL_LLM
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DENOISING_PROMPT = """\
You are a math reasoning expert. You will be given:
1. An original clean math question
2. The correct step-by-step solution to the original question
3. An adversarial version of the same question that contains irrelevant noise and distractors

Your task: Produce the correct step-by-step reasoning for the adversarial question. \
You MUST begin by explicitly listing the irrelevant noise/distractors you are ignoring \
(wrapped in the noise markers), then solve using only the mathematically relevant facts. \
The answer must be the same as the original.

Format your response EXACTLY like this:
@%^^ NOISE @%^^
&&&& <list the irrelevant facts you are ignoring> &&&&
<reasoning steps>
#### <ans>

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
    ans = ans.strip().replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        f = float(ans)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return ans.lower()



def call_llm(
    prompt: str,
    api_key: str,
    model: str,
    delay: float,
    max_retries: int = 6,
    base_backoff: float = 5.0,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, timeout=120
        )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else base_backoff * (2 ** attempt)
            print(f"  [429] Rate limited. Retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        time.sleep(delay)
        return content
    raise RuntimeError(f"Exceeded {max_retries} retries due to rate limiting.")


def make_clean_sample(entry: dict) -> dict:
    reasoning = entry["reasoning"].strip().replace("\n", " ")
    clean_reasoning = f"No noise identified in the question. {reasoning}"
    clean_raw = f"{clean_reasoning}\n#### {entry['answer']}"

    return {
        "question": entry["question"],
        "answer": entry["answer"],
        "raw": clean_raw,
        "reasoning": clean_reasoning,
        "type": "clean",
        "answer_match": True,
    }


def make_adversarial_sample(
    entry: dict,
    original_question: str,
    original_reasoning: str,
    api_key: str,
    model: str,
    delay: float,
) -> dict:
    answer = entry["answer"]
    original_raw = f"{original_reasoning}\n#### {answer}"

    prompt = DENOISING_PROMPT.format(
        original_question=original_question,
        original_raw=original_raw,
        adversarial_question=entry["question"],
    )

    llm_raw = call_llm(prompt, api_key, model, delay)
    reasoning, extracted_answer = parse_raw_response(llm_raw)
    answer_match = normalize_answer(extracted_answer) == normalize_answer(answer)
    clean_raw = f"{reasoning}\n#### {answer}"

    return {
        "question": entry["question"],
        "original_question": original_question,
        "answer": answer,
        "raw": clean_raw,
        "reasoning": reasoning,
        "type": "adversarial",
        "answer_match": answer_match,
        "llm_raw_response": llm_raw,
    }


def load_existing_questions(output_path: Path) -> set:
    existing = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        existing.add(sample.get("question", ""))
                    except json.JSONDecodeError:
                        pass
    return existing


def generate(args):
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No API key provided. Set OPENROUTER_API_KEY env var or pass --api-key."
        )

    model = args.model or EXTERNAL_LLM

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_questions = load_existing_questions(output_path)
    if existing_questions:
        print(f"Found {len(existing_questions)} already-processed questions in {output_path}, skipping them.")

    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    start = args.start_from
    end = args.end_to if args.end_to is not None else len(entries)
    entries = entries[start:end]

    total_clean = 0
    total_adversarial = 0
    total_match = 0
    total_mismatch = 0
    total_skipped = 0

    print(f"Processing entries [{start}:{end}] ({len(entries)} entries) from {input_path}")
    print(f"Model: {model}  |  delay: {args.delay}s\n")

    current_original_question = None
    current_original_reasoning = None

    with open(output_path, "a", encoding="utf-8") as f_out:
        for idx, entry in enumerate(entries, start=start + 1):
            question = entry["question"]
            source = entry["source"]

            if source == "original":
                current_original_question = question
                current_original_reasoning = entry["reasoning"]

                if question in existing_questions:
                    total_skipped += 1
                else:
                    sample = make_clean_sample(entry)
                    f_out.write(json.dumps(sample) + "\n")
                    f_out.flush()
                    existing_questions.add(question)
                    total_clean += 1

            elif source == "adversarial":
                if current_original_question is None:
                    print(f"  [WARN] entry {idx}: adversarial with no preceding original, skipping.")
                    total_skipped += 1
                    continue

                if question in existing_questions:
                    total_skipped += 1
                    continue

                try:
                    sample = make_adversarial_sample(
                        entry,
                        current_original_question,
                        current_original_reasoning,
                        api_key,
                        model,
                        args.delay,
                    )
                    f_out.write(json.dumps(sample) + "\n")
                    f_out.flush()
                    existing_questions.add(question)
                    total_adversarial += 1
                    if sample["answer_match"]:
                        total_match += 1
                    else:
                        total_mismatch += 1
                        print(
                            f"  [MISMATCH] entry {idx} | "
                            f"expected={entry['answer']} | "
                            f"llm_raw={sample['llm_raw_response'][:120]!r}"
                        )
                except Exception as exc:
                    print(f"  [ERROR] entry {idx}, adversarial skipped: {exc}")

            total = total_clean + total_adversarial
            clean_pct = 100.0 * total_clean / total if total else 0
            print(
                f"[{idx}/{end}] samples={total}  skipped={total_skipped}  clean={clean_pct:.1f}%  "
                f"adv_match={total_match}/{total_adversarial}"
            )

    total = total_clean + total_adversarial
    print(f"\nDone. Written to {output_path}")
    print(f"Total new samples : {total}")
    print(f"Skipped (existing): {total_skipped}")
    if total:
        print(f"Clean             : {total_clean}  ({100*total_clean/total:.1f}%)")
        print(f"Adversarial       : {total_adversarial}  ({100*total_adversarial/total:.1f}%)")
    if total_adversarial:
        print(
            f"Answer match      : {total_match}/{total_adversarial}  "
            f"({100*total_match/total_adversarial:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate denoised training data using an external LLM."
    )
    parser.add_argument("--input", default="dataset/train_data.jsonl")
    parser.add_argument("--output", default="dataset/denoised_training.jsonl")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None,
                        help=f"Model to use (default: {EXTERNAL_LLM})")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start index (0-based, inclusive)")
    parser.add_argument("--end-to", type=int, default=None,
                        help="End index (0-based, exclusive, default: all)")
    parser.add_argument("--delay", type=float, default=10,
                        help="Seconds to wait after each API call (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
