"""
Creates SFT dataset from pit-test.jsonl using original questions + adversarial questions,
all mapped to the original reasoning (CoT) and answer.

Each adversarial question for a given original question is treated as a variant that
should produce the same reasoning chain — this is the core PIT training signal.

Output format per line:
  {"question": str, "reasoning": str, "answer": str, "source": "original"|"adversarial"}
"""

import json
import unicodedata
import re
import argparse
from pathlib import Path


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_reasoning(raw: str) -> tuple[str, str]:
    if "####" in raw:
        reasoning, answer = raw.split("####", 1)
        return normalize(reasoning.strip()), normalize(answer.strip())
    return normalize(raw.strip()), ""


def process_file(input_path: str, output_path: str, include_adversarial: bool = True):
    records = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)

            reasoning, answer = extract_reasoning(example["original_raw"])

            # Original question → original reasoning
            records.append({
                "question": normalize(example["original_question"]),
                "reasoning": reasoning,
                "answer": answer,
                "source": "original",
            })

            if not include_adversarial:
                continue

            adversarials = example.get("modified_questions", {}).get("adverserials", [])
            for adv_q in adversarials:
                records.append({
                    "question": normalize(adv_q),
                    "reasoning": reasoning,
                    "answer": answer,
                    "source": "adversarial",
                })

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    orig_count = sum(1 for r in records if r["source"] == "original")
    adv_count = sum(1 for r in records if r["source"] == "adversarial")
    print(f"Written {len(records)} records to {output_path}")
    print(f"  Original: {orig_count}  |  Adversarial: {adv_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/pit-test.jsonl")
    parser.add_argument("--output", default="dataset/pit_sft_dataset.jsonl")
    parser.add_argument("--no-adversarial", action="store_true",
                        help="Only include original questions (no adversarial variants)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    process_file(args.input, args.output, include_adversarial=not args.no_adversarial)
