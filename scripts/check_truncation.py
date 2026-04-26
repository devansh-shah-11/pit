"""
Check how many training samples exceed max_length after tokenization.
No GPU required — runs on CPU.

Usage:
    python scripts/check_truncation.py \
        --train_file dataset/noise_cot_train.jsonl \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --max_length 1024
"""

import argparse
import json
from transformers import AutoTokenizer


def build_prompt(question):
    return f"Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    samples = []
    with open(args.train_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {args.train_file}")

    completion_field = "raw" if "raw" in samples[0] else None

    stats = {
        "clean":       {"total": 0, "trunc": 0, "lengths": []},
        "adversarial": {"total": 0, "trunc": 0, "lengths": []},
    }

    for s in samples:
        completion = s.get("raw") or (
            s.get("reasoning", "") + "\n####\n" + str(s.get("answer", ""))
        )
        full_text = build_prompt(s["question"]) + "\n" + completion
        n = len(tokenizer(full_text, add_special_tokens=True)["input_ids"])
        bucket = "adversarial" if s.get("type") == "adversarial" else "clean"
        stats[bucket]["total"] += 1
        stats[bucket]["lengths"].append(n)
        if n > args.max_length:
            stats[bucket]["trunc"] += 1

    print(f"\n{'='*55}")
    print(f"Truncation check @ max_length={args.max_length}")
    print(f"{'='*55}")
    for bucket, v in stats.items():
        if v["total"] == 0:
            continue
        lengths = v["lengths"]
        pct = 100 * v["trunc"] / v["total"]
        print(f"{bucket:>12}: {v['trunc']:>3}/{v['total']} truncated ({pct:.1f}%)  "
              f"min={min(lengths)}  median={sorted(lengths)[len(lengths)//2]}  max={max(lengths)}")
    print(f"{'='*55}")
    all_lengths = stats["clean"]["lengths"] + stats["adversarial"]["lengths"]
    print(f"  Recommended max_length to cover all samples: {max(all_lengths)}")


if __name__ == "__main__":
    main()
