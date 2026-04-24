"""
Quick local test: generate 5 adversarial examples with noise localization targets.

Usage:
    python generate_test_examples.py
    python generate_test_examples.py --input dataset/gsm8k_processed_train.json --n 5 --output test_output.jsonl
"""

import argparse
import json

from make_paraphrase import make_adverserials_for_one_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/gsm8k_processed_train.json")
    parser.add_argument("--n", type=int, default=5, help="Number of questions to process")
    parser.add_argument("--output", default="test_adversarial_output.jsonl")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    records = records[:args.n]
    print(f"Generating adversarial examples for {len(records)} questions -> {args.output}\n")

    with open(args.output, "w", encoding="utf-8") as out:
        for i, record in enumerate(records):
            question = record.get("question")
            answer_ref = record.get("answer")

            print(f"[{i+1}/{len(records)}] {question[:80]}...")

            results = make_adverserials_for_one_question(
                question=question,
                answer_ref=answer_ref,
                limit=1,
            )

            out_record = {
                "original_question": question,
                "original_answer": answer_ref,
                "original_raw": record.get("raw"),
                "modified_questions": results,
            }
            out.write(json.dumps(out_record) + "\n")
            out.flush()

            for adv in results["adverserials"]:
                print(f"  adversarial: {adv[:120]}...")
            print()

    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
