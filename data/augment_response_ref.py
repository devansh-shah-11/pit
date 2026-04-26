#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_ref_index(ref_input: str) -> dict:
    """Load ref JSON file and build a lookup index by 'question' field."""
    ref_path = Path(ref_input)
    if not ref_path.exists():
        raise FileNotFoundError(f"Ref input file not found: {ref_input}")

    with open(ref_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    if not isinstance(ref_data, list):
        raise ValueError("Ref input file must contain a JSON array at the top level.")

    # Build index: question -> list of matching entries
    index = defaultdict(list)
    for entry in ref_data:
        if "question" not in entry:
            raise ValueError(f"Ref entry missing 'question' field: {entry}")
        index[entry["question"]].append(entry)

    return index


def augment(raw_input: str, ref_input: str) -> None:
    raw_path = Path(raw_input)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw input file not found: {raw_input}")
    if raw_path.suffix != ".jsonl":
        raise ValueError(f"Raw input must be a .jsonl file, got: {raw_path.suffix}")

    ref_index = load_ref_index(ref_input)

    output_path = raw_path.parent / f"{raw_path.stem}_augmented.jsonl"

    with open(raw_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)

            if "original_question" not in entry:
                raise ValueError(f"Line {line_num}: missing 'original_question' field.")

            question = entry["original_question"]
            matches = ref_index.get(question, [])

            assert len(matches) > 0, (
                f"Line {line_num}: no match found in ref for question:\n  {question!r}"
            )
            assert len(matches) == 1, (
                f"Line {line_num}: expected exactly 1 match in ref, found {len(matches)} "
                f"for question:\n  {question!r}"
            )

            ref_entry = matches[0]

            if "raw" not in ref_entry:
                raise ValueError(
                    f"Line {line_num}: matched ref entry has no 'raw' field "
                    f"for question:\n  {question!r}"
                )

            entry["response_ref"] = ref_entry["raw"]
            f_out.write(json.dumps(entry) + "\n")

    print(f"Done. Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment a JSONL file with 'response_ref' values from a ref JSON file."
    )
    parser.add_argument(
        "--raw-input",
        required=True,
        help="Path to the raw input .jsonl file (must have 'original_question' field).",
    )
    parser.add_argument(
        "--ref-input",
        required=True,
        help="Path to the ref input .json file (must have 'question' and 'raw' fields).",
    )
    args = parser.parse_args()
    augment(args.raw_input, args.ref_input)


if __name__ == "__main__":
    main()