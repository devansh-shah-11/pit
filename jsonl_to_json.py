#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def jsonl_to_json(input_file: str) -> None:
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if input_path.suffix != ".jsonl":
        raise ValueError(f"Input file must be a .jsonl file, got: {input_path.suffix}")

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: invalid JSON — {e}")

    output_path = input_path.parent / f"{input_path.stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(records)} records -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a JSONL file to a JSON array file."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the input .jsonl file.",
    )
    args = parser.parse_args()
    jsonl_to_json(args.input_file)


if __name__ == "__main__":
    main()