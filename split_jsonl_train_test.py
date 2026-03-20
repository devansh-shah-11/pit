#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path


def split_json(input_file: str, split_ratio: float = 0.8) -> None:
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not input_path.suffix == ".json":
        raise ValueError(f"Input file must be a .json file, got: {input_path.suffix}")
    if not 0 < split_ratio < 1:
        raise ValueError(f"Split ratio must be between 0 and 1, got: {split_ratio}")

    # Read all records
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Input JSON file must contain a top-level array.")
    if not records:
        raise ValueError("Input file is empty.")

    # Shuffle and split
    random.shuffle(records)
    split_index = int(len(records) * split_ratio)
    train_records = records[:split_index]
    test_records = records[split_index:]

    # Build output paths: a.json -> a_train.json, a_test.json
    stem = input_path.stem
    parent = input_path.parent
    train_path = parent / f"{stem}_train.json"
    test_path = parent / f"{stem}_test.json"

    # Write output files
    for path, data in [(train_path, train_records), (test_path, test_records)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Total records : {len(records)}")
    print(f"Train records : {len(train_records)}  ->  {train_path}")
    print(f"Test records  : {len(test_records)}  ->  {test_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a JSON file into train and test sets."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the input .json file.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8).",
    )
    args = parser.parse_args()
    split_json(args.input_file, args.split_ratio)


if __name__ == "__main__":
    main()