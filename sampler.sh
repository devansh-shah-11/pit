#!/bin/bash

# Usage: ./sample_json.sh <input_file> <n> <output_file>

INPUT=$1
N=$2
OUTPUT=$3

jq --argjson n "$N" '[to_entries | .[] | .value] | [limit($n; .[])]' "$INPUT" > "$OUTPUT"

echo "Sampled $N entries from $INPUT -> $OUTPUT"