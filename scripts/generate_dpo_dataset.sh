#!/bin/bash
#SBATCH --job-name=pit-dpo-gen
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_DIR="/scratch/dns5508/pit"
INPUT_FILE="$REPO_DIR/dataset/train_data.jsonl"
OUTPUT_FILE="$REPO_DIR/dataset/dpo_train.jsonl"
CORRECT_FILE="$REPO_DIR/dataset/dpo_model_correct.jsonl"
MODEL_NAME="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"

MAX_NEW_TOKENS=512
NUM_REJECTED_PER_Q=1
TEMPERATURE=0.8
INCLUDE_CLEAN_FAILURES=0   # set to 1 to also probe clean (source=original) questions
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs
mkdir -p "$(dirname "$OUTPUT_FILE")"

EXTRA_FLAGS=""
if [ "$INCLUDE_CLEAN_FAILURES" = "1" ]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --include-clean-failures"
fi

echo "=============================="
echo "PIT DPO Pair Generation"
echo "=============================="
echo "Model:           $MODEL_NAME"
echo "Input:           $INPUT_FILE"
echo "Pairs output:    $OUTPUT_FILE"
echo "Correct output:  $CORRECT_FILE"
echo "Rejected/q:      $NUM_REJECTED_PER_Q  (temperature=$TEMPERATURE if >1)"
echo "Max new tokens:  $MAX_NEW_TOKENS"
echo "Clean failures:  $INCLUDE_CLEAN_FAILURES"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    python $REPO_DIR/data/generate_dpo_dataset.py \
      --input              $INPUT_FILE \
      --output             $OUTPUT_FILE \
      --correct-output     $CORRECT_FILE \
      --model              $MODEL_NAME \
      --max-new-tokens     $MAX_NEW_TOKENS \
      --num-rejected-per-q $NUM_REJECTED_PER_Q \
      --temperature        $TEMPERATURE \
      $EXTRA_FLAGS
  "
