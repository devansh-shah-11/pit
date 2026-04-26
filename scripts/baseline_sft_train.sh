#!/bin/bash
#SBATCH --job-name=pit-sft
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --output=./logs_pit_sft/%j_%x.out
#SBATCH --error=./logs_pit_sft/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_DIR="/scratch/dns5508/pit"
TRAIN_FILE="$REPO_DIR/dataset/train_data.jsonl"
EVAL_FILE="$REPO_DIR/dataset/test_data.jsonl"
MODEL_NAME="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"
OUTPUT_DIR="$REPO_DIR/checkpoints/pit_sft"
VENV_DIR="$REPO_DIR/venv"

EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=16
MAX_LENGTH=1024
LR=2e-5
EVAL_STEPS=100
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs_pit_sft
mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "PIT SFT Training"
echo "=============================="
echo "Model:      $MODEL_NAME"
echo "Train file: $TRAIN_FILE"
echo "Eval file:  $EVAL_FILE"
echo "Output:     $OUTPUT_DIR"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    pip install -q -r $REPO_DIR/requirements.txt

    python $REPO_DIR/train/baseline_sft_train.py \
      --train_file $TRAIN_FILE \
      --eval_file  $EVAL_FILE \
      --model_name $MODEL_NAME \
      --epochs     $EPOCHS \
      --batch_size $BATCH_SIZE \
      --grad_accum $GRAD_ACCUM \
      --max_length $MAX_LENGTH \
      --lr         $LR \
      --output_dir $OUTPUT_DIR \
      --eval_steps $EVAL_STEPS \
      --use_wandb \
      --use_vllm
  "
