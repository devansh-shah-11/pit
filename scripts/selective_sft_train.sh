#!/bin/bash
#SBATCH --job-name=pit-selective-sft
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --array=0-1

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_DIR="/scratch/dns5508/pit"
TRAIN_FILE="$REPO_DIR/dataset/noise_cot_train.jsonl"
EVAL_FILE="$REPO_DIR/dataset/noise_cot_test.jsonl"
MODEL_NAME="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"

EPOCHS=10
BATCH_SIZE=2
GRAD_ACCUM=16
MAX_LENGTH=1024
LR=2e-5
EVAL_STEPS=10
# ──────────────────────────────────────────────────────────────────────────────

# 2 configs: layers 18-20 (all), layers 23-27 (all)
LAYER_SETS=("18,19,20" "23,24,25,26,27")
SUFFIXES=("18-20_all" "23-27_all")

LAYERS="${LAYER_SETS[$SLURM_ARRAY_TASK_ID]}"
SUFFIX="${SUFFIXES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$REPO_DIR/checkpoints/pit_selective_${SUFFIX}"

mkdir -p ./logs
mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "PIT Selective-Layer SFT"
echo "=============================="
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Layers:      $LAYERS"
echo "Model:       $MODEL_NAME"
echo "Train file:  $TRAIN_FILE"
echo "Eval file:   $EVAL_FILE"
echo "Output:      $OUTPUT_DIR"
echo "Epochs:      $EPOCHS  (eval every $EVAL_STEPS steps)"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    python $REPO_DIR/train/baseline_sft_noise_cot_selective.py \
      --train_file   $TRAIN_FILE \
      --eval_file    $EVAL_FILE \
      --model_name   $MODEL_NAME \
      --train_layers '$LAYERS' \
      --epochs       $EPOCHS \
      --batch_size   $BATCH_SIZE \
      --grad_accum   $GRAD_ACCUM \
      --max_length   $MAX_LENGTH \
      --lr           $LR \
      --output_dir   $OUTPUT_DIR \
      --eval_steps   $EVAL_STEPS \
      --use_wandb \
      --use_vllm
  "
