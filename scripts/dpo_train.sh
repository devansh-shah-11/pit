#!/bin/bash
#SBATCH --job-name=pit-dpo
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_DIR="/scratch/dns5508/pit"
TRAIN_FILE="$REPO_DIR/dataset/dpo_train.jsonl"
EVAL_FILE="$REPO_DIR/dataset/test_data.jsonl"
MODEL_NAME="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"   # MUST be the model used to generate rejected
OUTPUT_DIR="$REPO_DIR/checkpoints/pit_dpo"

# Core DPO hyperparameters — tuned for Qwen2.5-1.5B-Instruct_gsm8k on a single A100.
EPOCHS=2
BATCH_SIZE=2
GRAD_ACCUM=8
LR=5e-7              # DPO sweet spot is usually 1e-7..1e-6. Try {1e-7, 5e-7, 1e-6}.
BETA=0.1             # Try {0.05, 0.1, 0.3}. Higher = closer to reference.
LOSS_TYPE=sigmoid    # Try {sigmoid, ipo}; ipo is length-insensitive.
LABEL_SMOOTHING=0.0  # 0.05–0.1 (cDPO) if labels are noisy.

MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024
GEN_MAX_NEW_TOKENS=1024
EVAL_STEPS=50
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs
mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "PIT DPO Training"
echo "=============================="
echo "Model:           $MODEL_NAME"
echo "Train file:      $TRAIN_FILE"
echo "Eval file:       $EVAL_FILE"
echo "Output:          $OUTPUT_DIR"
echo "Epochs:          $EPOCHS  bs=$BATCH_SIZE  ga=$GRAD_ACCUM  lr=$LR"
echo "DPO:             beta=$BETA  loss=$LOSS_TYPE  label_smoothing=$LABEL_SMOOTHING"
echo "Lengths:         max=$MAX_LENGTH  prompt=$MAX_PROMPT_LENGTH  gen=$GEN_MAX_NEW_TOKENS"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    python $REPO_DIR/train/dpo_train.py \
      --train_file         $TRAIN_FILE \
      --eval_file          $EVAL_FILE \
      --model_name         $MODEL_NAME \
      --output_dir         $OUTPUT_DIR \
      --epochs             $EPOCHS \
      --batch_size         $BATCH_SIZE \
      --grad_accum         $GRAD_ACCUM \
      --lr                 $LR \
      --beta               $BETA \
      --loss_type          $LOSS_TYPE \
      --label_smoothing    $LABEL_SMOOTHING \
      --max_length         $MAX_LENGTH \
      --max_prompt_length  $MAX_PROMPT_LENGTH \
      --gen_max_new_tokens $GEN_MAX_NEW_TOKENS \
      --eval_steps         $EVAL_STEPS \
      --use_wandb \
      --use_vllm
  "
