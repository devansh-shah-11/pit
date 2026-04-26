#!/bin/bash
#SBATCH --job-name=pit-clean-cot
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_DIR="/scratch/dns5508/pit"
TRAIN_FILE="$REPO_DIR/dataset/noise_cot_train.jsonl"
EVAL_FILE="$REPO_DIR/dataset/noise_cot_test.jsonl"
MODEL_NAME="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"
OUTPUT_DIR="$REPO_DIR/checkpoints/pit_noise_cot_clean"

# ~201 train rows, clean-only is ~half (~100); effective batch = 2 * 16 = 32
# => ~3 steps/epoch; 10 epochs = ~30 steps total; eval every 5 steps
EPOCHS=10
BATCH_SIZE=2
GRAD_ACCUM=16
MAX_LENGTH=1024
LR=2e-5
EVAL_STEPS=6
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs
mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "PIT Clean-CoT SFT Training"
echo "=============================="
echo "Model:      $MODEL_NAME"
echo "Train file: $TRAIN_FILE  (clean samples only)"
echo "Eval file:  $EVAL_FILE"
echo "Output:     $OUTPUT_DIR"
echo "Epochs:     $EPOCHS  (eval every $EVAL_STEPS steps)"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    pip install -q -r $REPO_DIR/requirements.txt

    python $REPO_DIR/train/baseline_sft_noise_cot.py \
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
      --clean_only \
      --use_wandb \
      --use_vllm
  "
