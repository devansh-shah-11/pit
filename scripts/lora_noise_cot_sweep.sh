#!/bin/bash
#SBATCH --job-name=pit-lora-sweep
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
TRAIN_FILE="$REPO_DIR/dataset/noise_cot_train.jsonl"
EVAL_FILE="$REPO_DIR/dataset/noise_cot_test.jsonl"
MODEL_NAME="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"
OUTPUT_ROOT="$REPO_DIR/checkpoints/pit_lora_sweep"

# ~200 train rows, effective batch = 2 * 16 = 32 => ~6 steps/epoch
EPOCHS=5
BATCH_SIZE=2
GRAD_ACCUM=16
MAX_LENGTH=1024
EVAL_STEPS=10

# Run a subset by passing config names: sbatch lora_noise_cot_sweep.sh small balanced
if [ "$#" -gt 0 ]; then
  CONFIGS="$*"
else
  CONFIGS="small balanced high_capacity"
fi
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs
mkdir -p "$OUTPUT_ROOT"

echo "=============================="
echo "PIT LoRA Sweep"
echo "=============================="
echo "Model:      $MODEL_NAME"
echo "Configs:    $CONFIGS"
echo "Output:     $OUTPUT_ROOT"
echo "Epochs:     $EPOCHS  (eval every $EVAL_STEPS steps)"
echo "=============================="

# Enter the container ONCE; loop over configs inside it so we pay the
# singularity / conda startup cost a single time.
singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    set -e
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    for CFG in $CONFIGS; do
      OUT=\"$OUTPUT_ROOT/\$CFG\"
      echo ''
      echo \"─── [\$CFG] → \$OUT ───────────────────────────────\"
      mkdir -p \"\$OUT\"

      python $REPO_DIR/train/baseline_sft_noise_cot_lora.py \
        --config     \$CFG \
        --train_file $TRAIN_FILE \
        --eval_file  $EVAL_FILE \
        --model_name $MODEL_NAME \
        --epochs     $EPOCHS \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --max_length $MAX_LENGTH \
        --eval_steps $EVAL_STEPS \
        --output_dir \"\$OUT\" \
        --merge_and_save \
        --use_wandb \
        --use_vllm
    done
  "

echo ""
echo "=============================="
echo "Sweep complete. Outputs under: $OUTPUT_ROOT/<config>/{best,final,final_merged}"
echo "=============================="
