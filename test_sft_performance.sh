#!/bin/bash
#SBATCH --job-name=pit-sft-perf
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --output=./logs_pit/%j_%x.out
#SBATCH --error=./logs_pit/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_DIR="/scratch/dns5508/pit"
BASE_MODEL="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"
BEST_MODEL="$REPO_DIR/checkpoints/pit_sft/best"

# Adversarial test set (base model only, clean + adversarial breakdown)
ADV_EVAL_FILE="$REPO_DIR/dataset/test_data.jsonl"
ADV_OUTPUT_FILE="$REPO_DIR/eval_results_base.json"

# Clean GSM8K test set (base vs best comparison)
GSM8K_EVAL_FILE="$REPO_DIR/usable_dataset/gsm8k_processed_test.json"
GSM8K_OUTPUT_FILE="$REPO_DIR/eval_gsm8k_results.json"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs_pit

echo "=============================="
echo "PIT SFT Performance Evaluation"
echo "=============================="
echo "Base model:       $BASE_MODEL"
echo "Best checkpoint:  $BEST_MODEL"
echo "Adv eval file:    $ADV_EVAL_FILE"
echo "GSM8K eval file:  $GSM8K_EVAL_FILE"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    pip install -q -r $REPO_DIR/requirements.txt

    echo ''
    echo '=== [1/2] Adversarial test set — base model (clean + adversarial breakdown) ==='
    python $REPO_DIR/eval_sft_base_model.py \
      --model_name  $BASE_MODEL \
      --eval_file   $ADV_EVAL_FILE \
      --batch_size  16 \
      --max_new_tokens 1024 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $ADV_OUTPUT_FILE \
      --use_vllm

    echo ''
    echo '=== [2/2] Clean GSM8K test set — base vs best checkpoint ==='
    python $REPO_DIR/eval_sft_gsm8k.py \
      --base_model  $BASE_MODEL \
      --best_model  $BEST_MODEL \
      --eval_file   $GSM8K_EVAL_FILE \
      --batch_size  16 \
      --max_new_tokens 1024 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $GSM8K_OUTPUT_FILE \
      --use_vllm
  "