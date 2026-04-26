#!/bin/bash
#SBATCH --job-name=pit-noise-cot-perf
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_DIR="/scratch/dns5508/pit"
BASE_MODEL="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"
CLEAN_MODEL="$REPO_DIR/checkpoints/pit_noise_cot_clean/best"
ALL_MODEL="$REPO_DIR/checkpoints/pit_noise_cot_all/best"

NOISE_COT_TEST="$REPO_DIR/dataset/noise_cot_test.jsonl"
GSM8K_TEST="$REPO_DIR/usable_dataset/gsm8k_processed_test.json"

RESULTS_DIR="$REPO_DIR/eval_results_noise_cot"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs
mkdir -p "$RESULTS_DIR"

echo "=============================="
echo "Noise-CoT Model Evaluation"
echo "=============================="
echo "Base model:   $BASE_MODEL"
echo "Clean model:  $CLEAN_MODEL"
echo "All model:    $ALL_MODEL"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    # ── [1/6] Base model — noise_cot_test ─────────────────────────────────────
    # Base model: use chat template (it's an instruct model).
    # echo ''
    # echo '=== [1/6] Base model — noise_cot_test ==='
    # python $REPO_DIR/evaluation/eval_sft_base_model.py \
    #   --model_name  $BASE_MODEL \
    #   --eval_file   $NOISE_COT_TEST \
    #   --batch_size  16 \
    #   --max_new_tokens 512 \
    #   --max_prompt_length 1024 \
    #   --gpu_memory_utilization 0.85 \
    #   --output_file $RESULTS_DIR/base_noise_cot_test.json \
    #   --use_vllm \
    #   --use_chat_template

    # ── [2/6] Noise-CoT clean model — noise_cot_test ──────────────────────────
    echo ''
    echo '=== [2/6] Noise-CoT clean model — noise_cot_test ==='
    python $REPO_DIR/evaluation/eval_sft_base_model.py \
      --model_name  $CLEAN_MODEL \
      --eval_file   $NOISE_COT_TEST \
      --batch_size  16 \
      --max_new_tokens 512 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $RESULTS_DIR/clean_model_noise_cot_test.json \
      --use_vllm

    # ── [3/6] Noise-CoT all model — noise_cot_test ────────────────────────────
    echo ''
    echo '=== [3/6] Noise-CoT all model — noise_cot_test ==='
    python $REPO_DIR/evaluation/eval_sft_base_model.py \
      --model_name  $ALL_MODEL \
      --eval_file   $NOISE_COT_TEST \
      --batch_size  16 \
      --max_new_tokens 512 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $RESULTS_DIR/all_model_noise_cot_test.json \
      --use_vllm

    # ── [4/6] Base model — GSM8K ──────────────────────────────────────────────
    # Base model: use chat template (it's an instruct model).
    # echo ''
    # echo '=== [4/6] Base model — GSM8K ==='
    # python $REPO_DIR/evaluation/eval_sft_base_model.py \
    #   --model_name  $BASE_MODEL \
    #   --eval_file   $GSM8K_TEST \
    #   --batch_size  16 \
    #   --max_new_tokens 512 \
    #   --max_prompt_length 1024 \
    #   --gpu_memory_utilization 0.85 \
    #   --output_file $RESULTS_DIR/base_gsm8k.json \
    #   --use_vllm \
    #   --use_chat_template

    # ── [5/6] Noise-CoT clean model — GSM8K ───────────────────────────────────
    echo ''
    echo '=== [5/6] Noise-CoT clean model — GSM8K ==='
    python $REPO_DIR/evaluation/eval_sft_base_model.py \
      --model_name  $CLEAN_MODEL \
      --eval_file   $GSM8K_TEST \
      --batch_size  16 \
      --max_new_tokens 512 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $RESULTS_DIR/clean_model_gsm8k.json \
      --use_vllm

    # ── [6/6] Noise-CoT all model — GSM8K ─────────────────────────────────────
    echo ''
    echo '=== [6/6] Noise-CoT all model — GSM8K ==='
    python $REPO_DIR/evaluation/eval_sft_base_model.py \
      --model_name  $ALL_MODEL \
      --eval_file   $GSM8K_TEST \
      --batch_size  16 \
      --max_new_tokens 512 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $RESULTS_DIR/all_model_gsm8k.json \
      --use_vllm

    echo ''
    echo '=============================='
    echo 'All results saved to $RESULTS_DIR'
    echo '=============================='
  "
