#!/bin/bash
#SBATCH --job-name=pit-eval-gsm8k
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
EVAL_FILE="$REPO_DIR/usable_dataset/gsm8k_processed_test_sample.json"
BASE_MODEL="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"
BEST_MODEL="$REPO_DIR/checkpoints/pit_sft/best"
OUTPUT_FILE="$REPO_DIR/eval_gsm8k_results.json"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs_pit

echo "=============================="
echo "GSM8K Clean Test Evaluation"
echo "=============================="
echo "Base model:  $BASE_MODEL"
echo "Best ckpt:   $BEST_MODEL"
echo "Eval file:   $EVAL_FILE"
echo "Output:      $OUTPUT_FILE"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    pip install -q -r $REPO_DIR/requirements.txt

    python $REPO_DIR/eval_sft_gsm8k.py \
      --base_model  $BASE_MODEL \
      --best_model  $BEST_MODEL \
      --eval_file   $EVAL_FILE \
      --batch_size  16 \
      --max_new_tokens 256 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $OUTPUT_FILE \
      --use_vllm
  "
