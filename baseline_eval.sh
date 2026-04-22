#!/bin/bash
#SBATCH --job-name=pit-eval
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --output=./logs_pit_eval/%j_%x.out
#SBATCH --error=./logs_pit_eval/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_DIR="/scratch/dns5508/pit"
EVAL_FILE="$REPO_DIR/dataset/test_data.jsonl"
MODEL_NAME="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k"
OUTPUT_FILE="$REPO_DIR/eval_results_base.json"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p ./logs_pit

echo "=============================="
echo "PIT Base Model Evaluation"
echo "=============================="
echo "Model:     $MODEL_NAME"
echo "Eval file: $EVAL_FILE"
echo "Output:    $OUTPUT_FILE"
echo "=============================="

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/home/dns5508/.local/bin:\$PATH
    conda activate llmr

    pip install -q -r $REPO_DIR/requirements.txt

    python $REPO_DIR/eval_base_model.py \
      --model_name  $MODEL_NAME \
      --eval_file   $EVAL_FILE \
      --batch_size  16 \
      --max_new_tokens 256 \
      --max_prompt_length 1024 \
      --gpu_memory_utilization 0.85 \
      --output_file $OUTPUT_FILE \
      --use_vllm
  "
