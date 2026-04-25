import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mlx_lm import load, generate  # <-- replace transformers imports
from utils.create_adverserial_dataset_test import extract_answer
from helpers.dataloader import get_gsm_adversarial_dataloaders
from utils.defaults import MODEL_NAME

train_loader, val_loader = get_gsm_adversarial_dataloaders(
    dataset_path="./test.jsonl",
    n_prompts_per_rollout_batch=1,
    seed=42,
    train_split=0.8,
    reduce_test=True
)
print("done")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Train examples: {len(train_loader.dataset)}")
print(f"Val examples:   {len(val_loader.dataset)}")

# ── Load model with MLX ──────────────────────────────────────────────────────
model, tokenizer = load(MODEL_NAME)  # <-- replaces AutoModelForCausalLM + AutoTokenizer


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-6
    except ValueError:
        return predicted.strip() == ground_truth.strip()


# ── Evaluation loop ──────────────────────────────────────────────────────────
total = correct = 0
orig_total = orig_correct = 0
adv_total  = adv_correct  = 0

print("starting the loop")

for batch_idx, batch in enumerate(val_loader):
    prompts        = batch["prompts"]
    ground_truths  = batch["ground_truths"]
    is_adversarial = batch["is_adversarial"]
    print("prompt:", prompts)
    print("gt:", ground_truths)

    for prompt, gt, is_adv in zip(prompts, ground_truths, is_adversarial):
        # MLX generate takes one prompt at a time
        pred_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False,
        )

        predicted = extract_answer(pred_text)
        print("predicted:", predicted)
        match = answers_match(predicted, gt)
        print("match:", match)

        total += 1
        correct += int(match)

        if is_adv:
            adv_total  += 1
            adv_correct += int(match)
        else:
            orig_total  += 1
            orig_correct += int(match)

    print(f"Batch {batch_idx+1}/{len(val_loader)} | "
          f"Running accuracy: {correct/total:.1%}")

# ── Results ──────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"Overall accuracy:      {correct}/{total} = {correct/total:.1%}")
print(f"Original questions:    {orig_correct}/{orig_total} = "
      f"{orig_correct/orig_total:.1%}" if orig_total else "No original questions")
print(f"Adversarial questions: {adv_correct}/{adv_total} = "
      f"{adv_correct/adv_total:.1%}" if adv_total else "No adversarial questions")
print("="*50)