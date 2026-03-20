import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from torch.optim import AdamW
import random
from utils.defaults import *
from utils.helpers import *
import json

import wandb
os.environ["WANDB_API_KEY"] = "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"
wandb.login(key=os.environ["WANDB_API_KEY"])



def main_training_loop():
    wandb.init(
        project="prompt_invariant_training",
        name=f"local-bs-{MODEL_NAME}",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": ITERATIONS,
            "model": "transformer",
        }
    )

    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

    with open(DATASET_GSM_TRAINING, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(DATASET_GSM_TESTING, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Tokenizer + Model
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=1e-6)

    print("Model initialized. Starting training...")

    for it_id in range(ITERATIONS):
        # Create a batch
        input_tensor, attention_mask_tensor, target_tensor = create_batch(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            tokenizer=tokenizer,
            max_len=CONTEXT_LENGTH,
            device=DEVICE
        )

        # Forward pass
        outputs = model(
            input_ids=input_tensor,
            attention_mask=attention_mask_tensor
        )
        logits = outputs.logits

        # Compute loss (masking prompt automatically handled)
        loss = run_cross_entropy_util(logits, target_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if it_id % 10 == 0:
            print(f"[Iteration {it_id}] Loss: {loss.item():.4f}")
            wandb.log({"train_loss": loss.item()}, step=it_id)

        val_batch_size = min(2, len(val_data))

        if it_id % VAL_LOSS_INTERVAL == 0:
            val_input, attention_val_mask_tensor, val_target = create_batch(
                dataset=val_data,
                batch_size=min(2, len(val_data)),  # small batch for speed
                tokenizer=tokenizer,
                max_len=CONTEXT_LENGTH,
                device=DEVICE
            )

            with torch.no_grad():
                val_outputs = model.generate(
                    input_ids=val_input,
                    attention_mask=attention_val_mask_tensor,
                    max_new_tokens=64,
                    do_sample=False
                )

                # Compute val_loss
                val_logits = model(input_ids=val_input).logits
                val_loss = run_cross_entropy_util(val_logits, val_target)

                wandb.log({"val_loss": val_loss}, step=it_id)

            for i in range(val_batch_size):
                prompt_len = val_input.shape[1]

                # Only decode the prompt portion
                prompt_text = f"Question: {val_input[i]}\nAnswer:"

                # Only decode generated tokens (after prompt)
                generated = tokenizer.decode(val_outputs[i], skip_special_tokens=True)

                print(f"\nSample {i + 1}:")
                print("Q:", prompt_text)
                print("Complete A:", generated)

        # Checkpointing
        if it_id % SAVE_CHECK_POINT_ITERATION == 0:
            save_checkpoint(
                model, optimizer, it_id + 1,
                os.path.join(CHECKPOINT_FOLDER, f"checkpoint_{it_id}.pt")
            )

if __name__ == '__main__':
    main_training_loop()