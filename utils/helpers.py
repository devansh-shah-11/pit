import torch
import random
import torch.nn.functional as F


def create_batch(dataset, batch_size, tokenizer, max_len, device):
    batch = random.sample(dataset, batch_size)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        # Prompt + target
        prompt = f"Question: {item['question']}\nAnswer:"
        target_text = item["combined"]
        full_text = prompt + " " + target_text

        # Tokenize full sequence
        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        input_ids = tokenized_full["input_ids"][0]
        attention_mask = tokenized_full["attention_mask"][0]

        labels = input_ids.clone()

        # ----------------------------------
        # Mask padding tokens
        # ----------------------------------
        labels[attention_mask == 0] = -100

        # ----------------------------------
        # Mask prompt tokens (FIXED)
        # ----------------------------------
        tokenized_prompt = tokenizer(
            prompt,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        prompt_len = tokenized_prompt["input_ids"].shape[1]

        # Find where real tokens start (because of left padding)
        seq_len = attention_mask.sum().item()
        start_idx = max_len - seq_len  # where actual tokens begin

        # Mask prompt part inside real tokens
        labels[start_idx : start_idx + prompt_len] = -100

        # ----------------------------------
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    # Stack
    input_tensor = torch.stack(input_ids_list).to(device)
    attention_mask_tensor = torch.stack(attention_mask_list).to(device)
    target_tensor = torch.stack(labels_list).to(device)

    return input_tensor, attention_mask_tensor, target_tensor



def data_loader(data, batch_size, max_len, device, tokenizer):
    batch = random.sample(data, batch_size)

    input_ids_list = []
    labels_list = []

    for item in batch:
        prompt = f"Question: {item['q']}\nAnswer:"
        full_text = prompt + " " + item["a"]

        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        tokenized_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        input_ids = tokenized_full["input_ids"][0]
        labels = input_ids.clone()

        # mask prompt tokens
        prompt_len = tokenized_prompt["input_ids"].shape[1]
        labels[:prompt_len] = -100

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    input_tensor = torch.stack(input_ids_list).to(device)
    target_tensor = torch.stack(labels_list).to(device)

    return input_tensor, target_tensor

# =========================
# 5. LOSS
# =========================
def run_cross_entropy_util(logits, targets):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100
    )

# =========================
# 6. CHECKPOINT
# =========================
def save_checkpoint(model, optimizer, step, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step
    }, path)