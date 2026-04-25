import json
import torch
from torch.utils.data import DataLoader, Dataset

class GSMAdversarialDataset(Dataset):
    def __init__(self, records):
        self.items = []
        for record in records:
            # Original question
            self.items.append({
                "question": record["original_question"],
                "answer": record["original_answer"],
                "is_adversarial": False,
            })
            # 3 adversarial questions
            adversarials = record["modified_questions"]["adverserials"]
            answers = record["modified_questions"]["answers"]
            for q, a in zip(adversarials, answers):
                self.items.append({
                    "question": q,
                    "answer": a,
                    "is_adversarial": True,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def format_prompt(question: str) -> str:
    return f"""Solve this math problem step by step:

{question}

Provide your final answer in the format:
[reasoning steps]
####
[final answer (just the number)]"""


def get_gsm_adversarial_dataloaders(
    dataset_path: str,
    n_prompts_per_rollout_batch: int,
    seed: int = 42,
    train_split: float = 0.8,
    reduce_test: bool = False,
):
    # Load JSONL
    records = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Train/test split at the record level (before expansion)
    # so adversarials from the same question stay in the same split
    rng = torch.Generator().manual_seed(seed)
    n_total = len(records)
    n_train = int(n_total * train_split)
    n_test = n_total - n_train
    indices = torch.randperm(n_total, generator=rng).tolist()
    train_records = [records[i] for i in indices[:n_train]]
    test_records  = [records[i] for i in indices[n_train:]]

    train_dataset = GSMAdversarialDataset(train_records)
    test_dataset  = GSMAdversarialDataset(test_records)

    if reduce_test:
        test_subset_size = max(1, int(0.3 * len(test_dataset)))
        test_dataset, _ = torch.utils.data.random_split(
            test_dataset,
            [test_subset_size, len(test_dataset) - test_subset_size],
            generator=torch.Generator().manual_seed(seed),
        )

    def collate_fn(batch):
        return {
            "prompts":       [format_prompt(item["question"]) for item in batch],
            "ground_truths": [item["answer"] for item in batch],
            "is_adversarial":[item["is_adversarial"] for item in batch],
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=n_prompts_per_rollout_batch,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=n_prompts_per_rollout_batch,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader