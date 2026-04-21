"""
SFT training on PIT dataset using HuggingFace Trainer.
Loss is computed only on the completion (reasoning + answer), not the question prompt.
After each eval, exact-match accuracy is computed via generation; the best checkpoint
is saved to {output_dir}/best and the final model to {output_dir}/final.
"""

import json
import os
import argparse

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)


PROMPT_TEMPLATE = """Solve this math problem step by step:

{question}

Provide your final answer in the format:
[reasoning steps]
####
[final answer (just the number)]"""

COMPLETION_TEMPLATE = "{reasoning}\n####\n{answer}"


def extract_answer(text: str) -> str:
    """Pull the line immediately after the last #### marker."""
    parts = text.split("####")
    if len(parts) < 2:
        return ""
    # answer is the first non-empty line after ####
    for line in parts[-1].splitlines():
        line = line.strip()
        if line:
            return line
    return ""


class PITDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt = PROMPT_TEMPLATE.format(question=sample["question"])
        completion = COMPLETION_TEMPLATE.format(
            reasoning=sample["reasoning"],
            answer=sample["answer"],
        )
        full_text = prompt + "\n" + completion

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Tokenize prompt alone to find its token length within the full sequence.
        # add_special_tokens=False avoids a double BOS when the full text already has one.
        prompt_enc = self.tokenizer(
            prompt + "\n",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Mask prompt tokens so loss is only on the completion.
        # +1 accounts for the BOS token prepended by the full-text tokenization.
        prompt_len = prompt_enc["input_ids"].shape[1] + 1
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class AccuracyEvalCallback(TrainerCallback):
    """
    After each evaluation, runs greedy generation on the eval set,
    computes exact-match accuracy, and saves a 'best' checkpoint when
    accuracy improves.
    """

    def __init__(self, eval_dataset, tokenizer, output_dir: str, max_new_tokens: int = 256, batch_size: int = 8, max_prompt_length: int = 1024):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.best_dir = os.path.join(output_dir, "best")
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.max_prompt_length = max_prompt_length
        self.best_accuracy = -1.0

    def on_evaluate(self, args, state, control, model, **kwargs):
        model.eval()
        device = next(model.parameters()).device

        correct = 0
        total = 0

        for start in range(0, len(self.eval_dataset), self.batch_size):
            batch_samples = [self.eval_dataset.samples[i]
                             for i in range(start, min(start + self.batch_size, len(self.eval_dataset)))]

            prompts = [PROMPT_TEMPLATE.format(question=s["question"]) for s in batch_samples]
            gold_answers = [s["answer"] for s in batch_samples]

            self.tokenizer.padding_side = "left"
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_length,
            ).to(device)
            self.tokenizer.padding_side = "right"

            with torch.no_grad():
                outputs = model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # With left-padding, all prompts end at the same position (enc shape[1]),
            # so we can use a single fixed offset for all samples in the batch.
            prompt_len = enc["input_ids"].shape[1]
            prompt_lengths = [prompt_len] * len(batch_samples)

            for i, (out, gold) in enumerate(zip(outputs, gold_answers)):
                generated = self.tokenizer.decode(out[int(prompt_lengths[i]):], skip_special_tokens=True)
                pred = extract_answer(generated)
                if pred.strip() == gold.strip():
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        print(f"\n[AccuracyEval] step={state.global_step}  accuracy={accuracy:.4f}  ({correct}/{total})")

        # Log via trainer so metric appears in wandb
        if hasattr(self, "_trainer"):
            self._trainer.log({"eval_accuracy": accuracy})

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"[AccuracyEval] New best accuracy={accuracy:.4f} — saving to {self.best_dir}")
            model.save_pretrained(self.best_dir)
            self.tokenizer.save_pretrained(self.best_dir)
            # Write a small metadata file
            with open(os.path.join(self.best_dir, "best_info.json"), "w") as f:
                json.dump({"step": state.global_step, "accuracy": accuracy}, f, indent=2)

        model.train()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    train_dataset = PITDataset(args.train_file, tokenizer, max_length=args.max_length)
    eval_dataset = PITDataset(args.eval_file, tokenizer, max_length=args.max_length) if args.eval_file else None

    eval_steps = args.eval_steps

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"pit-sft-{args.model_name.split('/')[-1]}",
        dataloader_num_workers=2,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    accuracy_callback = None
    callbacks = []
    if eval_dataset is not None:
        accuracy_callback = AccuracyEvalCallback(
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            max_new_tokens=256,
            batch_size=args.batch_size,
            max_prompt_length=args.max_length,
        )
        callbacks.append(accuracy_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    if accuracy_callback is not None:
        accuracy_callback._trainer = trainer

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")

    if eval_dataset is not None:
        best_dir = os.path.join(args.output_dir, "best")
        if os.path.isdir(best_dir):
            print(f"Best model (by accuracy) is at {best_dir}")
        else:
            print("No best checkpoint saved — eval_file may not have been provided.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--train_file", default="dataset/pit_sft_dataset.jsonl")
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--output_dir", default="checkpoints/pit_sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    main(args)
