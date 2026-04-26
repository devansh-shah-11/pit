"""
SFT training on the noise-CoT dataset (dataset/noise_cot_train.jsonl).

Each sample has fields: question, answer, reasoning, type ("clean" | "adversarial").

--clean_only  : train on clean samples only (type == "clean")
--noise_only  : train on adversarial samples only (type == "adversarial")
(default)     : train on all samples (clean + adversarial)

Evaluation always loads all samples from the eval file and reports
overall / clean / adversarial accuracy breakdown.
"""

import json
import os
import sys
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from evaluation.evaluate_sft import build_prompt, extract_answer, answers_match


def causal_lm_collator(pad_token_id):
    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


class NoiseCOTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 768,
                 clean_only: bool = False, noise_only: bool = False,
                 shuffle: bool = True):
        if clean_only and noise_only:
            raise ValueError("clean_only and noise_only are mutually exclusive")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if clean_only and sample.get("type") != "clean":
                    continue
                if noise_only and sample.get("type") != "adversarial":
                    continue
                self.samples.append(sample)

        if shuffle:
            random.shuffle(self.samples)

        # Filter out samples whose full tokenized length exceeds max_length.
        # These would lose their answer label to truncation (answer sits at end of sequence).
        before = len(self.samples)
        self.samples = [
            s for s in self.samples
            if len(tokenizer(
                build_prompt(s["question"]) + "\n" + s["raw"] + tokenizer.eos_token,
                add_special_tokens=True
            )["input_ids"]) <= max_length
        ]
        subset = "clean-only" if clean_only else ("noise-only" if noise_only else "all")
        print(f"Loaded {len(self.samples)} training samples ({subset}) from {path} "
              f"({before - len(self.samples)} filtered for exceeding max_length={max_length})")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt = build_prompt(sample["question"])
        # "raw" is the full model output: reasoning + #### + answer.
        # Append EOS so the model learns to stop after the answer; otherwise
        # generation runs until max_new_tokens and emits multiple "####" blocks.
        completion = sample["raw"] + self.tokenizer.eos_token
        full_text = prompt + "\n" + completion

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # add_special_tokens=False avoids a double BOS when full text already has one.
        prompt_enc = self.tokenizer(
            prompt + "\n",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze(0).tolist()
        attention_mask = full_enc["attention_mask"].squeeze(0).tolist()

        # +1 accounts for the BOS token prepended by the full-text tokenization.
        prompt_len = min(prompt_enc["input_ids"].shape[1] + 1, len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class AccuracyEvalCallback(TrainerCallback):
    """
    After each evaluation step, runs greedy generation on the full eval set,
    reports overall / clean / adversarial exact-match accuracy, and saves the
    best checkpoint (by overall accuracy) to {output_dir}/best.

    use_vllm=True offloads generation to vLLM. On a single A100 40GB,
    gpu_memory_utilization=0.45 leaves ~18GB for the training model and ~18GB
    for vLLM. The vLLM engine is re-initialized each eval with fresh weights.
    """

    def __init__(self, eval_dataset, tokenizer, output_dir: str,
                 max_new_tokens: int = 256, batch_size: int = 8,
                 max_prompt_length: int = 1024, use_vllm: bool = False,
                 model_name: str = ""):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.best_dir = os.path.join(output_dir, "best")
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.max_prompt_length = max_prompt_length
        self.use_vllm = use_vllm
        self.model_name = model_name
        self.best_accuracy = -1.0
        self._vllm = None

    def _generate_transformers(self, prompts, model, device):
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

        prompt_len = enc["input_ids"].shape[1]
        return [self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in outputs]

    def _init_vllm(self, model_path):
        from vllm import LLM
        self._vllm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.25,
            max_model_len=self.max_prompt_length + self.max_new_tokens,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,
        )

    def _generate_vllm(self, prompts):
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0)
        results = self._vllm.generate(prompts, params)
        return [r.outputs[0].text for r in results]

    def on_evaluate(self, args, state, control, model, **kwargs):
        model.eval()
        device = next(model.parameters()).device

        if self.use_vllm:
            tmp_dir = os.path.join(self.output_dir, "_vllm_tmp")
            model.save_pretrained(tmp_dir)
            self.tokenizer.save_pretrained(tmp_dir)
            if self._vllm is not None:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
                del self._vllm
                self._vllm = None
            # Free the GPU so vLLM can allocate its own weights + KV cache.
            model.to("cpu")
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self._init_vllm(tmp_dir)

        stats = {
            "overall":     {"correct": 0, "total": 0},
            "clean":       {"correct": 0, "total": 0},
            "adversarial": {"correct": 0, "total": 0},
        }

        for start in range(0, len(self.eval_dataset), self.batch_size):
            batch_samples = [self.eval_dataset.samples[i]
                             for i in range(start, min(start + self.batch_size, len(self.eval_dataset)))]

            prompts      = [build_prompt(s["question"]) for s in batch_samples]
            gold_answers = [s["answer"] for s in batch_samples]
            # noise_cot data uses "type"; fall back to "source" for older formats
            types        = [s.get("type") or s.get("source", "clean") for s in batch_samples]

            if self.use_vllm:
                generated_texts = self._generate_vllm(prompts)
            else:
                generated_texts = self._generate_transformers(prompts, model, device)

            for gen, gold, sample_type in zip(generated_texts, gold_answers, types):
                is_correct = answers_match(extract_answer(gen), gold)
                stats["overall"]["total"] += 1
                stats["overall"]["correct"] += is_correct
                bucket = "adversarial" if sample_type == "adversarial" else "clean"
                stats[bucket]["total"] += 1
                stats[bucket]["correct"] += is_correct

        def acc(d):
            return d["correct"] / d["total"] if d["total"] > 0 else 0.0

        overall_acc = acc(stats["overall"])
        clean_acc   = acc(stats["clean"])
        adv_acc     = acc(stats["adversarial"])

        print(
            f"\n[AccuracyEval] step={state.global_step}\n"
            f"  Overall:     {overall_acc:.4f}  ({stats['overall']['correct']}/{stats['overall']['total']})\n"
            f"  Clean:       {clean_acc:.4f}  ({stats['clean']['correct']}/{stats['clean']['total']})\n"
            f"  Adversarial: {adv_acc:.4f}  ({stats['adversarial']['correct']}/{stats['adversarial']['total']})"
        )

        if hasattr(self, "_trainer"):
            self._trainer.log({
                "eval_accuracy":             overall_acc,
                "eval_accuracy_clean":       clean_acc,
                "eval_accuracy_adversarial": adv_acc,
                "train/global_step":         state.global_step,
            })

        if overall_acc > self.best_accuracy:
            self.best_accuracy = overall_acc
            print(f"[AccuracyEval] New best accuracy={overall_acc:.4f} — saving to {self.best_dir}")
            model.save_pretrained(self.best_dir)
            self.tokenizer.save_pretrained(self.best_dir)
            with open(os.path.join(self.best_dir, "best_info.json"), "w") as f:
                json.dump({
                    "step":                state.global_step,
                    "accuracy":            overall_acc,
                    "accuracy_clean":      clean_acc,
                    "accuracy_adversarial": adv_acc,
                }, f, indent=2)

        if self.use_vllm and self._vllm is not None:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            del self._vllm
            self._vllm = None
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            model.to(device)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        model.train()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )

    train_dataset = NoiseCOTDataset(
        args.train_file, tokenizer,
        max_length=args.max_length,
        clean_only=args.clean_only,
        noise_only=args.noise_only,
    )
    eval_dataset = NoiseCOTDataset(
        args.eval_file, tokenizer,
        max_length=args.max_length,
        clean_only=False,   # always eval on all samples
    ) if args.eval_file else None

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
        eval_steps=args.eval_steps if eval_dataset else None,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"pit-noise-cot-{'clean' if args.clean_only else ('noise' if args.noise_only else 'all')}-{args.model_name.split('/')[-1]}",
        dataloader_num_workers=2,
    )

    data_collator = causal_lm_collator(tokenizer.pad_token_id)

    accuracy_callback = None
    callbacks = []
    if eval_dataset is not None:
        accuracy_callback = AccuracyEvalCallback(
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            max_new_tokens=1024,
            batch_size=args.batch_size,
            max_prompt_length=args.max_length,
            use_vllm=args.use_vllm,
            model_name=args.model_name,
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
    parser.add_argument("--train_file", default="dataset/noise_cot_train.jsonl")
    parser.add_argument("--eval_file",  default="dataset/noise_cot_test.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/pit_noise_cot")
    parser.add_argument("--clean_only", action="store_true",
                        help="Train on clean samples only (type=='clean'); eval always uses all samples")
    parser.add_argument("--noise_only", action="store_true",
                        help="Train on adversarial samples only (type=='adversarial'); eval always uses all samples")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--grad_accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_length", type=int,   default=768)
    parser.add_argument("--eval_steps", type=int,   default=100)
    parser.add_argument("--use_wandb",  action="store_true")
    parser.add_argument("--use_vllm",   action="store_true")
    parser.add_argument("--cache_dir",  default=None,
                        help="Local dir to cache/load model weights (avoids re-downloading)")
    args = parser.parse_args()

    main(args)
