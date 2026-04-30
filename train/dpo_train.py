"""
DPO training on the PIT preference dataset (dataset/dpo_train.jsonl).

Each example is:
    {
      "prompt":   "<chat-formatted prompt ending with assistant header>",
      "question": "...",
      "answer":   "<gold final answer>",
      "chosen":   "<reference reasoning + #### + answer>",
      "rejected": "<SFT model's wrong generation>",
      "model_predicted_answer": "...",
      "type":     "adversarial" | "original"
    }

We train with TRL's DPOTrainer. After each eval we run greedy generation on a
held-out set and compute exact-match accuracy (overall / original / adversarial),
saving a `best/` checkpoint when overall accuracy improves. `final/` is always
written at the end.

Notes on the recipe:
  * Base / reference model = the SFT model that produced the rejected samples
    (jahyungu/Qwen2.5-1.5B-Instruct_gsm8k). DPO needs the rejected samples to
    have non-trivial reference logprob; using a different base would break that
    assumption.
  * `prompt` already contains the Qwen chat template up through the assistant
    header. We hand prompt/chosen/rejected to TRL as plain strings and disable
    its built-in chat templating so we don't double-wrap.
  * Chosen/rejected must end with EOS so the policy learns to stop. We append
    tokenizer.eos_token if it's missing.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import DPOConfig, DPOTrainer

from evaluation.evaluate_sft import build_chat_prompt, extract_answer, answers_match


def load_dpo_jsonl(path: str, tokenizer, max_examples=None, max_prompt_tokens=None):
    """Load PIT-style DPO jsonl into an HF Dataset of {prompt, chosen, rejected}."""
    eos = tokenizer.eos_token or ""
    rows = []
    dropped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            prompt   = ex["prompt"]
            chosen   = ex["chosen"]
            rejected = ex["rejected"]
            if eos and not chosen.endswith(eos):
                chosen = chosen + eos
            if eos and not rejected.endswith(eos):
                rejected = rejected + eos
            if max_prompt_tokens is not None:
                question = ex.get("question", "") or prompt
                chat_prompt = build_chat_prompt(question, tokenizer)
                n_tok = len(tokenizer(chat_prompt, add_special_tokens=True)["input_ids"])
                if n_tok > max_prompt_tokens:
                    dropped += 1
                    continue
            rows.append({
                "prompt":   prompt,
                "chosen":   chosen,
                "rejected": rejected,
                "type":     ex.get("type", "original"),
                "question": ex.get("question", ""),
                "answer":   ex.get("answer", ""),
            })
            if max_examples is not None and len(rows) >= max_examples:
                break
    if dropped:
        print(f"  Dropped {dropped} train pair(s) with prompt > {max_prompt_tokens} tokens.")
    return Dataset.from_list(rows)


def load_eval_questions(path: str, max_examples=None, tokenizer=None, max_prompt_tokens=None):
    """Eval set is plain SFT-style records: {question, answer, source}."""
    rows = []
    dropped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if max_prompt_tokens is not None and tokenizer is not None:
                chat_prompt = build_chat_prompt(ex["question"], tokenizer)
                n_tok = len(tokenizer(chat_prompt, add_special_tokens=True)["input_ids"])
                if n_tok > max_prompt_tokens:
                    dropped += 1
                    continue
            rows.append({
                "question": ex["question"],
                "answer":   ex["answer"],
                "source":   ex.get("source") or ex.get("type", "original"),
            })
            if max_examples is not None and len(rows) >= max_examples:
                break
    if dropped:
        print(f"  Dropped {dropped} eval question(s) with prompt > {max_prompt_tokens} tokens.")
    return rows


class AccuracyEvalCallback(TrainerCallback):
    """Greedy-decode the eval set and log exact-match accuracy per source."""

    def __init__(self, eval_questions, tokenizer, output_dir,
                 max_new_tokens=512, batch_size=8, max_prompt_length=1024,
                 use_vllm=False, model_name=""):
        self.eval_questions    = eval_questions
        self.tokenizer         = tokenizer
        self.output_dir        = output_dir
        self.best_dir          = os.path.join(output_dir, "best")
        self.max_new_tokens    = max_new_tokens
        self.batch_size        = batch_size
        self.max_prompt_length = max_prompt_length
        self.use_vllm          = use_vllm
        self.model_name        = model_name
        self.best_accuracy     = -1.0
        self._vllm             = None

    def _generate_transformers(self, prompts, model, device):
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_prompt_length,
        ).to(device)
        self.tokenizer.padding_side = "right"
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        return [self.tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in out]

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

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
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
            model.to("cpu")
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self._init_vllm(tmp_dir)

        stats = {
            "overall":     {"correct": 0, "total": 0},
            "original":    {"correct": 0, "total": 0},
            "adversarial": {"correct": 0, "total": 0},
        }

        for start in range(0, len(self.eval_questions), self.batch_size):
            batch = self.eval_questions[start:start + self.batch_size]
            prompts = [build_chat_prompt(s["question"], self.tokenizer) for s in batch]
            golds   = [s["answer"] for s in batch]
            sources = [s.get("source", "original") for s in batch]

            if self.use_vllm:
                gens = self._generate_vllm(prompts)
            else:
                gens = self._generate_transformers(prompts, model, device)

            for gen, gold, source in zip(gens, golds, sources):
                ok = answers_match(extract_answer(gen), gold)
                stats["overall"]["total"] += 1
                stats["overall"]["correct"] += ok
                key = "adversarial" if source == "adversarial" else "original"
                stats[key]["total"] += 1
                stats[key]["correct"] += ok

        def acc(d):
            return d["correct"] / d["total"] if d["total"] > 0 else 0.0

        overall = acc(stats["overall"])
        orig    = acc(stats["original"])
        adv     = acc(stats["adversarial"])

        print(
            f"\n[AccuracyEval] step={state.global_step}\n"
            f"  Overall:     {overall:.4f}  ({stats['overall']['correct']}/{stats['overall']['total']})\n"
            f"  Original:    {orig:.4f}  ({stats['original']['correct']}/{stats['original']['total']})\n"
            f"  Adversarial: {adv:.4f}  ({stats['adversarial']['correct']}/{stats['adversarial']['total']})"
        )

        # Inject into the metrics dict so Trainer's best-model tracking can see it
        # (e.g. metric_for_best_model="eval_accuracy" if you ever turn it on).
        if metrics is not None:
            metrics["eval_accuracy"]             = overall
            metrics["eval_accuracy_clean"]       = orig
            metrics["eval_accuracy_adversarial"] = adv

        if hasattr(self, "_trainer"):
            self._trainer.log({
                "eval_accuracy":             overall,
                "eval_accuracy_clean":       orig,
                "eval_accuracy_adversarial": adv,
                "train/global_step":         state.global_step,
            })

        if overall > self.best_accuracy:
            self.best_accuracy = overall
            print(f"[AccuracyEval] New best={overall:.4f} → saving {self.best_dir}")
            model.save_pretrained(self.best_dir)
            self.tokenizer.save_pretrained(self.best_dir)
            with open(os.path.join(self.best_dir, "best_info.json"), "w") as f:
                json.dump({
                    "step": state.global_step,
                    "accuracy":             overall,
                    "accuracy_original":    orig,
                    "accuracy_adversarial": adv,
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
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading policy model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )
    model.config.use_cache = False  # required for grad checkpointing

    # ref_model=None tells DPOTrainer to clone the policy at init and freeze it.
    ref_model = None

    print(f"Loading DPO train file: {args.train_file}")
    train_ds = load_dpo_jsonl(
        args.train_file, tokenizer,
        max_examples=args.max_train_examples,
        max_prompt_tokens=args.max_prompt_length,
    )
    print(f"  → {len(train_ds)} preference pairs")

    eval_questions = None
    if args.eval_file:
        print(f"Loading eval file (for accuracy): {args.eval_file}")
        eval_questions = load_eval_questions(
            args.eval_file,
            max_examples=args.max_eval_examples,
            tokenizer=tokenizer,
            max_prompt_tokens=args.max_prompt_length,
        )
        print(f"  → {len(eval_questions)} eval questions")

    # Tiny held-out slice of preference pairs gives us eval_loss / reward margins.
    eval_pref_ds = None
    if len(train_ds) >= 50:
        split = train_ds.train_test_split(test_size=min(64, len(train_ds) // 10), seed=args.seed)
        train_ds, eval_pref_ds = split["train"], split["test"]
        print(f"  pref train={len(train_ds)}  pref eval={len(eval_pref_ds)}")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="steps" if (eval_questions or eval_pref_ds) else "no",
        eval_steps=args.eval_steps,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"pit-dpo-{args.model_name.split('/')[-1]}",
        dataloader_num_workers=2,
        seed=args.seed,
        # ── DPO-specific ──
        beta=args.beta,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        max_length=args.max_length,
        truncation_mode="keep_start",
        remove_unused_columns=False,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    callbacks = []
    accuracy_callback = None
    if eval_questions:
        accuracy_callback = AccuracyEvalCallback(
            eval_questions=eval_questions,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            max_new_tokens=args.gen_max_new_tokens,
            batch_size=args.eval_gen_batch_size,
            max_prompt_length=args.max_prompt_length,
            use_vllm=args.use_vllm,
            model_name=args.model_name,
        )
        callbacks.append(accuracy_callback)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_pref_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    if accuracy_callback is not None:
        accuracy_callback._trainer = trainer

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")

    if accuracy_callback is not None:
        best = os.path.join(args.output_dir, "best")
        if os.path.isdir(best):
            print(f"Best model (by accuracy) is at {best}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k",
                   help="Should be the SFT model that produced the rejected generations.")
    p.add_argument("--train_file", default="dataset/dpo_train.jsonl")
    p.add_argument("--eval_file", default="dataset/test_data.jsonl",
                   help="SFT-style {question,answer,source} file for accuracy eval.")
    p.add_argument("--output_dir", default="checkpoints/pit_dpo")
    p.add_argument("--cache_dir", default=None)

    # ── core hyperparameters ──
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-7,
                   help="DPO is very sensitive to LR. 1e-7..1e-6 is the usual sweet spot.")
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO temperature. 0.1 = standard. Higher → stay closer to ref.")
    p.add_argument("--loss_type", default="sigmoid",
                   choices=["sigmoid", "ipo", "hinge", "robust",
                            "exo_pair", "nca_pair", "bco_pair"],
                   help="sigmoid = vanilla DPO. ipo = length-insensitive. "
                        "robust = label-noise robust (uses label_smoothing).")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="cDPO; 0.05–0.1 helps if rejected labels are noisy.")

    # ── lengths ──
    p.add_argument("--max_length", type=int, default=2048,
                   help="DPO training: total seq cap for prompt+chosen / prompt+rejected. "
                        "Truncation_mode=keep_start drops the completion tail when over budget.")
    p.add_argument("--max_prompt_length", type=int, default=1024,
                   help="Eval-time only: caps the prompt fed to model.generate / vLLM. "
                        "TRL ≥1.0 no longer has a separate prompt budget for training.")
    p.add_argument("--gen_max_new_tokens", type=int, default=512)

    # ── eval / logging ──
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--eval_gen_batch_size", type=int, default=8)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--use_vllm", action="store_true")

    # ── data limits / repro ──
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_eval_examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    main(args)
