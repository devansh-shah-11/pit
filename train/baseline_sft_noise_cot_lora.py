"""
LoRA SFT on the noise-CoT dataset (dataset/noise_cot_train.jsonl).

Reuses NoiseCOTDataset + AccuracyEvalCallback from baseline_sft_noise_cot.py.
Trains on the full mixture (clean + adversarial) by default — the goal of this
experiment is to keep clean accuracy strong (LoRA touches a small parameter
subspace, leaving the base model largely intact) while lifting adversarial
accuracy.

Three preset configs (--config):

  small         r=8,  alpha=16, attention only (q,v),         lr=2e-4
                Smallest footprint. Strongest preservation of base behavior;
                a sanity-check that LoRA can move adv accuracy at all.

  balanced      r=16, alpha=32, attn + MLP (q,k,v,o,gate,up,down),
                lr=1e-4, dropout=0.05
                Default. Enough capacity to learn noise rejection across
                attention and MLP, with dropout to limit overfitting to
                adversarial templates.

  high_capacity r=32, alpha=64, all linear layers, lr=5e-5, dropout=0.1
                Closest to full-FT capacity. Lower LR + higher dropout to
                avoid clobbering clean math reasoning.

Any preset field can be overridden via CLI flags (--lora_r, --lora_alpha, ...).
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from train.baseline_sft_noise_cot import (
    NoiseCOTDataset,
    AccuracyEvalCallback,
    causal_lm_collator,
)


class LoRAAccuracyEvalCallback(AccuracyEvalCallback):
    """
    AccuracyEvalCallback variant for PEFT/LoRA models. The parent writes the
    live model with `save_pretrained` before spinning up vLLM, but on a PEFT
    model that only emits adapter weights — vLLM then fails because there is
    no config.json. Here we merge the adapter into a copy of the base model
    and save the full merged model to `_vllm_tmp`, then defer to the parent
    for the rest of the eval flow. The 'best' / final adapter saves remain
    adapter-only (handled by the parent path / main()).
    """

    def on_evaluate(self, args, state, control, model, **kwargs):
        if not self.use_vllm:
            return super().on_evaluate(args, state, control, model, **kwargs)

        # Reversibly merge the LoRA delta into the base weights so the saved
        # checkpoint has a config.json + full base+delta tensors that vLLM
        # can load. After eval we unmerge so training resumes with the
        # adapter parameters intact and trainable.
        tmp_dir = os.path.join(self.output_dir, "_vllm_tmp")
        model.merge_adapter()
        try:
            base = model.get_base_model()
            base.save_pretrained(tmp_dir)
            self.tokenizer.save_pretrained(tmp_dir)
        finally:
            model.unmerge_adapter()

        # Prevent the parent's save_pretrained call from overwriting our
        # merged dir with adapter-only weights.
        original_save = model.save_pretrained
        model.save_pretrained = lambda *a, **kw: None
        try:
            super().on_evaluate(args, state, control, model, **kwargs)
        finally:
            model.save_pretrained = original_save


LORA_PRESETS = {
    "small": {
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "lr": 2e-4,
    },
    "balanced": {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "lr": 1e-4,
    },
    "high_capacity": {
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "lr": 5e-5,
    },
}


def apply_preset(args):
    preset = LORA_PRESETS[args.config]
    for k, v in preset.items():
        # only fill if user did not override on the CLI
        if getattr(args, k, None) in (None, []):
            setattr(args, k, v)
    return args


def main(args):
    args = apply_preset(args)

    from peft import LoraConfig, get_peft_model, TaskType

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

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = NoiseCOTDataset(
        args.train_file, tokenizer,
        max_length=args.max_length,
        clean_only=args.clean_only,
        noise_only=args.noise_only,
    )
    eval_dataset = NoiseCOTDataset(
        args.eval_file, tokenizer,
        max_length=args.max_length,
        clean_only=False,
    ) if args.eval_file else None

    run_subset = "clean" if args.clean_only else ("noise" if args.noise_only else "all")
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
        run_name=f"pit-lora-{args.config}-{run_subset}-{args.model_name.split('/')[-1]}",
        dataloader_num_workers=2,
    )

    data_collator = causal_lm_collator(tokenizer.pad_token_id)

    callbacks = []
    accuracy_callback = None
    if eval_dataset is not None:
        accuracy_callback = LoRAAccuracyEvalCallback(
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
    # PEFT save_pretrained writes only the adapter weights.
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final LoRA adapter saved to {final_dir}")

    if args.merge_and_save:
        merged_dir = os.path.join(args.output_dir, "final_merged")
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged full model saved to {merged_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(LORA_PRESETS.keys()), default="balanced",
                        help="LoRA preset; individual fields can still be overridden below")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--train_file", default="dataset/noise_cot_train.jsonl")
    parser.add_argument("--eval_file",  default="dataset/noise_cot_test.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/pit_noise_cot_lora")
    parser.add_argument("--clean_only", action="store_true")
    parser.add_argument("--noise_only", action="store_true")

    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--grad_accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=None,
                        help="Overrides the preset LR if set")
    parser.add_argument("--max_length", type=int,   default=768)
    parser.add_argument("--eval_steps", type=int,   default=100)

    parser.add_argument("--lora_r",       type=int,   default=None)
    parser.add_argument("--lora_alpha",   type=int,   default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_modules", nargs="+", default=None,
                        help="Override the preset target_modules list")

    parser.add_argument("--merge_and_save", action="store_true",
                        help="Also save a merged (base+adapter) full model for vLLM eval")
    parser.add_argument("--use_wandb",  action="store_true")
    parser.add_argument("--use_vllm",   action="store_true")
    parser.add_argument("--cache_dir",  default=None)
    args = parser.parse_args()

    main(args)
