"""
Selective-layer SFT on the noise-CoT dataset.

Identical to baseline_sft_noise_cot.py, except only a configurable subset of
transformer decoder layers is trainable. Everything else (embeddings, lm_head,
norm, and all other layers) is frozen.

Defaults to layers 18, 19, 20 — override via --train_layers "16,17,18".

--clean_only  : train on clean samples only (type == "clean")
(default)     : train on all samples (clean + adversarial)
"""

import json
import os
import sys
import random
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


def get_decoder_layers(model):
    """Locate the ModuleList of transformer decoder layers across common archs."""
    candidates = [
        ("model", "layers"),          # Llama, Qwen2, Mistral
        ("transformer", "h"),         # GPT-2, GPT-NeoX-style
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError(
        "Could not locate decoder layers ModuleList on this model. "
        "Add the path to get_decoder_layers()."
    )


def freeze_all_except_layers(model, train_layer_indices):
    """Freeze every parameter, then unfreeze only the requested decoder layers."""
    for p in model.parameters():
        p.requires_grad = False

    layers = get_decoder_layers(model)
    n_layers = len(layers)
    bad = [i for i in train_layer_indices if i < 0 or i >= n_layers]
    if bad:
        raise ValueError(
            f"Requested layer indices {bad} are out of range; model has {n_layers} layers (0..{n_layers - 1})."
        )

    for i in train_layer_indices:
        for p in layers[i].parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[selective-ft] model has {n_layers} decoder layers; "
        f"unfroze layers {sorted(train_layer_indices)}\n"
        f"[selective-ft] trainable params: {trainable:,} / {total:,} "
        f"({100.0 * trainable / total:.3f}%)"
    )


def parse_layer_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


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

    train_layer_indices = parse_layer_list(args.train_layers)
    freeze_all_except_layers(model, train_layer_indices)

    # Gradient checkpointing requires inputs to need grads when most params are frozen.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_dataset = NoiseCOTDataset(
        args.train_file, tokenizer,
        max_length=args.max_length,
        clean_only=args.clean_only,
    )
    eval_dataset = NoiseCOTDataset(
        args.eval_file, tokenizer,
        max_length=args.max_length,
        clean_only=False,
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
        run_name=f"pit-noise-cot-selective-L{'-'.join(str(i) for i in train_layer_indices)}-"
                 f"{'clean' if args.clean_only else 'all'}-{args.model_name.split('/')[-1]}",
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
    parser.add_argument("--output_dir", default="checkpoints/pit_noise_cot_selective")
    parser.add_argument("--train_layers", default="18,19,20",
                        help="Comma-separated decoder layer indices to unfreeze (e.g. '18,19,20')")
    parser.add_argument("--clean_only", action="store_true")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--grad_accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_length", type=int,   default=768)
    parser.add_argument("--eval_steps", type=int,   default=100)
    parser.add_argument("--use_wandb",  action="store_true")
    parser.add_argument("--use_vllm",   action="store_true")
    parser.add_argument("--cache_dir",  default=None)
    args = parser.parse_args()

    main(args)
