"""
Interactive REPL for a trained (or base) SFT model.

Usage:
    python evaluation/interactive_chat.py --model_path checkpoints/pit_noise_cot_noise_only/best

Type a question at the prompt; the model will answer using the same
build_prompt() format used for training/eval. Type :quit to exit.
Commands:
    :quit / :q          exit
    :raw                toggle raw-prompt mode (send your text verbatim, no build_prompt wrapping)
    :tokens N           change max_new_tokens
    :temp T             change sampling temperature (0 = greedy)
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.evaluate_sft import build_chat_prompt, extract_answer


def _stop_token_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end != tokenizer.unk_token_id:
        ids.add(im_end)
    return [i for i in ids if i is not None]


def generate(model, tokenizer, prompt, max_new_tokens, temperature, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=_stop_token_ids(tokenizer),
            repetition_penalty=1.3,
        )
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to model dir (local checkpoint) or HF repo id")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy decoding")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu / mps; default = auto")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"Loading model from: {args.model_path}")
    print(f"Device: {device}  dtype: {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()

    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    raw_mode = False

    print("\nReady. Type a math question (or :quit to exit, :help for commands).")
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue

        if user_input in (":q", ":quit", ":exit"):
            break
        if user_input == ":help":
            print(":quit | :raw | :tokens N | :temp T")
            continue
        if user_input == ":raw":
            raw_mode = not raw_mode
            print(f"raw mode = {raw_mode}")
            continue
        if user_input.startswith(":tokens "):
            try:
                max_new_tokens = int(user_input.split()[1])
                print(f"max_new_tokens = {max_new_tokens}")
            except (IndexError, ValueError):
                print("usage: :tokens N")
            continue
        if user_input.startswith(":temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"temperature = {temperature}")
            except (IndexError, ValueError):
                print("usage: :temp T")
            continue

        prompt = user_input if raw_mode else build_chat_prompt(user_input, tokenizer)
        completion = generate(model, tokenizer, prompt,
                              max_new_tokens, temperature, device)

        print("\n--- model output ---")
        print(completion.strip())
        ans = extract_answer(completion)
        if ans is not None:
            print(f"\n[parsed answer] {ans}")


if __name__ == "__main__":
    main()
