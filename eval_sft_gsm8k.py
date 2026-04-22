"""
Evaluate base model and best checkpoint on gsm8k_processed_test_sample.json.
Reports exact-match accuracy for each model.
"""

import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_TEMPLATE = """Solve this math problem step by step:

{question}

Provide your final answer in the format:
[reasoning steps]
####
[final answer (just the number)]"""


def extract_answer(text: str) -> str:
    parts = text.split("####")
    if len(parts) < 2:
        return ""
    for line in parts[-1].splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def load_samples(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_vllm(llm, prompts, max_new_tokens):
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    results = llm.generate(prompts, params)
    return [r.outputs[0].text for r in results]


def generate_transformers(model, tokenizer, prompts, device, max_new_tokens, max_prompt_length):
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(device)
    tokenizer.padding_side = "right"
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    return [tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in outputs]


def evaluate_model(model_path, samples, args, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.use_vllm:
        from vllm import LLM
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_prompt_length + args.max_new_tokens,
            tensor_parallel_size=1,
        )
        model = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        llm = None

    correct, total = 0, 0
    results = []

    for start in range(0, len(samples), args.batch_size):
        batch = samples[start: start + args.batch_size]
        prompts      = [PROMPT_TEMPLATE.format(question=s["question"]) for s in batch]
        gold_answers = [s["answer"] for s in batch]

        if args.use_vllm:
            generated = generate_vllm(llm, prompts, args.max_new_tokens)
        else:
            generated = generate_transformers(model, tokenizer, prompts, device, args.max_new_tokens, args.max_prompt_length)

        for gen, gold, sample in zip(generated, gold_answers, batch):
            pred = extract_answer(gen).strip()
            is_correct = pred == gold.strip()
            correct += is_correct
            total += 1
            results.append({"question": sample["question"], "gold": gold, "pred": pred, "correct": is_correct})

        print(f"  [{min(start + args.batch_size, total)}/{len(samples)}] correct so far: {correct}", flush=True)

    if args.use_vllm:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm
        import gc; gc.collect()
        torch.cuda.empty_cache()

    return correct / total if total > 0 else 0.0, correct, total, results


def main(args):
    samples = load_samples(args.eval_file)
    print(f"Loaded {len(samples)} samples from {args.eval_file}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Evaluating BASE MODEL: {args.base_model} ===")
    base_acc, base_correct, base_total, base_results = evaluate_model(args.base_model, samples, args, device)

    print(f"\n=== Evaluating BEST CHECKPOINT: {args.best_model} ===")
    best_acc, best_correct, best_total, best_results = evaluate_model(args.best_model, samples, args, device)

    print("\n" + "=" * 55)
    print(f"{'Model':<35} {'Accuracy':>10}  {'Correct/Total':>15}")
    print("-" * 55)
    print(f"{'Base  ' + args.base_model.split('/')[-1]:<35} {base_acc:>10.4f}  {base_correct:>6}/{base_total}")
    print(f"{'Best  ' + args.best_model.split('/')[-1]:<35} {best_acc:>10.4f}  {best_correct:>6}/{best_total}")
    print("=" * 55)

    if args.output_file:
        out = {
            "eval_file": args.eval_file,
            "base_model":  {"path": args.base_model,  "accuracy": base_acc,  "correct": base_correct,  "total": base_total,  "results": base_results},
            "best_model":  {"path": args.best_model,  "accuracy": best_acc,  "correct": best_correct,  "total": best_total,  "results": best_results},
        }
        with open(args.output_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  required=True)
    parser.add_argument("--best_model",  required=True)
    parser.add_argument("--eval_file",   required=True)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--use_vllm",   action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()
    main(args)
