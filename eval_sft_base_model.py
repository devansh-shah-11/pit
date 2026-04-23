"""
Evaluate a model (base or fine-tuned) on test data.
Reports clean (original), adversarial, and overall exact-match accuracy.
"""

import re
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


def build_chat_prompt(question: str, tokenizer) -> str:
    """Wrap PROMPT_TEMPLATE in the model's chat format so instruct models respond correctly."""
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(text: str) -> str:
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
        numbers = re.findall(r"-?\d+\.?\d*", answer_part)
        if numbers:
            return numbers[0]
    # fallback: last number anywhere in the output
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


def answers_match(pred: str, gold: str) -> bool:
    pred = pred.strip().replace(",", "").replace(" ", "")
    gold = gold.strip().replace(",", "").replace(" ", "")
    if pred == gold:
        return True
    try:
        return float(pred) == float(gold)
    except ValueError:
        return False


def load_samples(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def generate_transformers(prompts, model, tokenizer, device, max_new_tokens, max_prompt_length):
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


def generate_vllm(prompts, llm, max_new_tokens, tokenizer=None, max_prompt_length=None):
    from vllm import SamplingParams
    
    # Truncate prompts if tokenizer and max_prompt_length are provided
    if tokenizer and max_prompt_length:
        truncated_prompts = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt, truncation=False)
            if len(tokens) > max_prompt_length:
                # Decode back to text with truncation
                truncated_text = tokenizer.decode(tokens[:max_prompt_length], skip_special_tokens=False)
                truncated_prompts.append(truncated_text)
            else:
                truncated_prompts.append(prompt)
        prompts = truncated_prompts
    
    params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    results = llm.generate(prompts, params)
    return [r.outputs[0].text for r in results]


def evaluate(args):
    samples = load_samples(args.eval_file)
    print(f"Loaded {len(samples)} samples from {args.eval_file}")

    stats = {
        "overall":     {"correct": 0, "total": 0},
        "original":    {"correct": 0, "total": 0},
        "adversarial": {"correct": 0, "total": 0},
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.use_vllm:
        from vllm import LLM
        llm = LLM(
            model=args.model_name,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_prompt_length + args.max_new_tokens,
            tensor_parallel_size=1,
        )
        model, device = None, None
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
        ).to(device)
        model.eval()
        llm = None

    results = []

    for start in range(0, len(samples), args.batch_size):
        batch = samples[start: start + args.batch_size]
        prompts      = [build_chat_prompt(s["question"], tokenizer) for s in batch]
        gold_answers = [s["answer"] for s in batch]
        sources      = [s.get("source", "original") for s in batch]

        if args.use_vllm:
            generated = generate_vllm(prompts, llm, args.max_new_tokens, tokenizer, args.max_prompt_length)
        else:
            generated = generate_transformers(
                prompts, model, tokenizer, device,
                args.max_new_tokens, args.max_prompt_length,
            )

        for gen, gold, source, sample in zip(generated, gold_answers, sources, batch):
            pred = extract_answer(gen)
            is_correct = answers_match(pred, gold)

            stats["overall"]["total"] += 1
            stats["overall"]["correct"] += is_correct
            src_key = "adversarial" if source == "adversarial" else "original"
            stats[src_key]["total"] += 1
            stats[src_key]["correct"] += is_correct

            results.append({
                "question": sample["question"],
                "source": source,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "generated": gen,
            })

        done = min(start + args.batch_size, len(samples))
        print(f"  [{done}/{len(samples)}] running overall={stats['overall']['correct']}/{stats['overall']['total']}", flush=True)

    def acc(d):
        return d["correct"] / d["total"] if d["total"] > 0 else 0.0

    overall_acc = acc(stats["overall"])
    orig_acc    = acc(stats["original"])
    adv_acc     = acc(stats["adversarial"])

    print("\n" + "=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Eval file: {args.eval_file}")
    print("=" * 50)
    print(f"Overall accuracy:     {overall_acc:.4f}  ({stats['overall']['correct']}/{stats['overall']['total']})")
    print(f"Clean accuracy:       {orig_acc:.4f}  ({stats['original']['correct']}/{stats['original']['total']})")
    print(f"Adversarial accuracy: {adv_acc:.4f}  ({stats['adversarial']['correct']}/{stats['adversarial']['total']})")
    print("=" * 50)

    if args.output_file:
        summary = {
            "model": args.model_name,
            "eval_file": args.eval_file,
            "accuracy_overall":     overall_acc,
            "accuracy_clean":       orig_acc,
            "accuracy_adversarial": adv_acc,
            "counts": stats,
        }
        with open(args.output_file, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",   required=True,  help="HF model name or local path")
    parser.add_argument("--eval_file",    required=True,  help="Path to test JSONL")
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--use_vllm",    action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--cache_dir",   default=None)
    parser.add_argument("--output_file", default=None, help="Save JSON results here")
    args = parser.parse_args()

    evaluate(args)
