#!/usr/bin/env python3
"""
Generate DPO-style preference pairs (chosen / rejected chains-of-thought).

Chosen   = gold reasoning from train_data.jsonl (source=adversarial, paired
           with its preceding source=original entry for the answer + reasoning).
Rejected = the SFT model's own (wrong) generation on the same question.

For each adversarial question (and optionally clean questions, with
--include-clean-failures), we run greedy/sampled inference, extract the final
answer, and:
  - if the model is wrong  -> write a preference pair to --output
  - if the model is correct -> log it to --correct-output (no pair)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.evaluate_sft import (
    answers_match,
    build_chat_prompt,
    extract_answer,
)


def load_entries(path: Path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_existing_questions(path: Path) -> set:
    existing = set()
    if not path.exists():
        return existing
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                existing.add(rec.get("question", ""))
            except json.JSONDecodeError:
                pass
    return existing


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
    num_return_sequences: int,
    do_sample: bool,
    temperature: float,
):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_return_sequences,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    outputs = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for seq in outputs:
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        responses.append(text)
    return responses


def build_pair_record(
    question: str,
    answer: str,
    chosen_reasoning: str,
    rejected_response: str,
    predicted: Optional[str],
    sample_type: str,
    prompt_text: str,
) -> dict:
    chosen_text = f"{chosen_reasoning.strip()}\n#### {answer}"
    return {
        "prompt": prompt_text,
        "question": question,
        "answer": answer,
        "chosen": chosen_text,
        "rejected": rejected_response,
        "model_predicted_answer": predicted,
        "type": sample_type,
    }


def build_correct_record(
    question: str,
    answer: str,
    model_response: str,
    predicted: Optional[str],
    sample_type: str,
) -> dict:
    return {
        "question": question,
        "answer": answer,
        "model_response": model_response,
        "model_predicted_answer": predicted,
        "type": sample_type,
    }


def iter_targets(entries, include_clean_failures: bool):
    """
    Yield (question, answer, chosen_reasoning, sample_type) tuples.

    train_data.jsonl alternates source=original then several source=adversarial
    entries. We carry the most recent original's reasoning + answer as the gold
    for the adversarial questions that follow it.
    """
    cur_orig_question = None
    cur_orig_reasoning = None
    cur_orig_answer = None

    for entry in entries:
        source = entry.get("source")
        question = entry["question"]

        if source == "original":
            cur_orig_question = question
            cur_orig_reasoning = entry["reasoning"]
            cur_orig_answer = entry["answer"]
            if include_clean_failures:
                yield (question, cur_orig_answer, cur_orig_reasoning, "clean")
        elif source == "adversarial":
            if cur_orig_reasoning is None:
                continue
            yield (question, entry["answer"], cur_orig_reasoning, "adversarial")


def process_outputs(
    targets_chunk,
    prompts_chunk,
    responses_per_prompt,
    f_pair,
    f_correct,
    counters: dict,
):
    """Score responses for a chunk and append to the right files."""
    for (question, answer, chosen_reasoning, sample_type), prompt_text, responses in zip(
        targets_chunk, prompts_chunk, responses_per_prompt
    ):
        wrote_correct_for_q = False
        for resp in responses:
            pred = extract_answer(resp)
            if answers_match(pred, answer):
                if not wrote_correct_for_q:
                    rec = build_correct_record(question, answer, resp, pred, sample_type)
                    f_correct.write(json.dumps(rec) + "\n")
                    f_correct.flush()
                    counters["correct"] += 1
                    wrote_correct_for_q = True
            else:
                rec = build_pair_record(
                    question, answer, chosen_reasoning, resp, pred,
                    sample_type, prompt_text,
                )
                f_pair.write(json.dumps(rec) + "\n")
                f_pair.flush()
                counters["pairs"] += 1


def run_hf(args, targets, tokenizer, output_path, correct_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading HF model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    do_sample = args.num_rejected_per_q > 1
    counters = {"pairs": 0, "correct": 0}

    pbar = tqdm(targets, desc="DPO gen (HF)")
    with open(output_path, "a", encoding="utf-8") as f_pair, \
         open(correct_path, "a", encoding="utf-8") as f_correct:
        for target in pbar:
            question, answer, chosen_reasoning, sample_type = target
            pbar.set_postfix(pairs=counters["pairs"], correct=counters["correct"],
                             q=question[:40])
            prompt_text = build_chat_prompt(question, tokenizer)
            responses = generate_responses(
                model, tokenizer, prompt_text, device,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_rejected_per_q,
                do_sample=do_sample,
                temperature=args.temperature,
            )
            process_outputs([target], [prompt_text], [responses],
                            f_pair, f_correct, counters)

    return counters


def run_vllm(args, targets, tokenizer, output_path, correct_path):
    from vllm import LLM, SamplingParams

    print(f"Loading vLLM model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.vllm_gpu_mem_util,
        max_model_len=args.vllm_max_model_len,
    )

    do_sample = args.num_rejected_per_q > 1
    sampling_params = SamplingParams(
        n=args.num_rejected_per_q,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if do_sample else 0.0,
        top_p=0.95 if do_sample else 1.0,
    )

    counters = {"pairs": 0, "correct": 0}
    chunk_size = args.vllm_chunk_size
    failed_chunks = 0
    failed_prompts = 0

    pbar = tqdm(total=len(targets), desc="DPO gen (vLLM)")
    with open(output_path, "a", encoding="utf-8") as f_pair, \
         open(correct_path, "a", encoding="utf-8") as f_correct:
        for i in range(0, len(targets), chunk_size):
            chunk = targets[i : i + chunk_size]
            prompts = [build_chat_prompt(t[0], tokenizer) for t in chunk]

            try:
                outs = llm.generate(prompts, sampling_params, use_tqdm=False)
                responses_per_prompt = [
                    [o.text for o in out.outputs] for out in outs
                ]
                process_outputs(chunk, prompts, responses_per_prompt,
                                f_pair, f_correct, counters)
            except Exception as exc:
                failed_chunks += 1
                failed_prompts += len(chunk)
                print(f"\n[WARN] chunk starting at idx {i} failed ({len(chunk)} prompts): "
                      f"{type(exc).__name__}: {exc}. Skipping chunk.")

            pbar.update(len(chunk))
            pbar.set_postfix(pairs=counters["pairs"], correct=counters["correct"],
                             failed=failed_prompts)
    pbar.close()

    if failed_chunks:
        print(f"[WARN] {failed_chunks} chunk(s) ({failed_prompts} prompt(s)) skipped due to errors.")

    return counters


def generate(args):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    correct_path = Path(args.correct_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct_path.parent.mkdir(parents=True, exist_ok=True)

    seen_pair = load_existing_questions(output_path)
    seen_correct = load_existing_questions(correct_path)
    already_done = seen_pair | seen_correct
    if already_done:
        print(f"Resuming: {len(seen_pair)} pair(s) and {len(seen_correct)} correct entries already on disk.")

    entries = load_entries(input_path)
    targets = list(iter_targets(entries, args.include_clean_failures))

    start = args.start_from
    end = args.end_to if args.end_to is not None else len(targets)
    targets = targets[start:end]

    pre_filter_count = len(targets)
    targets = [t for t in targets if t[0] not in already_done]
    filtered_out = pre_filter_count - len(targets)
    if filtered_out:
        print(f"Filtered out {filtered_out} target(s) already present in output files.")

    if args.n_samples is not None:
        targets = targets[: args.n_samples]

    print(f"Targets to process: {len(targets)}")
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.max_prompt_tokens is not None:
        before = len(targets)
        kept = []
        for t in targets:
            prompt_text = build_chat_prompt(t[0], tokenizer)
            n_tok = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            if n_tok <= args.max_prompt_tokens:
                kept.append(t)
        dropped = before - len(kept)
        targets = kept
        if dropped:
            print(f"Dropped {dropped} target(s) whose prompt exceeded {args.max_prompt_tokens} tokens.")

    do_sample = args.num_rejected_per_q > 1
    print(
        f"Generation: num_rejected_per_q={args.num_rejected_per_q}  "
        f"do_sample={do_sample}  temperature={args.temperature}  "
        f"max_new_tokens={args.max_new_tokens}  use_vllm={args.use_vllm}"
    )

    if args.use_vllm:
        counters = run_vllm(args, targets, tokenizer, output_path, correct_path)
    else:
        counters = run_hf(args, targets, tokenizer, output_path, correct_path)

    print(f"\nDone. Pairs: {counters['pairs']}  Correct-logged: {counters['correct']}")
    print(f"Pairs written to  : {output_path}")
    print(f"Correct logged to : {correct_path}")


def main():
    p = argparse.ArgumentParser(description="Generate DPO preference pairs.")
    p.add_argument("--input", default="dataset/train_data.jsonl",
                   help="Source jsonl with source=original / source=adversarial entries.")
    p.add_argument("--output", default="dataset/dpo_train.jsonl",
                   help="Where to append preference pairs (model wrong).")
    p.add_argument("--correct-output", default="dataset/dpo_model_correct.jsonl",
                   help="Where to log questions the model already gets right.")
    p.add_argument("--model", default="jahyungu/Qwen2.5-1.5B-Instruct_gsm8k",
                   help="HF model id or local checkpoint path for the SFT model under test.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--num-rejected-per-q", type=int, default=1,
                   help="How many generations to attempt per question. >1 enables sampling.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature when num-rejected-per-q > 1.")
    p.add_argument("--include-clean-failures", action="store_true",
                   help="Also probe original (clean) questions; pair only when model is wrong.")
    p.add_argument("--start-from", type=int, default=0)
    p.add_argument("--end-to", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None,
                   help="Optional cap on number of target questions to process.")
    p.add_argument("--use-vllm", action="store_true",
                   help="Use vLLM for fast batched inference (requires vllm package).")
    p.add_argument("--vllm-chunk-size", type=int, default=256,
                   help="Number of prompts per vLLM batch (controls memory + flush cadence).")
    p.add_argument("--vllm-gpu-mem-util", type=float, default=0.9,
                   help="vLLM gpu_memory_utilization.")
    p.add_argument("--vllm-max-model-len", type=int, default=2048,
                   help="vLLM max_model_len (prompt + max_new_tokens must fit).")
    p.add_argument("--max-prompt-tokens", type=int, default=None,
                   help="Drop targets whose chat-templated prompt exceeds this token count. "
                        "Recommended: vllm_max_model_len - max_new_tokens.")
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()