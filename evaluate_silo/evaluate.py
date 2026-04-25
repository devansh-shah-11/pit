"""Minimal evaluation script for MATH and Intellect test sets."""
import logging
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from dr_grpo_grader import question_only_reward_fn


def setup_logger(log_path: str = "eval.log") -> logging.Logger:
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)

    # File handler — writes everything to disk
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    logger.addHandler(fh)
    return logger


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def evaluate(llm, prompts, ground_truths,n_examples=3,sampling_temperature=0.0,sampling_max_tokens=2048,sampling_min_tokens=0,stop_tokens=None,reward_fn=question_only_reward_fn, verbose=False):
    """Run evaluation and return accuracy."""

    res = defaultdict(lambda: 0)
    categories = defaultdict(lambda: 0)

    params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=stop_tokens if stop_tokens is not None else []
    )
    outputs = llm.generate(prompts, params)

    correct = 0
    cases_format_0 = []
    cases_format_1_ans_0 = []

    example_rollouts = []

    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        reward = reward_fn(text, ground_truths[i])

        for key in reward:
            res[key] += reward[key]

        correct += reward["reward"]

        if verbose:
            print("accuracy so far:", correct, i + 1, correct / (i + 1))

        if i < n_examples:
            example_rollouts.append({
                "prompt": prompts[i],
                "output": text,
                "ground_truth": ground_truths[i],
                "reward": reward,
            })


        if reward['format_reward'] == 0:
            cases_format_0.append({
                "prompt": prompts[i],
                "output": text,
                "ground_truth": ground_truths[i],
                "reward": reward,
            })

        if reward['format_reward'] == 1 and reward['answer_reward'] == 0:
            cases_format_1_ans_0.append({
                "prompt": prompts[i],
                "output": text,
                "ground_truth": ground_truths[i],
                "reward": reward,
            })


        if reward['format_reward'] == 1 and reward['answer_reward'] == 1:
            categories['CAT (1)'] += 1
        elif reward['format_reward'] == 1 and reward['answer_reward'] == 0:
            categories['CAT (2)'] += 1
        elif reward['format_reward'] == 0 and reward['answer_reward'] == 0:
            categories['CAT (3)'] += 1

        # print(f"PROMPT: {prompts[i][:25]} | OUTPUT: {text[:25]} | REWARD: {reward} | REWARD SO FAR: {dict(res)}")


    print("CATEGORIES: ", categories)

    #print("CASES FORMAT REWARD 0: ", cases_format_0[:10])

    #print("CASES FORMAT REWARD 1 ANSWER REWARD 0:", cases_format_1_ans_0[:10])
    for key in res:
        res[key] = res[key] / len(outputs)

    if example_rollouts:
        print("\n=== Example Rollouts ===")
        for ex in example_rollouts:
            print(f"PROMPT: {ex['prompt']} ...")
            print(f"OUTPUT: {ex['output'][-50:]} ...")
            #print(f"GROUND TRUTH: {ex['ground_truth']} ...")
            print(f"REWARD: {ex['reward']}\n")

    return correct / len(outputs), res


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--intellect-path", default="student/data/intellect_math/test")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--log-file", default="eval.log")

    args = parser.parse_args()

    logger = setup_logger(args.log_file)


    prompt_template = load_prompt("intellect")

    # Load model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Evaluate on Intellect test
    # print(f"\n=== Intellect Test ({args.intellect_path}) ===")
    # dataset = load_from_disk(args.intellect_path)
    # if args.max_examples:
    #     dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    #
    # prompts, gts = [], []
    # for ex in dataset:
    #     msgs = ex.get("messages", [])
    #     sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    #     user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    #     prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
    #     gts.append(ex.get("ground_truth", ""))
    #
    # print(f"[Sample] {prompts[0][:200]}...")
    # acc, _ = evaluate(llm, prompts, gts)
    # print(f"Intellect Accuracy: {acc:.4f}")

    # Evaluate on MATH
    logger.info("=== MATH Test ===")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]

    print("[Sample] ", prompts[0][:200])
    acc, _ = evaluate(llm, prompts, gts)
    print("MATH Accuracy:", acc)


if __name__ == "__main__":
    main()
