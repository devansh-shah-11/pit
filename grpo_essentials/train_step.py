from typing import Callable, Literal

import numpy as np
import torch
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import helpers.student_util as utils
from grpo_essentials.helpers.student_sec4_run_experiment import load_policy_into_vllm_instance
# from student.sec_4.run_experiment import load_policy_into_vllm_instance, init_vllm, run_get_response_log_probs_util
from helpers.student_sec4_sec4 import run_tokenize_prompt_and_output_util, run_masked_normalize_util
from helpers.student_sec7_sec7 import run_compute_policy_gradient_loss_util, run_masked_mean_util, \
    run_compute_group_normalized_rewards_util
# from vllm import SamplingParams
from tqdm import tqdm
import argparse
import copy

from utils.defaults import DEVICE

TRAIN_DEVICE = DEVICE
VLLM_DEVICE = DEVICE #probably not going to be used

def run_grpo_microbatch_train_step_util(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        wandb=None,
        step_count=None,
        normalize_type = "masked_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_per_token, metadata = run_compute_policy_gradient_loss_util(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    if wandb is not None and step_count is not None:
        if "clip_fraction" in metadata:
            wandb.log({
                "train/clip_fraction": metadata['clip_fraction'],
            }, step=step_count)

    # loss = loss_per_token.mean()

    # print("loss initial 1 shape", loss_per_token.shape, loss_per_token)
    if normalize_type == "masked_mean":
        print("using masked mean")
        masked_loss = run_masked_mean_util(loss_per_token, response_mask, dim=1)
    elif normalize_type == "masked_normalize":
        print("using masked normalize")
        masked_loss = run_masked_normalize_util(loss_per_token, response_mask, dim=-1, normalize_constant=1024)
    else:
        raise Exception("Got normalize type that was not valid")

    # print("masked loss 1", masked_loss)
    masked_loss = masked_loss.mean()
    # print("masked loss 2", masked_loss)
    masked_loss = masked_loss / gradient_accumulation_steps
    # print("masked loss 3", masked_loss)

    masked_loss.backward()

    return masked_loss, metadata
    pass


def run_grpo_training(
        model_train,
        dataloader,
        tokenizer,
        optimizer,
        eval_vllm_model,
        eval_prompts,
        eval_gts,
        device=utils.DEVICE,
        eval_after=5,
        run_name=None,

        # GRPO PARAMETERS
        n_grpo_steps: int = 200,
        advantage_eps: float = 1e-6,
        rollout_batch_size: int = 16,
        group_size: int = 8,
        sampling_temperature: float = 0.7,
        sampling_min_tokens: int = 4,
        sampling_max_tokens: int = 1024,
        epochs_per_rollout_batch: int = 1,  # On-policy
        train_batch_size: int = 64,  # On-policy
        gradient_accumulation_steps: int = 128,
        loss_type: Literal[
            "no_baseline",
            "reinforce_with_baseline",
            "grpo_clip",
        ] = "reinforce_with_baseline",
        use_std_normalization: bool = True,
        grpo_clip=1.0,
        normalize_type="masked_mean",
):
    model_train.train()
    step_count = 0
    optimizer.zero_grad()

    train_iter = iter(dataloader)

    for step in range(n_grpo_steps):
        print("starting grpo step", step)
        questions_batch = next(train_iter)

        old_policy_model = copy.deepcopy(model_train).eval().to(device)
        for p in old_policy_model.parameters():
            p.requires_grad = False

        if USE_VLLM:
            load_policy_into_vllm_instance(old_policy_model, eval_vllm_model)

        rollout_responses = []
        repeated_ground_truths = []

        sampling_params = SamplingParams(
            temperature=sampling_temperature,
            min_tokens=sampling_min_tokens,
            max_tokens=sampling_max_tokens,
            stop=["</answer>"],
        )

        if USE_VLLM:
            load_policy_into_vllm_instance(model_train, eval_vllm_model)
            for question, gt in zip(questions_batch["prompts"], questions_batch["ground_truths"]):
                for _ in range(group_size):
                    outputs = eval_vllm_model.generate(question, sampling_params=sampling_params)
                    response = outputs[0].outputs[0].text
                    print("this prompt:", question)
                    print("this response:", response)
                    rollout_responses.append(response)
                    repeated_ground_truths.append(gt)

        print("got rollouts", len(rollout_responses), len(repeated_ground_truths))
        print("repeated gts:", repeated_ground_truths)
        repeated_prompts = []
        for question in questions_batch["prompts"]:
            for _ in range(group_size):
                repeated_prompts.append(question)

