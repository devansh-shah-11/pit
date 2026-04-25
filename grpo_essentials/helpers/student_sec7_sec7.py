from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from collections import defaultdict


def run_compute_group_normalized_rewards_util(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses,
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    rewards = []
    reward_log = defaultdict(lambda: 0)
    for i, rollout_response in enumerate(rollout_responses):
        reward_outp = reward_fn(rollout_response, repeated_ground_truths[i])
        for key in reward_outp:
            reward_log[key] += reward_outp[key]
        reward = reward_outp['reward']
        rewards.append(reward)

    # mean_rwd = np.mean(rewards)
    # std_rwd = np.std(rewards)

    for key in reward_log:
        reward_log[key] = reward_log[key] / len(rewards)

    reward_tensor = torch.tensor(rewards)

    group_count = len(rollout_responses) // group_size
    grouped_reward_tensor = reward_tensor.view(group_count, group_size)

    mean_tensor = grouped_reward_tensor.mean(dim=1, keepdim=True)
    Advantage = grouped_reward_tensor - mean_tensor

    if normalize_by_std:
        group_stds = grouped_reward_tensor.std(dim=1, keepdim=True)
        Advantage = Advantage / (group_stds + advantage_eps)

    Advantage = Advantage.view(-1)


    return Advantage, reward_tensor, reward_log


def run_compute_naive_policy_gradient_loss_util(
        raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """

    npgl = - raw_rewards_or_advantages * policy_log_probs

    return npgl

def run_compute_grpo_clip_loss_util(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length):
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss
                (used to compute clip fraction).
    """
    #print("RUN COMPUTE GRPO CLI PLOSS UTIL", advantages)
    #print("POLICY LOG PROBS", policy_log_probs)
    #print("OLD LOG PROBS", old_log_probs)

    ratio = torch.exp(policy_log_probs - old_log_probs)
    #print("RATIO", ratio)


    res = torch.min(
        ratio * advantages, torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
    )
    clipped_mask = (ratio > 1 + cliprange) | (ratio < 1 - cliprange)

    clip_fraction = clipped_mask.float().mean()

    loss = -res

    return loss, {"clip_loss": res, "ratio": ratio, "clip_fraction": clip_fraction}


def run_compute_policy_gradient_loss_util(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """

    print("loss_type", loss_type)

    if loss_type == 'no_baseline':
        return run_compute_naive_policy_gradient_loss_util(
            raw_rewards,
            policy_log_probs
        ), {}
    elif loss_type == 'reinforce_with_baseline':
        return run_compute_naive_policy_gradient_loss_util(
            advantages,
            policy_log_probs
        ), {}
    else:
        return run_compute_grpo_clip_loss_util(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange
        )


def run_masked_mean_util(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """

    res = (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)

    return res









