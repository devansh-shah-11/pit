import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from torch.nn.utils.rnn import pad_sequence
import student_util as utils


# TODO: UNDERSTAND THIS ONE PROPERLY
# TODO: change normalization

def run_tokenize_prompt_and_output_util(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    input_ids = []
    # labels = []
    masks = []
    max_len = 0

    for i in range(len(prompt_strs)):
        prompt_str = prompt_strs[i]
        output_str = output_strs[i]

        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_ids = tokenizer.encode(output_str, add_special_tokens=False)

        print("lengths: prompt", len(prompt_ids), "outpout", len(output_ids), "total",
              len(prompt_ids) + len(output_ids))

        prompt_output_ids = prompt_ids + output_ids

        response_mask = [0] * len(prompt_ids) + [1] * len(output_ids)

        input_ids.append(torch.tensor(prompt_output_ids))
        masks.append(torch.tensor(response_mask))

    pad_id = None
    if pad_id is None:
        pad_id = 128001  # that happens to be the pad id value
    print("PAD ID", pad_id)

    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=pad_id
    )

    response_mask_padded = pad_sequence(
        masks, batch_first=True, padding_value=0
    )

    input_ids = input_ids_padded[:, :-1]
    labels = input_ids_padded[:, 1:]
    response_mask = response_mask_padded[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }


def run_compute_entropy_util(logits: torch.Tensor):
    # log_probs = utils.run_log_softmax_util(logits, -1)
    # probs = utils.run_softmax_util(logits, -1)

    # res = probs * log_probs

    # res = -torch.sum(res, dim=-1)

    # return res
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # Use the identity: H = -sum(p * log_p) = -sum(exp(log_p) * log_p)
    return -(torch.exp(log_probs) * log_probs).sum(dim=-1)


def run_get_response_log_probs_util(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool,
):
    res = model(input_ids).logits  # (B, L, V)

    log_probs = utils.run_log_softmax_util(res, -1)  # outp shape, (B, L, V)

    selected_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, L, 1).squeeze = (B, L)

    final = {"log_probs": selected_log_probs}

    if return_token_entropy:
        token_entropy = run_compute_entropy_util(res)  # per token entropy (B, L)
        final['token_entropy'] = token_entropy

    return final


def run_masked_normalize_util(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
        normalize_constant: float = 1.0,
) -> torch.Tensor:
    res = (tensor * mask).sum(dim=dim) / normalize_constant

    return res


def run_sft_microbatch_train_step_util(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    print("policy log probs", policy_log_probs.shape)
    print("response mask", response_mask.shape)

    loss = -run_masked_normalize_util(policy_log_probs, response_mask, -1,
                                      normalize_constant) / gradient_accumulation_steps

    print("shape of loss", loss.shape)

    loss = loss.mean()

    loss.backward()

    return loss, {
        "loss": loss
    }


