import math
from typing import Any
import torch
from numpy.random import normal
from torch import Tensor

def run_log_softmax_util(
    in_features,
    dim: int
):

    max_val, _ = torch.max(in_features, dim=dim, keepdim=True)
    shifted = in_features - max_val

    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))

    return shifted - log_sum_exp


def run_softmax_util(in_features, dim: int):
    # taking in_features[dim], understand keepdim true meaning

    tensor_inp = in_features

    max_val, _ = torch.max(tensor_inp,dim=dim, keepdim=True)
    # # print("max val", max_val)

    tensor_inp = tensor_inp - max_val

    # exp_sm = torch.exp(tensor_inp).sum(dim=dim, keepdim=True)
    #
    # tensor_inp = torch.exp(tensor_inp) / exp_sm

    # return tensor_inp

    exp_vals = torch.exp(tensor_inp)
    exp_sum = exp_vals.sum(dim=dim, keepdim=True)
    return exp_vals / exp_sum