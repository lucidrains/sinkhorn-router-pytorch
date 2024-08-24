from __future__ import annotations
from contextlib import nullcontext

from typing import List

import torch
from torch import Tensor, nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

import einx
from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# sinkhorn related functions

def log(t, eps = 1e-6):
    return torch.log(t.clamp(min = eps))

def gumbel_like(t, eps = 1e-6):
    noise = torch.rand_like(t)
    return -log(-log(noise, eps), eps)

def sinkhorn(
    t,
    num_iters = 8,
    gumbel = False,
    temperature = 1.,
    eps = 1e-6
):
    t = log(t, eps)

    assert temperature > 0.
    t = t / temperature

    if gumbel:
        t = t + gumbel_like(t, eps)

    for _ in range(num_iters):
        t = t - t.logsumexp(dim = -2, keepdim = True)
        t = t - t.logsumexp(dim = -1, keepdim = True)

    return t.exp()

# main module

class SinkhornRouter(Module):
    def __init__(
        self,
        dim,
        experts: ModuleList | List[Module] | Tensor,
        causal = False,
        gumbel_noise = False,
        temperature = 1.,
        competitive_gates: bool | None = None
    ):
        super().__init__()

        # only use competitive gates for non-causal by default

        competitive_gates = default(competitive_gates, not causal)
        assert not (causal and competitive_gates), 'causal sequences cannot have competitive gates'
        self.competitive_gates = competitive_gates

        # experts are a ModuleList where length is number of experts
        # if a Tensor is given, it must be in shape of (experts, dim_in, dim_out)

        if isinstance(experts, list):
            experts = ModuleList(experts)

        self.experts = experts
        self.num_experts = len(experts)

        # gating and sinkhorn related

        self.to_gates = nn.Linear(dim, self.num_experts, bias = False)

        self.temperature = temperature
        self.gumbel_noise = gumbel_noise

    def forward(
        self,
        x,
        mask = None
    ):
        seq_len = x.shape[-2]
        assert divisible_by(seq_len, self.num_experts)

        gates = self.to_gates(x)

        # masking for variable sequence lengths

        if exists(mask):
            gates = einx.where('b n, b n d, -> b n d', mask, gates, -torch.finfo(gates.dtype).max)

        # if non-competitive, do not differentiate through sinkhorn, technique came from megatron, afaict
        # they then just select the sigmoid of the selected gate, which should work given recent papers (Sigmoid MoE etc)

        sinkhorn_context = nullcontext if self.competitive_gates else torch.no_grad

        with sinkhorn_context():
            gates = sinkhorn(
                gates,
                temperature = self.temperature,
                gumbel = self.gumbel_noise
            )

        return gates
