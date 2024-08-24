from __future__ import annotations
from contextlib import nullcontext

from typing import List

import torch
from torch import Tensor, nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

import einx
from einops import rearrange, einsum

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
        sinkhorn_iters = 8,
        heads = 1,
        temperature = 1.,
        competitive: bool | None = None
    ):
        super().__init__()

        self.heads = heads

        # only use competitive gates for non-causal by default

        competitive = default(competitive, not causal)
        assert not (causal and competitive), 'causal sequences cannot have competitive gates'
        self.competitive = competitive

        # experts are a ModuleList where length is number of experts
        # if a Tensor is given, it must be in shape of (experts, [optional] heads, dim_in, dim_out)

        if heads > 1:
            assert torch.is_tensor(experts) and experts.ndim == 4 and experts.shape[1] == heads

        if isinstance(experts, list):
            experts = ModuleList(experts)

        self.experts = experts
        num_experts = len(experts)
        self.num_experts = num_experts

        # gating and sinkhorn related

        self.to_gate_weight = nn.Parameter(torch.randn(heads, dim, num_experts))

        self.temperature = temperature
        self.gumbel_noise = gumbel_noise
        self.sinkhorn_iters = sinkhorn_iters

    def forward(
        self,
        x,
        mask = None
    ):
        """
        ein notation:
        b - batch
        h - heads
        n - sequence length
        e - experts
        m - tokens per expert
        d, i, o - feature dimension (input and output dimension)
        """

        seq_len, single_headed = x.shape[-2], x.ndim == 3
        assert divisible_by(seq_len, self.num_experts)
        tokens_per_expert = seq_len // self.num_experts

        # handle single headed

        if single_headed:
            x = rearrange(x, 'b n d -> b 1 n d')

        assert x.shape[1] == self.heads

        # project to gates

        gate_logits = einsum(x, self.to_gate_weight, 'b h n d, h d e -> b h n e')

        # masking for variable sequence lengths

        if exists(mask):
            gate_logits = einx.where('b n, b h n e, -> b h n e', mask, gate_logits, -torch.finfo(gates.dtype).max)

        # sinkhorn ensures balanced routing
        # if non-competitive, do not differentiate through sinkhorn, technique came from megatron, afaict
        # they then just select the sigmoid of the selected gate, which should work given recent papers (Sigmoid MoE etc)

        sinkhorn_context = nullcontext if self.competitive else torch.no_grad

        with sinkhorn_context():

            competitive_gates = sinkhorn(
                gate_logits,
                temperature = self.temperature,
                gumbel = self.gumbel_noise,
                num_iters = self.sinkhorn_iters
            )

            gate_values, routed_indices = competitive_gates.topk(tokens_per_expert, dim = -2)

        if not self.competitive:
            selected_gate_logits = einx.get_at('... [n] e, ... m e -> ... m e', gate_logits, routed_indices)
            gate_values = gate_values * selected_gate_logits.sigmoid()

        gate_values = rearrange(gate_values, '... m e -> e ... m 1')

        # get routed input

        routed = einx.get_at('... [n] d, ... m e -> e ... m d', x, routed_indices)

        # handle experts being multiheaded tensor

        experts = self.experts

        if torch.is_tensor(experts) and experts.ndim == 3:
            experts = rearrange(experts, 'e i o -> e 1 i o')

        # forward routed input through the correct experts

        outputs = []

        for routed_input, expert in zip(routed, experts):
            if torch.is_tensor(expert):
                output = einsum(routed_input, expert, 'b h m i, h i o -> b h m o')
            else:
                output = expert(routed_input)

            outputs.append(output)

        outputs = torch.stack(outputs)

        # multiply by the gates, competitive or not

        outputs = outputs * gate_values

        # route back

        routed_back_outputs = torch.zeros_like(x)

        routed_back_outputs = einx.set_at(
            '... [n] d, ... m e, e ... m d -> ... [n] d',
            routed_back_outputs,
            routed_indices,
            outputs
        )

        # if single headed to start off with, squeeze out

        if single_headed:
            routed_back_outputs = rearrange(routed_back_outputs, 'b 1 n d -> b n d')

        return routed_back_outputs
