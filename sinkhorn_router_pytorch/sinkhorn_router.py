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
        sinkhorn_iters = 8,
        temperature = 1.,
        competitive: bool | None = None
    ):
        super().__init__()

        # only use competitive gates for non-causal by default

        competitive = default(competitive, not causal)
        assert not (causal and competitive), 'causal sequences cannot have competitive gates'
        self.competitive = competitive

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
        self.sinkhorn_iters = sinkhorn_iters

    def forward(
        self,
        x,
        mask = None
    ):
        """
        ein notation:
        b - batch
        n - sequence length
        d - feature dimension
        e - experts
        m - tokens per expert
        """

        seq_len = x.shape[-2]
        assert divisible_by(seq_len, self.num_experts)
        tokens_per_expert = seq_len // self.num_experts

        # project to gates

        gate_logits = self.to_gates(x)

        # masking for variable sequence lengths

        if exists(mask):
            gate_logits = einx.where('b n, b n e, -> b n e', mask, gate_logits, -torch.finfo(gates.dtype).max)

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
            selected_gate_logits = einx.get_at('b [n] e, b m e -> b m e', gate_logits, routed_indices)
            gate_values = gate_values * selected_gate_logits.sigmoid()

        gate_values = rearrange(gate_values, 'b m e -> e b m 1')

        # get routed input

        routed = einx.get_at('b [n] d, b m e -> e b m d', x, routed_indices)

        # forward routed input through the correct experts

        outputs = []

        for routed_input, expert in zip(routed, self.experts):
            if torch.is_tensor(expert):
                output = routed_input @ expert
            else:
                output = expert(routed_input)

            outputs.append(output)

        outputs = torch.stack(outputs)

        # multiply by the gates, competitive or not

        outputs = outputs * gate_values

        # route back

        routed_back_outputs = torch.zeros_like(x)

        routed_back_outputs = einx.set_at(
            'b [n] d, b m e, e b m d -> b [n] d',
            routed_back_outputs,
            routed_indices,
            outputs
        )

        return routed_back_outputs
