from __future__ import annotations
from math import ceil
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

def pad_seq_to_multiple(t, mult):
    seq_len = t.shape[1]
    next_seq_len_mult = ceil(seq_len / mult) * mult
    remainder = next_seq_len_mult - seq_len

    if remainder == 0:
        return t, remainder

    t = F.pad(t, (0, 0, 0, remainder), value = 0.)
    return t, remainder

# sinkhorn related functions

def log(t, eps = 1e-6):
    return t.clamp(min = eps).log()

def gumbel_like(t, eps = 1e-6):
    noise = torch.rand_like(t)
    return -log(-log(noise, eps), eps)

def sinkhorn(
    t,
    num_iters = 8,
    gumbel = False,
    temperature = 1.,
    eps = 1e-6,
    noise_inv_temp = 1.
):
    t = log(t, eps)

    assert temperature > 0.
    t = t / temperature

    if gumbel:
        t = t + gumbel_like(t, eps) * noise_inv_temp

    for _ in range(num_iters):
        t = t - t.logsumexp(dim = -2, keepdim = True)
        t = t - t.logsumexp(dim = -1, keepdim = True)

    return t.exp()

# main module

class SinkhornRouter(Module):
    def __init__(
        self,
        dim,
        *,
        experts: ModuleList | List[Module] | Tensor | None = None,
        causal = False,
        sinkhorn_iters = 8,
        temperature = 1.,
        gumbel_noise = False,
        noise_inv_temp = 1.,
        num_experts: int | None = None,
        competitive: bool | None = None,
        min_seq_len_route: int | None = None
    ):
        super().__init__()
        assert exists(experts) ^ exists(num_experts), 'either `experts` or `num_experts` is given, but not both'

        if exists(experts):
            num_experts = len(experts)

        # only use competitive gates for non-causal by default

        competitive = default(competitive, not causal)
        assert not (causal and competitive), 'causal sequences cannot have competitive gates'
        self.competitive = competitive

        self.min_seq_len_route = default(min_seq_len_route, num_experts)

        # experts are a ModuleList where length is number of experts
        # if a Tensor is given, it must be in shape of (experts, [optional] heads, dim_in, dim_out)

        heads = None

        if exists(experts):
            heads = default(heads, 1 if experts.ndim == 3 else experts.shape[1])

            assert torch.is_tensor(experts) and experts.ndim == 4 and experts.shape[1] == heads, f'experts do not have the correct number of heads. expected {self.heads} - received {experts.shape[1]}'

        self.heads = default(heads, 1)

        if isinstance(experts, list):
            experts = ModuleList(experts)

        self.experts = experts
        self.num_experts = num_experts

        # gating and sinkhorn related

        self.to_gate_weight = nn.Parameter(torch.randn(heads, dim, num_experts))

        self.temperature = temperature
        self.gumbel_noise = gumbel_noise
        self.noise_inv_temp = noise_inv_temp
        self.sinkhorn_iters = sinkhorn_iters

    def non_routed_forward(
        self,
        x,
        mask = None
    ):
        single_headed = x.ndim == 3

        # handle single headed

        if single_headed:
            x = rearrange(x, 'b n d -> b 1 n d')

        assert x.shape[1] == self.heads, f'expected input to have head dimension of {self.heads} but received {x.shape[1]}'

        # expert routing is just an argmax

        gates = einsum(x, self.to_gate_weight, 'b h n d, h d e -> b h n e')

        expert_indices = gates.argmax(dim = -1)

        # final gate values will just sigmoid(gates) for non-competitive, and nothing for competitive

        gate_values = None

        if not self.competitive:
            gate_values = einx.get('b h n [e], b h n -> b h n 1', gates, expert_indices).sigmoid()

        # forward experts in a non-balanced routed way

        experts = self.experts

        if single_headed and torch.is_tensor(experts):
            experts = rearrange(experts, 'e i o -> e 1 i o')

        # handle both experts being a multi-headed tensor (for switchehead mixture-of-attention)
        # or the traditional experts, where each expert is a separate module (usually feedforward)

        if torch.is_tensor(self.experts):
            selected_experts = einx.get_at('[e] h i o, b h n -> b h n i o', experts, expert_indices)
            output = einsum(x, selected_experts, 'b h n i, b h n i o -> b h n o')
        else:
            output = torch.zeros_like(x)

            for expert_index, expert in enumerate(experts):
                one_expert_mask = expert_indices == expert_index
                one_expert_inputs = x[one_expert_mask]
                one_expert_output = expert(one_expert_inputs)

                output[one_expert_mask] = one_expert_output

        # gate values

        if exists(gate_values):
            output = einx.multiply('b h n d, b h n -> b h n d', output, gate_values)

        # squeeze out single head

        if single_headed:
            output = rearrange(output, 'b 1 n d -> b n d')

        return output

    def forward(
        self,
        x,
        mask = None,
        noise_inv_temp: float | None = None,
        route: bool | None = None
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

        device = x.device

        noise_inv_temp = default(noise_inv_temp, self.noise_inv_temp)

        batch, seq_len, single_headed = x.shape[0], x.shape[-2], x.ndim == 3

        # determine whether routing is needed or not

        route = default(route, seq_len > self.min_seq_len_route)

        if not route:
            return self.non_routed_forward(x, mask)

        # auto pad to correct multiple for divying up tokens amongst 'experts'

        x, seq_pad = pad_seq_to_multiple(x, self.num_experts)

        padded_seq_len = x.shape[-2]

        if seq_pad > 0 and not exists(mask):
            mask = einx.less(
                'n, b -> b n',
                torch.arange(padded_seq_len, device = device),
                torch.full((batch,), seq_len, device = device)
            )

        tokens_per_expert = padded_seq_len // self.num_experts

        # handle single headed

        if single_headed:
            x = rearrange(x, 'b n d -> b 1 n d')

        assert x.shape[1] == self.heads, f'expected input to have head dimension of {self.heads} but received {x.shape[1]}'

        # project to gates

        gate_logits = einsum(x, self.to_gate_weight, 'b h n d, h d e -> b h n e')

        # masking for variable sequence lengths

        if exists(mask):
            gate_logits = einx.where('b n, b h n e, -> b h n e', mask, gate_logits, -torch.finfo(gate_logits.dtype).max)

        # sinkhorn ensures balanced routing
        # if non-competitive, do not differentiate through sinkhorn, technique came from megatron, afaict
        # they then just select the sigmoid of the selected gate, which should work given recent papers (Sigmoid MoE etc)

        sinkhorn_context = nullcontext if self.competitive else torch.no_grad

        with sinkhorn_context():

            competitive_gates = sinkhorn(
                gate_logits,
                temperature = self.temperature,
                gumbel = self.gumbel_noise,
                num_iters = self.sinkhorn_iters,
                noise_inv_temp = noise_inv_temp
            )

            if exists(mask):
                competitive_gates = einx.where('b n, b h n e, -> b h n e', mask, competitive_gates, 0.)

            gate_values, routed_indices = competitive_gates.topk(tokens_per_expert, dim = -2)
            hard_gate_values = (gate_values > 0.5).float()

        # straight through if training

        if self.training and self.competitive:
            gate_values = hard_gate_values + gate_values - gate_values.detach()
        else:
            gate_values = hard_gate_values

        # causal, non-competitive will select the gate logits and use the sigmoid - used in megatron

        if not self.competitive:
            selected_gate_logits = einx.get_at('... [n] e, ... m e -> ... m e', gate_logits, routed_indices)
            gate_values = gate_values * selected_gate_logits.sigmoid()

        # return gates and routing indices if no experts

        if not exists(self.experts):
            if single_headed:
                routed_indices, gate_values = tuple(rearrange(t, 'b 1 ... -> b ...') for t in (routed_indices, gate_values))

            return routed_indices, gate_values

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

        outputs = einx.multiply(
            'e b h m d, b h m e -> e b h m d',
            outputs, gate_values
        )

        # route back

        routed_back_outputs = torch.zeros_like(x)

        routed_back_outputs = einx.set_at(
            '... [n] d, ... m e, e ... m d -> ... [n] d',
            routed_back_outputs,
            routed_indices,
            outputs
        )

        # remove seq padding if auto-padded

        routed_back_outputs = routed_back_outputs[..., :seq_len, :]

        # if single headed to start off with, squeeze out

        if single_headed:
            routed_back_outputs = rearrange(routed_back_outputs, 'b 1 n d -> b n d')

        return routed_back_outputs
