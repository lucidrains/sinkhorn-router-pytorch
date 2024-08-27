from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import Module

import einx
from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange

from sinkhorn_router_pytorch.sinkhorn_router import Gating, SinkhornRouter

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# class

class SwitchHead(Module):
    def __init__(
        self,
        dim,
        *,
        num_experts,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = True,
    ):
        super().__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        self.causal = causal

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_k = nn.Linear(dim, dim_inner, bias = False)

        self.value_experts = nn.Parameter(torch.randn(num_experts, heads, dim, dim_head))
        self.output_experts = nn.Parameter(torch.randn(num_experts, heads, dim_head, dim))

        self.value_router = SinkhornRouter(
            dim = dim,
            experts = self.value_experts,
            causal = causal,
            has_gating = False
        )

        self.output_router = SinkhornRouter(
            dim = dim,
            experts = self.output_experts,
            causal = causal,
            has_gating = False
        )

        self.to_gates = Gating(dim = dim, heads = heads, num_experts = num_experts)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        mask = None
    ):

        queries = self.to_q(x)
        keys = self.to_k(x)

        queries = self.split_heads(queries)
        keys = self.split_heads(keys)

        gates = self.to_gates(x)

        x = repeat(x, 'b n d -> b h n d', h = self.heads)
        values = self.value_router(x, gates = gates)

        queries = queries * self.scale
        sim = einsum(queries, keys, 'b h i d, b h j d -> b h i j')

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum(attn, values, 'b h i j, b h j d -> b h i d')

        out = self.output_router(out, gates = gates)

        out = reduce(out, 'b h n d -> b n d', 'sum')
        return out