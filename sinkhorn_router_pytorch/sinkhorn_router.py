from __future__ import annotations

from typing import List

import torch
from torch import Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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
        experts: ModuleList | List[Module] | Module | Tensor,
        causal = False,
        competitive_gates: bool | None = None,
        gumbel_noise = False,
        temperature = 1.
    ):
        super().__init__()
        self.experts = experts

    def forward(self, x):
        return x
