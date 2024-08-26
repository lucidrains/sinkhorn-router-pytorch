import pytest

import torch
from torch import nn
from sinkhorn_router_pytorch import SinkhornRouter

@pytest.mark.parametrize('competitive,causal', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('seq_len', (1, 77, 1024, 1999))
def test_moe(
    competitive,
    causal,
    seq_len
):

    experts = [
        nn.Linear(512, 512),
        nn.Linear(512, 512),
        nn.Linear(512, 512),
        nn.Linear(512, 512),
    ]

    router = SinkhornRouter(
        dim = 512,
        experts = experts,
        competitive = competitive,
        causal = causal,
    )

    assert router.num_experts == 4

    x = torch.randn(2, seq_len, 512)
    out = router(x)

    assert out.shape == (2, seq_len, 512)

@pytest.mark.parametrize('competitive,causal', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('seq_len', (1, 77, 1024, 1999))
def test_multiheaded_experts(
    competitive,
    causal,
    seq_len
):

    experts = nn.Parameter(torch.randn(16, 8, 512, 256)) # (experts, heads, dim [in], dim [out])

    router = SinkhornRouter(
        dim = 512,
        experts = experts,
        competitive = competitive,
        causal = causal,
    )

    assert router.num_experts == 16

    x = torch.randn(2, 8, seq_len, 512)
    out = router(x)

    assert out.shape == (2, 8, seq_len, 256)
