import pytest

import torch
from torch import nn
from sinkhorn_router_pytorch import SinkhornRouter

@pytest.mark.parametrize('competitive,causal', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('seq_len', (1, 77, 1024, 1999))
@pytest.mark.parametrize('experts_as_tensor', (True, False))
@pytest.mark.parametrize('has_masking', (True, False))
def test_moe(
    competitive,
    causal,
    seq_len,
    experts_as_tensor,
    has_masking
):

    if experts_as_tensor:
        experts = nn.Parameter(torch.randn(4, 512, 512))
    else:
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

    mask = None
    if has_masking:
        mask = torch.randint(0, 2, (2, seq_len)).bool()

    out = router(x, mask = mask)

    assert out.shape == (2, seq_len, 512)

@pytest.mark.parametrize('competitive,causal', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('seq_len', (1, 77, 1024, 1999))
@pytest.mark.parametrize('has_masking', (True, False))
def test_multiheaded_experts(
    competitive,
    causal,
    seq_len,
    has_masking
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

    mask = None
    if has_masking:
        mask = torch.randint(0, 2, (2, seq_len)).bool()

    out = router(x, mask = mask)

    assert out.shape == (2, 8, seq_len, 256)
