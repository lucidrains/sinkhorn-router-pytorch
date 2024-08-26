import torch
from torch import nn
from sinkhorn_router_pytorch import SinkhornRouter

def test_moe():

    experts = [
        nn.Linear(512, 512),
        nn.Linear(512, 512),
        nn.Linear(512, 512),
        nn.Linear(512, 512),
    ]

    router = SinkhornRouter(
        dim = 512,
        experts = experts,
        competitive = False,
        causal = False,
    )

    assert router.num_experts == 4

    x = torch.randn(1, 1017, 512)
    out = router(x)

    assert out.shape == (1, 1017, 512)

def test_multiheaded_experts():
    experts = nn.Parameter(torch.randn(16, 8, 512, 256)) # (experts, heads, dim [in], dim [out])

    router = SinkhornRouter(
        dim = 512,
        experts = experts,
        competitive = False,
        causal = False,
    )

    assert router.num_experts == 16

    x = torch.randn(1, 8, 1017, 512)
    out = router(x)

    assert out.shape == (1, 8, 1017, 256)
