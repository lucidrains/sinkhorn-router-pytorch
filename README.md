## Sinkhorn Router - Pytorch (wip)

Self contained pytorch implementation of a sinkhorn based router, for mixture of experts or otherwise. Will contain both a causal and non-causal variant. The causal variant will follow the example used in Megatron

## Install

```bash
$ pip install sinkhorn-router-pytorch
```

## Usage

```python
import torch
from torch import nn
from sinkhorn_router_pytorch import SinkhornRouter

experts = nn.Parameter(torch.randn(8, 8, 512, 256)) # (experts, heads, dim [in], dim [out])

router = SinkhornRouter(
    dim = 512,
    experts = experts,
    competitive = True,
    causal = False,
)

x = torch.randn(1, 8, 1017, 512)
out = router(x) # (1, 8, 1017, 256)
```

## Citations

```bibtex
@article{Shoeybi2019MegatronLMTM,
    title   = {Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
    author  = {Mohammad Shoeybi and Mostofa Patwary and Raul Puri and Patrick LeGresley and Jared Casper and Bryan Catanzaro},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1909.08053},
    url     = {https://api.semanticscholar.org/CorpusID:202660670}
}
```

```bibtex
@article{Anthony2024BlackMambaMO,
    title   = {BlackMamba: Mixture of Experts for State-Space Models},
    author  = {Quentin Anthony and Yury Tokpanov and Paolo Glorioso and Beren Millidge},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.01771},
    url     = {https://api.semanticscholar.org/CorpusID:267413070}
}
```
