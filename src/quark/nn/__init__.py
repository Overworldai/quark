"""``quark.nn`` — minimal inference-only module system.

No autograd, no training, no optimizer. Holds named parameters as
backend-native tensors (``mx.array`` on Metal, raw CUDA device
pointers on CUDA — no torch in the runtime path), supports
``state_dict()`` / ``load_state_dict()``, and a ``forward()``
convention.

    import quark.nn as nn

    class MyModel(nn.Module):
        def __init__(self, d):
            self.weight = nn.Parameter(PT.randn(d, d))
            self.layers = nn.ModuleList([MyBlock(d) for _ in range(12)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x @ self.weight.data.T
"""

from quark.nn.layers import (
    MLP,
    AdaGateResidual,
    AdaRMSNorm,
    Add,
    Cast,
    ControllerInputEmbedding,
    EulerStep,
    HeadRMSNorm,
    KVCacheUpdate,
    Linear,
    MLPFusion,
    OwlAttn,
    Patchify,
    RMSNorm,
    SiLU,
    Unpatchify,
    ValueResidualPacked,
)
from quark.nn.module import Module, ModuleList, Parameter
from quark.nn.moe import MoE

__all__ = [
    "MLP",
    "AdaGateResidual",
    "AdaRMSNorm",
    "Add",
    "Cast",
    "ControllerInputEmbedding",
    "EulerStep",
    "HeadRMSNorm",
    "KVCacheUpdate",
    "Linear",
    "MLPFusion",
    "MoE",
    "Module",
    "ModuleList",
    "OwlAttn",
    "Parameter",
    "Patchify",
    "RMSNorm",
    "SiLU",
    "Unpatchify",
    "ValueResidualPacked",
]
