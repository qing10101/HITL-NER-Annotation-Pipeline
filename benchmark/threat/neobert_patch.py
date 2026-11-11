"""NeoBERT's remote code builds its FFN with xformers.ops.SwiGLU, whose fused CUDA
kernels are not shipped in the Windows xformers wheels ("No such operator
xformers::swiglu_packedw"), and the eager op refuses packed weights.  We swap in
a plain-PyTorch forward that reuses the module's own parameters: identical maths,
gradients flow normally.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _eager_forward(self, x):
    w1, b1, w2, b2, w3, b3 = self._ordered_params()
    hidden = F.silu(F.linear(x, w1, b1)) * F.linear(x, w2, b2)
    return F.linear(hidden, w3, b3)


def use_eager_swiglu(model) -> int:
    try:
        import xformers.ops as xo
    except ImportError:
        return 0
    n = 0
    for m in model.modules():
        if isinstance(m, xo.SwiGLU):
            m.forward = types.MethodType(_eager_forward, m)
            n += 1
    return n


def enable_layer_checkpointing(model) -> int:
    """NeoBERT does not implement HF gradient checkpointing; wrap each encoder
    block's forward in torch.utils.checkpoint (non-reentrant) during training so
    only block inputs are kept and activations are recomputed in backward.  On a
    6 GB GPU this keeps fp32 AdamW fine-tuning at batch 8 x 256 under ~4.5 GB."""
    n = 0
    for name, mod in model.named_modules():
        if name.endswith("transformer_encoder") and isinstance(mod, torch.nn.ModuleList):
            for layer in mod:
                orig = layer.forward

                def wrapped(*args, _orig=orig, _layer=layer, **kwargs):
                    if _layer.training and torch.is_grad_enabled():
                        return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
                    return _orig(*args, **kwargs)

                layer.forward = wrapped
                n += 1
    return n
