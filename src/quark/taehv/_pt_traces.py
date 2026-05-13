"""PyTorch trace classes shared between the CoreML and OpenVINO
exporters.

The TAEHV decoder is stateful via 3 ``MemBlock`` buffers. The
preferred path is to pass state as explicit inputs/outputs (rather
than the in-graph ``StateType`` mechanism CoreML offers but ANE
rejects with error -14). This file factors the two
``torch.nn.Module`` wrappers out of the CoreML exporter so OpenVINO
can reuse them — the math is identical; only the IR target differs.

Hardcoded to ``taehv1_5`` (T=4, patch_size=2). To export a different
TAEHV variant, update the block indices.

Runtime callers (``quark.taehv.coreml.runtime`` /
``quark.taehv.openvino.runtime``) DO NOT import this module — it
pulls in torch + the upstream ``taehv`` package, neither of which
the runtime backends need. Only the export-time scripts touch it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderStatic(nn.Module):
    """Stateless encoder.

    Input:  ``[4, 12, H, W]`` (4 frames after pixel_unshuffle(2))
    Output: ``[1, 32, H/8, W/8]``  (1 latent for the 4-frame chunk)
    """

    def __init__(self, taehv, h: int, w: int):
        super().__init__()
        self.blocks = nn.ModuleList(list(taehv.encoder))
        self._h, self._w = h, w

    def forward(self, x):
        h, w = self._h, self._w
        x = self.blocks[0](x)
        x = self.blocks[1](x)

        x = self.blocks[2].conv(x.reshape(2, 128, h, w))
        x = self.blocks[3](x)

        for i in (4, 5, 6):
            past = torch.cat([torch.zeros_like(x[:1]), x[:-1]], dim=0)
            x = self.blocks[i](x, past)

        x = self.blocks[7].conv(x.reshape(1, 128, h // 2, w // 2))
        x = self.blocks[8](x)

        for i in (9, 10, 11):
            x = self.blocks[i](x, torch.zeros_like(x))

        x = self.blocks[12].conv(x)
        x = self.blocks[13](x)

        for i in (14, 15, 16):
            x = self.blocks[i](x, torch.zeros_like(x))

        return self.blocks[17](x)


class DecoderExplicitState(nn.Module):
    """Stateful decoder; state passed as explicit I/O tensors.

    Inputs:  ``x [1, 32, latH, latW]``,
             ``state_lo  [3, 256, latH,    latW]``,
             ``state_mid [3, 128, latH*2,  latW*2]``,
             ``state_hi  [3,  64, latH*4,  latW*4]``
    Outputs: ``frames [4, 3, latH*16, latW*16]``,
             updated state_lo / state_mid / state_hi
    """

    def __init__(self, taehv, lat_h: int, lat_w: int):
        super().__init__()
        self.blocks = nn.ModuleList(list(taehv.decoder))
        self._h2, self._w2 = lat_h * 4, lat_w * 4
        self._h4, self._w4 = lat_h * 8, lat_w * 8

    def forward(self, x, state_lo, state_mid, state_hi):
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        x = self.blocks[2](x)

        save_3 = x
        x = self.blocks[3](x, state_lo[0:1])
        save_4 = x
        x = self.blocks[4](x, state_lo[1:2])
        save_5 = x
        x = self.blocks[5](x, state_lo[2:3])
        new_lo = torch.cat([save_3, save_4, save_5], dim=0)

        x = self.blocks[6](x)
        x = self.blocks[7].conv(x)
        x = self.blocks[8](x)

        save_9 = x
        x = self.blocks[9](x, state_mid[0:1])
        save_10 = x
        x = self.blocks[10](x, state_mid[1:2])
        save_11 = x
        x = self.blocks[11](x, state_mid[2:3])
        new_mid = torch.cat([save_9, save_10, save_11], dim=0)

        x = self.blocks[12](x)
        x = self.blocks[13].conv(x)
        x = x.reshape(2, 128, self._h2, self._w2)
        x = self.blocks[14](x)

        past = torch.cat([state_hi[0:1], x[:1]], dim=0)
        save_15 = x[1:2]
        x = self.blocks[15](x, past)

        past = torch.cat([state_hi[1:2], x[:1]], dim=0)
        save_16 = x[1:2]
        x = self.blocks[16](x, past)

        past = torch.cat([state_hi[2:3], x[:1]], dim=0)
        save_17 = x[1:2]
        x = self.blocks[17](x, past)
        new_hi = torch.cat([save_15, save_16, save_17], dim=0)

        x = self.blocks[18](x)
        x = self.blocks[19].conv(x)
        x = x.reshape(4, 64, self._h4, self._w4)
        x = self.blocks[20](x)
        x = self.blocks[21](x)
        x = self.blocks[22](x)

        x = F.pixel_shuffle(x, 2)
        x = x.clamp(0, 1)

        return x, new_lo, new_mid, new_hi


def load_taehv_pytorch(ae_repo: str):
    """Snapshot-download the upstream TAEHV checkpoint and return a
    loaded ``taehv.TAEHV`` instance (f32, eval())."""
    import pathlib

    try:
        from taehv import TAEHV
    except ImportError as e:
        raise ImportError(
            "Exporting TAEHV requires the upstream 'taehv' package — "
            "install with ``pip install 'quark[export]'``."
        ) from e

    try:
        import huggingface_hub
        base = pathlib.Path(huggingface_hub.snapshot_download(ae_repo))
    except Exception:
        base = pathlib.Path(ae_repo)

    ckpt = base if base.is_file() else base / "taehv1_5.pth"
    return TAEHV(str(ckpt)).eval().to(torch.float32)
