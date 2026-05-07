"""``Engine`` — Intel SPIR-V backend (stub).

PORTABILITY_PLAN §3.4. The Engine subclass that consumes the §3.1
SpvDriver + the (still-stub) §3.2 SpirVLowerer + the §3.6 attention
path. Construction succeeds (so consumer code can probe the family
and feature-flag accordingly) but every method that needs the lowerer
or VAE backend raises ``NotImplementedError`` with a pointer at the
plan.

Architectural choices baked in here, per §3.4 of the plan:

  * **Host tensor format: numpy + tagged-dtype carriers.** Mirrors
    EngineMetal — host code has no torch dependency, parameters /
    activations / segments tables / KV cache pin into Vulkan
    buffers as ``QuarkTensor`` instances, the latent boundary is
    plain numpy (no torch round-trip).

  * **Lazy dispatch idiom: VkCommandBuffer accumulation.** Same
    semantics as Metal's ``quark.lazy()``: ops record into the
    active VkCommandBuffer; commit fires on output read, explicit
    sync, or op-count threshold. The two-queue + VkSemaphore
    overlap (sync=False parity) lands once a perf number from §3.7
    v1 demands it.

  * **VAE backend: OpenVINO TAEHV (planned).** The current stub
    raises NotImplementedError on ``gen_frame`` returning pixels;
    the OpenVINO export is a parallel workstream tracked in §3.4
    of the plan. Until that lands, ``EngineIntel`` is useful for
    the v1 smoke / kernel-coverage path but not real inference.

  * **Weight pinning: Vulkan device-local buffers.** Mirrors
    EngineMetal's ``_pin_params_to_device`` — numpy weight
    carriers get pinned into ``DEVICE_LOCAL`` Vulkan buffers at
    startup so the dispatcher's input cache hits 100% on the
    hot path. Lands with the §3.1 compile/launch commit.

Construction does:
  1. Build the ``SpvDriver`` (probes the device, populates caps).
  2. Resolve / load the YAML config (same path as EngineMetal +
     EngineCUDA).
  3. **Skip** model construction for now — the ``Waypoint15``
     forward needs SpirVLowerer to actually emit kernels, which
     is §3.2. The stub stores enough state for ``model_cfg`` to
     work (Biome's metadata reads) and raises on anything that
     would actually dispatch.

Quant is forced to all-bf16 here too: Intel Battlemage doesn't
expose fp8 cooperative-matrix shapes (probed 2026-05-08; see
``scripts/spirv/probe_coopmat.md``).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from quark.engine.base import Engine, _resolve_quant
from quark.models.waypoint_15 import (
    CtrlInput,
    QuantConfig,
    Waypoint15Config,
)


class EngineIntel(Engine):
    """Stub Engine subclass for the SPIR-V / Vulkan backend.

    Constructable today; not yet inferrable. Every method past
    config-load raises ``NotImplementedError`` with a clear
    PORTABILITY_PLAN pointer at the section that's responsible
    for unlocking it.
    """

    def __init__(
        self,
        model_uri: str | Path,
        *,
        quant: str | QuantConfig | None = "bf16",
        config_overrides: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        load_weights: bool = True,
        taehv_cache_dir: str | Path | None = None,
        device_index: int | None = None,
    ) -> None:
        del taehv_cache_dir  # OpenVINO-TAEHV path not yet implemented
        del load_weights  # no model construction yet — see class docstring

        # 1. Probe the Vulkan device + build the driver. Failure here is
        #    a hard error: if the consumer asked for the Intel engine
        #    explicitly and Vulkan isn't reachable, surface that loudly
        #    rather than silently falling back to CPU.
        from quark.drivers import spv  # noqa: PLC0415

        if not spv.is_available():
            raise RuntimeError(
                "EngineIntel: Vulkan is not available on this host. "
                "Install libvulkan + an ICD (Mesa for open-source Intel, "
                "or vendor proprietary), then rebuild quark via "
                "``pip install -e .`` on Linux. See PORTABILITY_PLAN §3.1."
            )
        self._driver = spv.SpvDriver(device_index=device_index)
        self._caps = self._driver.caps

        # 2. Resolve config (same as EngineMetal / EngineCUDA — central
        #    YAML loader). Imported lazily to avoid a circular import
        #    when ``quark.models`` decides to start importing engines.
        from quark.models.config import _resolve_path, load_yaml_config  # noqa: PLC0415

        model_path = _resolve_path(str(model_uri))
        self.raw_cfg = load_yaml_config(model_path)
        if config_overrides:
            self.raw_cfg = {**self.raw_cfg, **config_overrides}

        # Battlemage exposes no fp8 coopmat — same constraint as Metal.
        # If the caller asked for fp8, downgrade with a warning.
        resolved = _resolve_quant(quant)
        all_bf16 = QuantConfig.all_bf16()
        if resolved != all_bf16:
            warnings.warn(
                "EngineIntel: forcing all-bf16 quant — Intel Battlemage "
                "doesn't expose fp8 cooperative-matrix shapes. See "
                "scripts/spirv/probe_coopmat.md.",
                stacklevel=2,
            )
            resolved = all_bf16
        self._quant = resolved

        # ``Waypoint15Config`` is built up-front so ``model_cfg`` works
        # for Biome metadata reads even though the model itself isn't
        # constructed yet.
        self.cfg = Waypoint15Config.from_dict(self.raw_cfg, quant=resolved)

        # 3. ``device`` is informational on Intel (no torch device
        #    binding — Vulkan handles device state internally). Keep the
        #    attribute for parity with the parent class' type hint.
        self.device = torch.device("cpu") if device is None else torch.device(str(device))
        self.dtype = dtype

        # Model construction defers to §3.2 — we need a working
        # SpirVLowerer to actually emit kernels. Until then, stash
        # the load_weights flag for the eventual implementation.
        self.model = None
        self._ctrl_fill = None  # base class' _encode_ctrl checks this

    # ── Shared API surface — overrides ──────────────────────────

    def reset(self) -> None:
        """Reset the KV ring buffer + frame counter.

        Lands alongside the model-forward path in §3.2. Today the
        stub doesn't have any state to reset, so this is a no-op
        rather than NotImplementedError — matches the contract
        Biome expects when starting a fresh stream.
        """
        return None

    def append_frame(self, img, ctrl: CtrlInput | None = None):
        del img, ctrl
        raise NotImplementedError(
            "EngineIntel.append_frame: requires the SpirVLowerer + VAE "
            "encode path. Tracked in PORTABILITY_PLAN §3.2 (lowerer) and "
            "§3.4 (OpenVINO TAEHV). Today only construction + caps are "
            "wired."
        )

    def gen_frame(self, ctrl: CtrlInput | None = None, return_img: bool = True):
        del ctrl, return_img
        raise NotImplementedError(
            "EngineIntel.gen_frame: requires the SpirVLowerer + VAE "
            "decode path. See PORTABILITY_PLAN §3.7 v1 (kernel coverage) "
            "and §3.4 (VAE backend choice)."
        )

    def flush_pixels(self) -> torch.Tensor | None:
        return None  # no in-flight pipelined decode in the stub

    def submit_frame(self, ctrl: CtrlInput | None = None) -> None:
        del ctrl
        raise NotImplementedError(
            "EngineIntel.submit_frame: requires the pipelined-decode "
            "worker pattern. Lands with §3.7 v3 (real flash-attention "
            "+ sustained inference)."
        )

    def next_pixels(self) -> torch.Tensor | None:
        return None  # nothing in flight

    # ── Intel-specific accessors ────────────────────────────────

    @property
    def caps(self):
        """The ``DeviceCaps`` for the bound Vulkan device.

        Useful for tooling that wants to see what the engine
        actually probed (matmul_shapes, subgroup_width, smem
        budget) without poking at the driver internals.
        """
        return self._caps

    @property
    def driver(self):
        """The underlying ``SpvDriver``. Exposed for debugging /
        per-test inspection; production code should not poke at
        this."""
        return self._driver
