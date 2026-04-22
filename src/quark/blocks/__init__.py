"""L2 composable blocks for the IR Builder.

Layered architecture:
  L0 (blocks/l0/)  — single-pattern IR emission functions
  L1 (blocks/l1/)  — Block classes wrapping L0 into declarative specs
  L2 (blocks/l2/)  — composition blocks (KLoop, SmemPlan, etc.)
  DSL (blocks/dsl)  — framework: Block, BlockContext, KernelContext, C, Accumulators, SmemTile

Importers use the flat public API below. The l0/l1/l2 directories
exist for authors to find things, not for importers to navigate.
"""

# DSL primitives
from quark.blocks.dsl import (
    Accumulators,
    Block,
    BlockContext,
    C,
    Carry,
    KernelContext,
    SetupBlock,
    SmemTile,
    SmemTileSpec,
    Stage,
    TensorDecl,
    active_bctx,
    active_kctx,
    barrier,
    block_base,
    block_idx,
    c,
    gid,
    lane_id,
    tid,
    tig,
    warp_id,
)

# Legacy emit functions (still used by old-style kernel emit() bodies
# and internally by L1 blocks). Will be removed once all kernels migrate.
from quark.blocks.l0.epilogue import (
    emit_atomic_scatter_epilogue,
    emit_silu_cast_epilogue,
)
from quark.blocks.l0.gathered_tile_loader import emit_gathered_tile_load
from quark.blocks.l0.index_cache import emit_index_cache
from quark.blocks.l0.tile_loader import emit_tile_load

# L1 blocks. ScatterStore / SiluCastStore / AtomicScatterAdd retired
# in A8 — use ``pop.store_acc`` / ``pop.atomic_store_acc`` from
# quark.lang.epilogue instead. IndexCache / WorkListLoad retired in
# the inline-blocks pass — use ``pop.index_cache`` / ``pop.work_list_load``.
from quark.blocks.l1.mma_body import MmaBody

# L2 blocks
from quark.blocks.l2.run_pipeline import IterCtx, PipelineBody, run_pipeline
from quark.blocks.l2.smem_plan import SmemPlan

__all__ = [  # noqa: RUF022 — grouped by category (DSL / L1 / L2 / helpers / legacy)
    # DSL
    "Accumulators",
    # L1
    "Block",
    "BlockContext",
    "C",
    "Carry",
    "IterCtx",
    # L2
    "KernelContext",
    "MmaBody",
    "PipelineBody",
    "SetupBlock",
    "SmemPlan",
    "SmemTile",
    "SmemTileSpec",
    "Stage",
    "TensorDecl",
    "run_pipeline",
    # Active-ctx accessors + free-function helpers
    "active_bctx",
    "active_kctx",
    "barrier",
    "block_base",
    "block_idx",
    "c",
    # Legacy (transitional)
    "emit_atomic_scatter_epilogue",
    "emit_gathered_tile_load",
    "emit_index_cache",
    "emit_silu_cast_epilogue",
    "emit_tile_load",
    "gid",
    "lane_id",
    "tid",
    "tig",
    "warp_id",
]
