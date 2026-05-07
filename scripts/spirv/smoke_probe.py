"""Quick end-to-end smoke for the _spv_dispatch C extension.

Runs probe → caps_from_probe → SpvDriver and prints the highlights.
Useful for sanity-checking the build on a new chip.

    uv run python scripts/spirv/smoke_probe.py
"""

from __future__ import annotations

from quark.drivers import spv


def main() -> None:
    print("available:", spv.is_available())
    print()
    print("--- devices ---")
    for d in spv.enumerate_devices():
        print(
            f"  index={d['index']}, name={d['device_name']!r}, "
            f"vendor=0x{d['vendor_id']:04x}, type={d['device_type']}"
        )

    idx = spv.pick_default_device()
    print(f"\npicked: device_index={idx}")

    print("\n--- probe device 0 (highlights) ---")
    p = spv.probe(0)
    keys = (
        "device_name",
        "vendor_id",
        "device_id",
        "subgroup_size",
        "bf16_cooperative_matrix",
        "max_compute_shared_memory_size",
        "max_push_constants_size",
        "atomic_f32_add_buffer",
        "atomic_f16_add_buffer",
    )
    for k in keys:
        print(f"  {k}: {p[k]!r}")

    shapes = p["cooperative_matrix_shapes"]
    print(f"\n  cooperative_matrix_shapes ({len(shapes)}):")
    for s in shapes:
        sat = "  sat" if s["saturating_accumulation"] else ""
        print(
            f"    {s['m']}x{s['n']}x{s['k']}  "
            f"{s['a_dtype']}/{s['b_dtype']} -> {s['result_dtype']}  "
            f"{s['scope']}{sat}"
        )

    print("\n--- SpvDriver.caps (Battlemage) ---")
    drv = spv.SpvDriver()
    caps = drv.caps
    print(f"  family: {caps.family}")
    print(f"  name: {caps.name}")
    print(f"  arch_tag: {caps.arch_tag}")
    print(f"  subgroup_width: {caps.subgroup_width}")
    print(f"  supports_bf16_mma: {caps.supports_bf16_mma}")
    print(f"  max_smem_per_block: {caps.max_smem_per_block}")
    print(f"  matmul_shapes:")
    for s in sorted(caps.matmul_shapes):
        print(f"    {s}")


if __name__ == "__main__":
    main()
