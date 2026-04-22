"""Backend lowerers for the quark IR.

Each backend is a visitor package under `quark.lower.<target>` that
walks a `Module` and emits target source text. V1 ships PTX only.

See QUARK_IR_PROPOSAL.md §9–§10.
"""
