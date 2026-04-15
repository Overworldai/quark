"""Backend lowerers for the popcorn IR.

Each backend is a visitor package under `popcorn.lower.<target>` that
walks a `Module` and emits target source text. V1 ships PTX only.

See POPCORN_IR_PROPOSAL.md §9–§10.
"""
