"""Model implementations on ``quark.functional`` + ``quark.nn``.

Each module wires the quark kernel surface into a full forward pass.
No torch in the runtime path — tensor storage is ``QuarkTensor``
on both backends (Metal pool buffer on Metal, ``cuMemAlloc`` on CUDA).

    from quark.models.waypoint_15 import Waypoint15, Waypoint15Config

    model = Waypoint15(Waypoint15Config())
    model.load_state_dict(quark.nn.io.load_safetensors("model.safetensors"))
    caches = model.init_kv_caches()
    out = model(latent, sigma=0.7, caches=caches)
"""
