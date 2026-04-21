"""Model implementations on ``popcorn.functional`` + ``popcorn.nn``.

Each module wires the popcorn kernel surface into a full forward pass.
No torch in the runtime path — tensor storage is backend-native
(``mx.array`` on Metal, future ctypes CUDA tensors on CUDA).

    from popcorn.models.waypoint_15 import Waypoint15, Waypoint15Config

    model = Waypoint15(Waypoint15Config())
    model.load_state_dict(popcorn.nn.io.load_safetensors("model.safetensors"))
    caches = model.init_kv_caches()
    out = model(latent, sigma=0.7, caches=caches)
"""
