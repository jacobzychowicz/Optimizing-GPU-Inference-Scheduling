"""Real GPU load without a real model: each request runs matrix multiplies sized to its input.

Used by defaults/models/synthetic-gpu.json. Needs PyTorch. A request with
max_tokens N (or N characters of text) runs base_iters + N * iters_per_unit
multiplies of two matrix x matrix tensors, then waits for the GPU to finish.

Copy this file to start your own backend: load() once, infer() per request.
"""

import time


def load(config):
    import torch

    device = config.get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        elif config.get("allow_cpu"):
            device = "cpu"
        else:
            raise RuntimeError("no GPU visible to PyTorch. Set config.allow_cpu to true to run on the CPU anyway.")
    size = int(config.get("matrix", 2048))
    dtype = getattr(torch, config.get("dtype", "float16" if device != "cpu" else "float32"))
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    return {
        "torch": torch,
        "device": device,
        "a": a,
        "b": b,
        "base_iters": int(config.get("base_iters", 2)),
        "iters_per_unit": float(config.get("iters_per_unit", 0.5)),
        "token_field": config.get("token_field", "max_tokens"),
        "text_field": config.get("text_field", "text"),
    }


def _sync(state):
    torch = state["torch"]
    if state["device"].startswith("cuda"):
        torch.cuda.synchronize()
    elif state["device"] == "mps":
        torch.mps.synchronize()


def warmup(state):
    for _ in range(5):
        state["a"] @ state["b"]
    _sync(state)


def infer(state, job):
    payload = job.get("input") or {}
    if state["token_field"] in payload:
        units = float(payload[state["token_field"]])
    else:
        units = len(str(payload.get(state["text_field"], "")))
    iters = max(1, int(state["base_iters"] + units * state["iters_per_unit"]))
    started = time.perf_counter()
    out = state["a"]
    for _ in range(iters):
        out = out @ state["b"]
        out = out / out.abs().amax().clamp(min=1)  # keep values finite in float16
    _sync(state)
    return {"iters": iters, "seconds": time.perf_counter() - started}


def close(state):
    torch = state["torch"]
    if state["device"].startswith("cuda"):
        torch.cuda.empty_cache()
