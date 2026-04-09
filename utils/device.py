"""
Device detection and memory management utilities.

Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
"""

import gc
import torch


def get_device() -> str:
    """Return the best available compute device string."""
    if torch.backends.mps.is_available():
        # Quick sanity check — MPS can be reported available but fail
        try:
            _x = torch.zeros(1, device="mps") + 1
            return "mps"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """Return the most efficient dtype for the given device."""
    if device in ("mps", "cuda"):
        return torch.float16
    return torch.float32


def free_memory(model=None) -> None:
    """
    Move model to CPU and free device cache.
    Safe to call with model=None (just flushes cache).
    """
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_stats() -> dict:
    """Return a dict of current memory usage (MB)."""
    stats = {}
    if torch.backends.mps.is_available():
        # MPS doesn't expose detailed per-tensor stats like CUDA
        stats["backend"] = "mps"
        stats["allocated_mb"] = torch.mps.current_allocated_memory() / (1024 ** 2)
    elif torch.cuda.is_available():
        stats["backend"] = "cuda"
        stats["allocated_mb"]  = torch.cuda.memory_allocated()  / (1024 ** 2)
        stats["reserved_mb"]   = torch.cuda.memory_reserved()   / (1024 ** 2)
    else:
        stats["backend"] = "cpu"
    return stats
