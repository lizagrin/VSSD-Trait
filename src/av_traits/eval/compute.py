"""Computational characteristics: parameter count, FLOPs, latency, memory"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch


def count_params_M(model: torch.nn.Module, only_trainable: bool = False) -> float:
    s = sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_trainable))
    return round(s / 1e6, 1)


def measure_flops_G(model: torch.nn.Module, batch: Dict) -> float:
    """Total GFLOPs/clip for a single forward pass on ``batch``."""
    model.eval()
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, (batch,))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return round(flops.total() / 1e9, 0)
    except Exception as e:  # pragma: no cover — best effort
        print("FlopCountAnalysis failed:", e)
        return float("nan")


def measure_latency_ms(model: torch.nn.Module, batch: Dict, device: str,
                       n: int = 100, warmup: int = 10, fp16: bool = True
                       ) -> float:
    """Median latency in milliseconds over ``n`` runs (after ``warmup``)."""
    model = model.to(device).eval()
    bdev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    if device == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        ctx = torch.cuda.amp.autocast(enabled=fp16)
    else:
        ctx = torch.cuda.amp.autocast(enabled=False)

    times = []
    with torch.no_grad():
        for _ in range(warmup):
            with ctx:
                model(bdev)
        for _ in range(n):
            if device == "cuda":
                s.record()
                with ctx:
                    model(bdev)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                model(bdev)
                times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 0)


def peak_memory_MB(model: torch.nn.Module, batch: Dict) -> float:
    """Peak GPU memory (MiB) recorded during a single forward pass."""
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.reset_peak_memory_stats()
    bdev = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        model(bdev)
    return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 0)
