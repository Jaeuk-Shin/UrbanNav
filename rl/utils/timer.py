"""
Per-component timing instrumentation for the PPO training loop.

Tracks wall-clock time (with optional CUDA synchronisation for GPU ops)
and computes per-iteration statistics that can be logged to W&B.
"""

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch


class StepTimer:
    """Accumulate named timing measurements across a rollout iteration.

    Usage::

        timer = StepTimer(cuda_sync=True)

        # Context-manager style (preferred):
        with timer("encoder"):
            features = encoder(obs)

        # Manual start/stop:
        timer.start("env_step")
        obs, r, d, t, info = env.step(action)
        timer.stop("env_step")

        # After the iteration, get stats dict for W&B:
        stats = timer.stats()   # {name: {mean, std, min, max, total, count, pct}}
        log_dict = timer.wandb_dict()  # flat dict ready for wandb.log()
        timer.reset()
    """

    def __init__(self, cuda_sync: bool = True):
        self.cuda_sync = cuda_sync and torch.cuda.is_available()
        self._records: dict[str, list[float]] = defaultdict(list)
        self._starts: dict[str, float] = {}

    # ── context manager ──────────────────────────────────────────────
    @contextmanager
    def __call__(self, name: str):
        """Time a block via ``with timer("name"): ...``."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    # ── manual start / stop ──────────────────────────────────────────
    def start(self, name: str):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self._starts[name] = time.perf_counter()

    def stop(self, name: str):
        if self.cuda_sync:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._starts.pop(name)
        self._records[name].append(elapsed)

    # ── record an externally-measured duration ───────────────────────
    def record(self, name: str, seconds: float):
        """Add a pre-measured duration (e.g. from a subprocess info dict)."""
        self._records[name].append(seconds)

    # ── statistics ───────────────────────────────────────────────────
    def stats(self) -> dict:
        """Return {name: {mean, std, min, max, total, count}} for all timers.

        Also adds ``pct`` — percentage of the iteration total (sum of all
        timer totals).  This is approximate when timers overlap.
        """
        result = {}
        for name, vals in self._records.items():
            arr = np.array(vals)
            result[name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "total": float(arr.sum()),
                "count": len(vals),
            }

        grand_total = sum(s["total"] for s in result.values())
        for s in result.values():
            s["pct"] = s["total"] / grand_total * 100 if grand_total > 0 else 0.0

        return result

    def wandb_dict(self, prefix: str = "timing") -> dict:
        """Flat dict for ``wandb.log()``.

        Logs ``{prefix}/{name}_mean``, ``_std``, ``_total``, ``_pct``
        for each timer.
        """
        st = self.stats()
        d = {}
        for name, s in st.items():
            d[f"{prefix}/{name}_mean"] = s["mean"]
            d[f"{prefix}/{name}_std"] = s["std"]
            d[f"{prefix}/{name}_total"] = s["total"]
            d[f"{prefix}/{name}_pct"] = s["pct"]
        return d

    def summary_str(self) -> str:
        """One-line console summary of per-component totals and percentages."""
        st = self.stats()
        if not st:
            return ""
        parts = []
        # Sort by total descending
        for name, s in sorted(st.items(), key=lambda x: -x[1]["total"]):
            parts.append(f"{name}={s['total']:.2f}s({s['pct']:.0f}%)")
        return "  timing: " + "  ".join(parts)

    def reset(self):
        self._records.clear()
        self._starts.clear()
