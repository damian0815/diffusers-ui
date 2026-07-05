"""
mem_stats.py — Adaptive batch-size advisor for the pipeline worker.

Strategy
--------
* Start every session with ``probe_batch_size = 1``.
* After each *successful* generation record::

      (pixel_area, batch_size, peak_bytes_used)

  and bump ``probe_batch_size += 1``.
* An OOM event resets ``probe_batch_size`` to ``max(1, failed_batch_size − 1)``.

Model fitting
-------------
Once at least ``MIN_SAMPLES`` records exist, the advisor fits a power-law::

    memory_per_image ≈ a · pixel_area^k

for k ∈ {1, 2, 3} (linear / quadratic / cubic scaling with respect to
pixel count).  The best-fit exponent k is chosen by minimum RMSE on the
log-transformed residuals (i.e. best log-log fit).

Batch-size cap
--------------
``get_max_batch_size(pixel_area)`` returns the minimum of:

  * ``probe_batch_size``   — ramp-up guard (never jumps more than +1 per run)
  * ``floor(available_vram × SAFETY_MARGIN / predicted_memory_per_image)``
    — model-based cap (only active once ≥ MIN_SAMPLES records exist and
    the device reports a total memory figure)

Logging
-------
After every successful record ``record_success()`` emits a compact INFO log
line with:

  * measured peak memory and per-image breakdown
  * predicted memory under each of the three scaling laws
  * ★ marker on the currently preferred law
  * current probe and model-based batch-size ceiling
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
MIN_SAMPLES: int = 3          # records needed before we trust the fitted model
SAFETY_MARGIN: float = 0.80   # fraction of total VRAM considered safe to target


# ── device helpers ────────────────────────────────────────────────────────────

def _current_bytes(device: str) -> int:
    """Return currently-allocated device memory in bytes (best-effort)."""
    try:
        import torch
        if device.startswith("cuda"):
            return int(torch.cuda.memory_allocated())
        if device == "mps":
            return int(torch.mps.current_allocated_memory())
    except Exception:
        pass
    return 0


def _peak_bytes(device: str) -> int:
    """Return peak-allocated bytes since the last reset (best-effort)."""
    try:
        import torch
        if device.startswith("cuda"):
            return int(torch.cuda.max_memory_allocated())
        if device == "mps":
            # MPS doesn't track a separate peak counter; return current.
            return int(torch.mps.current_allocated_memory())
    except Exception:
        pass
    return 0


def _reset_peak(device: str) -> None:
    """Reset the peak-memory counter (CUDA only; no-op elsewhere)."""
    try:
        import torch
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _total_bytes(device: str) -> int:
    """Return total device memory in bytes (0 if unavailable)."""
    try:
        import torch
        if device.startswith("cuda"):
            return int(torch.cuda.get_device_properties(device).total_memory)
        if device == "mps":
            fn = getattr(torch.mps, "recommended_max_memory", None)
            if fn is not None:
                return int(fn())
    except Exception:
        pass
    return 0


# ── power-law fitting (log-log linear regression) ─────────────────────────────

def _fit_power_law(
    xs: "numpy.ndarray",
    ys: "numpy.ndarray",
    k: int,
) -> tuple[float, float]:
    """Fit ``y = a · x^k`` in log space.

    Returns ``(a, rmse_log)`` where *rmse_log* is the root-mean-square error
    of the log-transformed residuals — a scale-free quality metric.
    """
    import numpy as np

    log_x = np.log(xs.astype(float))
    log_y = np.log(ys.astype(float))
    # log y = log a + k · log x  →  solve for scalar log_a
    log_a = float(np.mean(log_y - k * log_x))
    a = math.exp(log_a)
    residuals = log_y - (log_a + k * log_x)
    rmse = math.sqrt(float(np.mean(residuals ** 2)))
    return a, rmse


# ── data record ───────────────────────────────────────────────────────────────

@dataclass
class MemRecord:
    pixel_area: int   # width × height
    batch_size: int   # number of images in the batch
    peak_bytes: int   # peak VRAM bytes during inference

    @property
    def bytes_per_image(self) -> float:
        return self.peak_bytes / max(1, self.batch_size)


# ── main class ────────────────────────────────────────────────────────────────

class MemoryAdvisor:
    """Adaptive batch-size advisor backed by empirical memory measurements.

    Usage in the worker::

        advisor = MemoryAdvisor(device=settings.device)

        # Before inference:
        advisor.reset_peak()

        # Run pipeline …

        # After inference:
        peak = advisor.capture_peak()
        advisor.record_success(pixel_area, batch_size, peak)

        # On OOM:
        advisor.record_oom(failed_batch_size)

        # When deciding batch size:
        max_bs = advisor.get_max_batch_size(pixel_area)
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._records: list[MemRecord] = []
        self._probe_batch_size: int = 1   # ramp up 1 → 2 → 3 … on success
        self._gen_count_at_current_batch_size: int = 0  # consecutive successes at this probe
        self._has_oomed: bool = False     # True after first OOM; switches to slow ramp-up
        self._max_batch_size_override: int = 0  # 0 = no override; >0 caps probe
        self._best_k: int = 1             # current best scaling exponent
        self._best_a: float = 0.0         # coefficient for best model

    # ── worker-facing API ─────────────────────────────────────────────────────

    def reset_peak(self) -> None:
        """Reset the device peak-memory counter.

        Call this immediately *before* launching pipeline inference so that
        ``capture_peak()`` reflects only the inference run itself.
        """
        _reset_peak(self._device)

    def capture_peak(self) -> int:
        """Read the peak device memory since the last ``reset_peak()`` call.

        Returns bytes, or 0 when the counter is not available (CPU, older MPS).
        """
        return _peak_bytes(self._device)

    def get_max_batch_size(self, pixel_area: int) -> int:
        """Return the recommended maximum batch size for *pixel_area*.

        Always ≥ 1.  Combines the probe ramp-up guard with the model-based
        VRAM prediction (the latter is only active once enough samples exist).
        If ``max_batch_size_override > 0`` it caps the result.
        """
        probe = self._probe_batch_size
        model_lim = self._model_limit(pixel_area)  # 0 = not enough data yet
        if model_lim > 0:
            result = max(1, min(probe, model_lim))
        else:
            result = max(1, probe)
        if self._max_batch_size_override > 0:
            result = min(result, self._max_batch_size_override)
        return result

    @property
    def max_batch_size_override(self) -> int:
        """User-set cap on batch size (0 = auto)."""
        return self._max_batch_size_override

    @max_batch_size_override.setter
    def max_batch_size_override(self, value: int) -> None:
        self._max_batch_size_override = max(0, value)

    def record_success(
        self,
        pixel_area: int,
        batch_size: int,
        peak_bytes: int,
    ) -> None:
        """Record a successful inference run.

        Advances the probe batch size by 1 only after >100 consecutive
        successful generations at the current probe size, updates the
        power-law model, and logs a fit report.
        """
        if peak_bytes > 0:
            self._records.append(MemRecord(pixel_area, batch_size, peak_bytes))
            if len(self._records) >= MIN_SAMPLES:
                self._refit()

        if batch_size >= self._probe_batch_size:
            old = self._probe_batch_size
            if not self._has_oomed:
                self._probe_batch_size = batch_size + 1
                logger.info(
                    "MemoryAdvisor: batch_size probe %d → %d  (rapid ramp-up, pre-OOM)",
                    old, self._probe_batch_size,
                )
            else:
                self._gen_count_at_current_batch_size += 1
                if self._gen_count_at_current_batch_size > 100:
                    self._probe_batch_size = batch_size + 1
                    logger.info(
                        "MemoryAdvisor: batch_size probe %d → %d  (after %d consecutive successes at %d)",
                        old, self._probe_batch_size,
                        self._gen_count_at_current_batch_size, old,
                    )
                    self._gen_count_at_current_batch_size = 0

        self._log_report(pixel_area, batch_size, peak_bytes)

    def record_oom(self, failed_batch_size: int) -> None:
        """Record an OOM failure; regresses the probe batch size.

        Also useful to call from the OOM handler before requeueing so that
        the next attempt will use a smaller batch.
        """
        self._has_oomed = True
        new_probe = max(1, failed_batch_size - 1)
        if new_probe < self._probe_batch_size:
            logger.warning(
                "MemoryAdvisor: OOM at batch_size=%d — lowering probe to %d",
                failed_batch_size,
                new_probe,
            )
            self._probe_batch_size = new_probe
        self._gen_count_at_current_batch_size = 0

    # ── internals ─────────────────────────────────────────────────────────────

    def _model_limit(self, pixel_area: int) -> int:
        """Max batch_size predicted by the fitted model (0 = unavailable)."""
        if len(self._records) < MIN_SAMPLES or self._best_a <= 0:
            return 0
        total = _total_bytes(self._device)
        if total <= 0:
            return 0
        budget = total * SAFETY_MARGIN
        predicted_per_image = self._best_a * (pixel_area ** self._best_k)
        if predicted_per_image <= 0:
            return 0
        return max(1, int(budget // predicted_per_image))

    def _refit(self) -> None:
        """Fit linear / quadratic / cubic power laws; keep the best one."""
        import numpy as np

        xs = np.array([r.pixel_area for r in self._records], dtype=float)
        ys = np.array([r.bytes_per_image for r in self._records], dtype=float)

        best_k, best_a, best_rmse = 1, 1.0, float("inf")
        for k in (1, 2, 3):
            try:
                a, rmse = _fit_power_law(xs, ys, k)
                if rmse < best_rmse:
                    best_rmse, best_k, best_a = rmse, k, a
            except Exception:
                pass

        self._best_k = best_k
        self._best_a = best_a

    def _log_report(
        self,
        pixel_area: int,
        batch_size: int,
        peak_bytes: int,
    ) -> None:
        """Emit a compact INFO log line with fit quality for all three laws."""
        import numpy as np

        n = len(self._records)
        measured_per_img = peak_bytes / max(1, batch_size) if peak_bytes > 0 else 0

        header = (
            f"MemoryAdvisor: n={n}  "
            f"pixels={pixel_area}  "
            f"batch={batch_size}  "
            f"peak={peak_bytes / 1e6:.1f} MB  "
            f"per_img={measured_per_img / 1e6:.1f} MB"
        )

        if n < MIN_SAMPLES:
            logger.info(
                "%s  (collecting samples, need %d more to fit)", header, MIN_SAMPLES - n
            )
            return

        xs = np.array([r.pixel_area for r in self._records], dtype=float)
        ys = np.array([r.bytes_per_image for r in self._records], dtype=float)

        law_names = {1: "linear", 2: "quadratic", 3: "cubic"}
        parts: list[str] = []
        for k in (1, 2, 3):
            try:
                a, rmse = _fit_power_law(xs, ys, k)
                predicted = a * (pixel_area ** k)
                marker = "★" if k == self._best_k else " "
                err_pct = (
                    100 * (predicted - measured_per_img) / measured_per_img
                    if measured_per_img > 0 else float("nan")
                )
                parts.append(
                    f"{marker}{law_names[k]}(k={k}): "
                    f"pred={predicted / 1e6:.1f}MB  "
                    f"err={err_pct:+.1f}%  "
                    f"rmse_log={rmse:.3f}"
                )
            except Exception:
                pass

        total = _total_bytes(self._device)
        model_lim = self._model_limit(pixel_area)
        footer = (
            f"best_law=k={self._best_k}  "
            f"probe={self._probe_batch_size}  "
            f"model_limit={model_lim}  "
            f"vram_total={total / 1e9:.2f} GB"
        )

        logger.info(
            "%s\n    %s\n    %s",
            header,
            "\n    ".join(parts),
            footer,
        )

