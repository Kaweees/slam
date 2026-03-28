"""DPVO / DPV-SLAM backend wrapper.

This wrapper intentionally keeps a minimal streaming adapter surface so the rest of
this project can switch backends uniformly. Deep backend internals differ from
ORB-SLAM3 and may require per-model calibration and GPU setup.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from mono_slam.backends.base import (
    BaseSLAMBackend,
    SLAMBackendResult,
    format_missing_dependency_error,
)


class DPVOBackend(BaseSLAMBackend):
    """Experimental adapter for DPVO / DPV-SLAM."""

    name = "dpvo"

    def __init__(self, width: int, height: int, **kwargs: Any):
        super().__init__(width, height, **kwargs)

        self._torch = None
        self._dpvo = None
        self._import_error = None

        try:
            self._torch = importlib.import_module("torch")
            dpvo_mod = importlib.import_module("dpvo.dpvo")
            self._dpvo = getattr(dpvo_mod, "DPVO")
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._import_error = exc

        if self._dpvo is None:
            raise RuntimeError(
                format_missing_dependency_error(
                    "DPVO",
                    "dpvo",
                    "Install DPVO from https://github.com/princeton-vl/DPVO and ensure CUDA/PyTorch are configured.",
                )
            ) from self._import_error

        # Full live-stream integration requires model weights, intrinsics, and DPVO config.
        # Keep this explicit so users can supply project-specific settings.
        self._reason = kwargs.get(
            "not_ready_reason",
            "Live-stream DPVO adapter requires weights + calibration wiring for this pipeline.",
        )

    def process(self, gray: np.ndarray, timestamp: float) -> SLAMBackendResult:
        _ = gray, timestamp
        raise RuntimeError(
            "DPVO backend is installed but not fully wired for this stream format. "
            f"{self._reason}"
        )

    def shutdown(self) -> None:
        pass
