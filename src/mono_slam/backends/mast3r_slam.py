"""MASt3R-SLAM backend wrapper (experimental)."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from mono_slam.backends.base import BaseSLAMBackend, SLAMBackendResult, format_missing_dependency_error


class MASt3RSLAMBackend(BaseSLAMBackend):
    """Experimental adapter for MASt3R-SLAM."""

    name = "mast3r"

    def __init__(self, width: int, height: int, **kwargs: Any):
        super().__init__(width, height, **kwargs)
        self._import_error = None

        try:
            importlib.import_module("mast3r_slam")
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._import_error = exc

        if self._import_error is not None:
            raise RuntimeError(
                format_missing_dependency_error(
                    "MASt3R-SLAM",
                    "mast3r_slam",
                    "Install from the official MASt3R-SLAM repository and match its required PyTorch/CUDA versions.",
                )
            ) from self._import_error

        self._reason = kwargs.get(
            "not_ready_reason",
            "Live-stream MASt3R-SLAM adapter requires sequence-level optimization wiring.",
        )

    def process(self, gray: np.ndarray, timestamp: float) -> SLAMBackendResult:
        _ = gray, timestamp
        raise RuntimeError(
            "MASt3R-SLAM backend is installed but not fully wired for this stream format. "
            f"{self._reason}"
        )

    def shutdown(self) -> None:
        pass
