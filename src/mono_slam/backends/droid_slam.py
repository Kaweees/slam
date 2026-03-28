"""DROID-SLAM backend wrapper (experimental)."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from mono_slam.backends.base import BaseSLAMBackend, SLAMBackendResult, format_missing_dependency_error


class DROIDSLAMBackend(BaseSLAMBackend):
    """Experimental adapter for DROID-SLAM."""

    name = "droid"

    def __init__(self, width: int, height: int, **kwargs: Any):
        super().__init__(width, height, **kwargs)
        self._import_error = None

        try:
            importlib.import_module("torch")
            importlib.import_module("droid_slam")
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._import_error = exc

        if self._import_error is not None:
            raise RuntimeError(
                format_missing_dependency_error(
                    "DROID-SLAM",
                    "droid_slam",
                    "Install from the official repository and build custom CUDA ops before selecting this backend.",
                )
            ) from self._import_error

        self._reason = kwargs.get(
            "not_ready_reason",
            "Live-stream DROID-SLAM adapter requires calibrated intrinsics and model/runtime config wiring.",
        )

    def process(self, gray: np.ndarray, timestamp: float) -> SLAMBackendResult:
        _ = gray, timestamp
        raise RuntimeError(
            "DROID-SLAM backend is installed but not fully wired for this stream format. "
            f"{self._reason}"
        )

    def shutdown(self) -> None:
        pass
