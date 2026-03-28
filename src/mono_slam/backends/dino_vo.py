"""DINO-VO backend wrapper (experimental)."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from mono_slam.backends.base import BaseSLAMBackend, SLAMBackendResult, format_missing_dependency_error


class DINOVOBackend(BaseSLAMBackend):
    """Experimental adapter for DINO-VO."""

    name = "dino"

    def __init__(self, width: int, height: int, **kwargs: Any):
        super().__init__(width, height, **kwargs)
        self._import_error = None

        try:
            importlib.import_module("torch")
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._import_error = exc

        if self._import_error is not None:
            raise RuntimeError(
                format_missing_dependency_error(
                    "DINO-VO",
                    "torch",
                    "Install the official DINO-VO codebase and its dependencies, then expose its Python package in this environment.",
                )
            ) from self._import_error

        self._reason = kwargs.get(
            "not_ready_reason",
            "Live-stream DINO-VO adapter requires project-specific model/runtime integration.",
        )

    def process(self, gray: np.ndarray, timestamp: float) -> SLAMBackendResult:
        _ = gray, timestamp
        raise RuntimeError(
            "DINO-VO backend is installed but not fully wired for this stream format. "
            f"{self._reason}"
        )

    def shutdown(self) -> None:
        pass
