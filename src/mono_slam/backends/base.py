"""Common backend interface and result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SLAMBackendResult:
    """Normalized per-frame SLAM result used by the subscriber."""

    state: str
    num_features_tracked: int = 0
    num_features_detected: int = 0
    num_keyframes: int = 0
    num_map_points: int = 0
    processing_time_ms: float = 0.0
    pose: np.ndarray | None = None
    points: np.ndarray | None = None
    depth_mm: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class BaseSLAMBackend:
    """Base class for pluggable SLAM backends."""

    name = "base"

    def __init__(self, width: int, height: int, **kwargs: Any):
        self.width = int(width)
        self.height = int(height)
        self.kwargs = kwargs

    def process(self, gray: np.ndarray, timestamp: float) -> SLAMBackendResult:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


def format_missing_dependency_error(backend_name: str, package_name: str, details: str) -> str:
    """Format a user-facing dependency error for an optional backend."""
    return (
        f"{backend_name} backend is unavailable because '{package_name}' is not installed or failed to load. "
        f"{details}"
    )
