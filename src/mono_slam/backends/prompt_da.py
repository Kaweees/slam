"""Prompt Depth Anything backend — ORB-SLAM3 tracking + dense depth completion.

Runs ORB-SLAM3 for visual odometry and map point extraction, then feeds the
sparse map points as a depth prompt into Prompt Depth Anything to produce a
dense metric depth map every N frames.

Requires:
    pip install monopriors torch transformers
    # or: uv pip install mono-slam[depth]

See https://github.com/rerun-io/prompt-da
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from mono_slam.backends.base import BaseSLAMBackend, SLAMBackendResult
from mono_slam.backends.orbslam3_backend import ORBSLAM3Backend
from mono_slam.depth_completion import DepthCompleter, sparse_points_to_prompt_depth


class PromptDABackend(BaseSLAMBackend):
    """ORB-SLAM3 tracking + Prompt Depth Anything dense depth completion."""

    name = "prompt-da"

    def __init__(self, width: int, height: int, **kwargs):
        super().__init__(width, height, **kwargs)

        # Underlying SLAM backend for tracking + map points
        self._slam = ORBSLAM3Backend(width=width, height=height, **kwargs)

        model_type = kwargs.get("depth_model", "large")
        device = kwargs.get("depth_device", "cuda")
        self._depth_every = int(kwargs.get("depth_every", 5))
        self._depth_completer = DepthCompleter(
            model_type=model_type, device=device,
        )

        self._focal = kwargs.get("focal") or float(width) * 0.55
        self._cx = width / 2.0
        self._cy = height / 2.0
        self._frame_count = 0
        self._last_depth_mm: np.ndarray | None = None

    def process(self, gray: np.ndarray, timestamp: float) -> SLAMBackendResult:
        t0 = time.perf_counter()

        # Run SLAM tracking
        result = self._slam.process(gray, timestamp)
        self._frame_count += 1

        # Run depth completion periodically when we have enough data
        depth_mm = None
        run_depth = (
            self._frame_count % self._depth_every == 0
            and result.pose is not None
            and result.points is not None
            and result.points.shape[0] > 10
        )

        if run_depth:
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            prompt_depth = sparse_points_to_prompt_depth(
                result.points[:, :3],
                result.pose,
                self._focal, self._focal,
                self._cx, self._cy,
                self.width, self.height,
            )
            depth_mm = self._depth_completer.predict(rgb, prompt_depth)
            self._last_depth_mm = depth_mm
        else:
            # Reuse last depth on intermediate frames
            depth_mm = self._last_depth_mm

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return SLAMBackendResult(
            state=result.state,
            num_features_tracked=result.num_features_tracked,
            num_features_detected=result.num_features_detected,
            num_keyframes=result.num_keyframes,
            num_map_points=result.num_map_points,
            processing_time_ms=elapsed_ms,
            pose=result.pose,
            points=result.points,
            depth_mm=depth_mm,
        )

    def shutdown(self) -> None:
        self._slam.shutdown()
