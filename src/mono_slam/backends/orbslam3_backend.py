"""ORB-SLAM3 backend adapter using the existing local wrapper."""

from __future__ import annotations

import time

import numpy as np

from mono_slam.backends.base import BaseSLAMBackend, SLAMBackendResult
from mono_slam.slam import SLAMSystem


class ORBSLAM3Backend(BaseSLAMBackend):
    """Adapter around the existing ORB-SLAM3 system."""

    name = "orb"

    def __init__(self, width: int, height: int, **kwargs):
        super().__init__(width, height, **kwargs)
        vocab_path = kwargs.get("vocab_path")
        settings_path = kwargs.get("settings_path")
        focal = kwargs.get("focal")
        self._slam = SLAMSystem(
            vocab_path=vocab_path,
            settings_path=settings_path,
            width=width,
            height=height,
            focal=focal,
        )

    def process(self, gray: np.ndarray, timestamp: float) -> SLAMBackendResult:
        t0 = time.perf_counter()
        result = self._slam.process(gray, timestamp)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        state_name = str(result.state).rsplit(".", 1)[-1]
        pose = self._slam.get_pose()
        points = self._slam.get_map_points()

        pts_np = None
        if points:
            # get_map_points() returns tuples of ((x,y,z), (u,v));
            # extract just the 3D world coordinates
            coords = []
            for p in points:
                if isinstance(p, (tuple, list)) and len(p) >= 2 and len(p[0]) >= 3:
                    coords.append(p[0][:3])
                elif hasattr(p, '__len__') and len(p) >= 3:
                    coords.append(p[:3])
            if coords:
                pts_np = np.array(coords, dtype=np.float32)

        return SLAMBackendResult(
            state=state_name,
            num_features_tracked=int(getattr(result, "num_features_tracked", 0)),
            num_features_detected=int(getattr(result, "num_features_detected", 0)),
            num_keyframes=int(getattr(result, "num_keyframes", 0)),
            num_map_points=int(getattr(result, "num_map_points", 0)),
            processing_time_ms=float(getattr(result, "processing_time_ms", elapsed_ms)),
            pose=np.asarray(pose) if pose is not None else None,
            points=pts_np,
        )

    def shutdown(self) -> None:
        self._slam.shutdown()
