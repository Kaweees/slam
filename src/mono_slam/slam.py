"""
SLAM System — wraps ORB-SLAM3 Python bindings for monocular visual SLAM.

Handles:
  - Auto-generation of ORB vocabulary and camera settings files
  - Frame processing via ORB-SLAM3
  - Map point and pose extraction
"""

import os
import tempfile

import cv2
import numpy as np
import orbslam3


def write_settings_yaml(path: str, width: int = 640, height: int = 480,
                        focal: float = None, fps: float = 30.0):
    """Write an OpenCV FileStorage YAML settings file for ORB-SLAM3."""
    if focal is None:
        # ~65° horizontal FOV, a reasonable default for dashcam/action camera
        focal = float(width) * 0.55
    cx = width / 2.0
    cy = height / 2.0

    # ORB-SLAM3 uses OpenCV's FileStorage YAML format (%YAML:1.0)
    # Both Camera.* (legacy) and Camera1.* (new) keys are provided
    content = f"""%YAML:1.0
---

Camera.type: "PinHole"

Camera.fx: {focal:.6f}
Camera.fy: {focal:.6f}
Camera.cx: {cx:.6f}
Camera.cy: {cy:.6f}

Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

Camera.width: {width}
Camera.height: {height}
Camera.fps: {fps:.1f}
Camera.RGB: 0

Camera1.fx: {focal:.6f}
Camera1.fy: {focal:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}

Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0

ORBextractor.nFeatures: 2000
ORBextractor.scaleFactor: 1.200000
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
    with open(path, "w") as f:
        f.write(content)



class SLAMSystem:
    """Monocular SLAM system backed by ORB-SLAM3."""

    def __init__(self, vocab_path: str | None = None, settings_path: str | None = None,
                 width: int = 640, height: int = 480, focal: float | None = None):
        self._tmpdir = tempfile.mkdtemp(prefix="mono_slam_")
        self._width = width
        self._height = height

        # resolve vocabulary file
        default_vocab = os.path.join(
            os.path.dirname(__file__), "..", "..", "vocab", "ORBvoc.txt"
        )
        if vocab_path and os.path.isfile(vocab_path):
            self._vocab_path = vocab_path
        elif os.path.isfile(default_vocab):
            self._vocab_path = os.path.abspath(default_vocab)
        else:
            raise FileNotFoundError(
                "ORB vocabulary not found. Download ORBvoc.txt from "
                "https://github.com/UZ-SLAMLab/ORB_SLAM3 and place it in vocab/"
            )

        # resolve settings file
        if settings_path and os.path.isfile(settings_path):
            self._settings_path = settings_path
        else:
            self._settings_path = os.path.join(self._tmpdir, "settings.yaml")
            write_settings_yaml(self._settings_path, width, height, focal=focal)
            if settings_path:
                print(f"Warning: settings file '{settings_path}' not found, "
                      "using generated defaults")

        # initialize ORB-SLAM3
        self._system = orbslam3.System(
            self._vocab_path,
            self._settings_path,
            orbslam3.Sensor.MONOCULAR,
        )
        self._system.set_use_viewer(False)
        self._system.initialize()

        self._frame_count = 0
        self._last_result = None
        print(f"ORB-SLAM3 initialized (monocular, {width}x{height})")

    def process(self, gray: np.ndarray, timestamp: float) -> orbslam3.TrackingResult:
        """Process a grayscale frame through ORB-SLAM3."""
        h, w = gray.shape[:2]
        if (w, h) != (self._width, self._height):
            gray = cv2.resize(gray, (self._width, self._height))

        result = self._system.process_mono_enhanced(gray, timestamp)
        self._frame_count += 1
        self._last_result = result
        return result

    def get_pose(self) -> np.ndarray | None:
        """Get the current camera pose as a 4x4 matrix, or None."""
        try:
            pose = self._system.get_frame_pose()
            if pose is not None:
                return np.array(pose)
        except Exception:
            pass
        return None

    def get_map_points(self) -> list:
        """Get current 3D map points."""
        try:
            return self._system.get_current_points()
        except Exception:
            return []

    def get_map_info(self) -> dict:
        """Get map statistics."""
        try:
            info = self._system.get_map_info()
            return {
                "num_keyframes": info.num_keyframes,
                "num_map_points": info.num_map_points,
                "coverage_area": info.coverage_area,
            }
        except Exception:
            return {"num_keyframes": 0, "num_map_points": 0, "coverage_area": 0.0}

    def shutdown(self):
        """Shut down the SLAM system."""
        try:
            self._system.shutdown()
        except Exception:
            pass
        print(f"SLAM shut down. Processed {self._frame_count} frames.")
