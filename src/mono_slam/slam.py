"""
SLAM System — wraps ORB-SLAM3 Python bindings for monocular and
monocular-inertial visual SLAM.

Handles:
  - Auto-generation of ORB vocabulary and camera settings files
  - Frame processing via ORB-SLAM3 (mono or mono-inertial)
  - Map point and pose extraction
"""

import os
import tempfile

import cv2
import numpy as np
import orbslam3


# Default IMU noise parameters for the comma device's BMI088 IMU.
# These are conservative starting values; tune from Allan variance if needed.
IMU_DEFAULTS = {
    "NoiseGyro": 1.7e-4,       # gyroscope noise density (rad/s/√Hz)
    "NoiseAcc": 2.0e-3,        # accelerometer noise density (m/s²/√Hz)
    "GyroWalk": 1.9e-5,        # gyroscope random walk (rad/s²/√Hz)
    "AccWalk": 3.0e-3,         # accelerometer random walk (m/s³/√Hz)
    "Frequency": 100,          # IMU sample rate (Hz)
}

# Camera-to-body (IMU) transform for comma body (mici).
# wideFromDeviceEuler ≈ [0, 0, 0] — wide camera is aligned with device frame.
# Identity rotation, small translation for physical offset between IMU and
# camera chip on the PCB.
TBC_DEFAULT = np.eye(4, dtype=np.float64)


# OS04C10 wide fisheye calibration at native 1344×760.
# Intrinsics from cv2.fisheye calibration (K matrix), distortion from D vector.
NATIVE_W, NATIVE_H = 1344, 760
NATIVE_FX, NATIVE_FY = 598.0, 598.0
NATIVE_CX, NATIVE_CY = 672.0, 380.0
NATIVE_K1 = -0.0143559
NATIVE_K2 = -0.00558797
NATIVE_K3 = 0.00237681
NATIVE_K4 = -0.00077131


def write_settings_yaml(path: str, width: int = 640, height: int = 480,
                        focal: float = None, fps: float = 20.0,
                        use_imu: bool = False):
    """Write an OpenCV FileStorage YAML settings file for ORB-SLAM3.

    Uses the KannalaBrandt8 fisheye model with OS04C10 wide camera
    calibration.  Intrinsics are scaled from native 1344×760 to the
    requested resolution.
    """
    scale = float(width) / NATIVE_W
    if focal is None:
        fx = NATIVE_FX * scale
        fy = NATIVE_FY * scale
    else:
        fx = fy = focal
    cx = NATIVE_CX * scale
    cy = NATIVE_CY * scale
    h = int(NATIVE_H * scale)

    # KannalaBrandt8 distortion coefficients are resolution-independent
    k1, k2, k3, k4 = NATIVE_K1, NATIVE_K2, NATIVE_K3, NATIVE_K4

    # ORB-SLAM3 uses OpenCV's FileStorage YAML format (%YAML:1.0)
    content = f"""%YAML:1.0
---

Camera.type: "KannalaBrandt8"

Camera.fx: {fx:.6f}
Camera.fy: {fy:.6f}
Camera.cx: {cx:.6f}
Camera.cy: {cy:.6f}

Camera.k1: {k1}
Camera.k2: {k2}
Camera.k3: {k3}
Camera.k4: {k4}

Camera.width: {width}
Camera.height: {h}
Camera.fps: {fps:.1f}
Camera.RGB: 0

Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}

Camera1.k1: {k1}
Camera1.k2: {k2}
Camera1.k3: {k3}
Camera1.k4: {k4}

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

    if use_imu:
        tbc = TBC_DEFAULT
        content += f"""
# --- IMU parameters (comma device BMI088) ---
IMU.NoiseGyro: {IMU_DEFAULTS['NoiseGyro']:.6e}
IMU.NoiseAcc: {IMU_DEFAULTS['NoiseAcc']:.6e}
IMU.GyroWalk: {IMU_DEFAULTS['GyroWalk']:.6e}
IMU.AccWalk: {IMU_DEFAULTS['AccWalk']:.6e}
IMU.Frequency: {IMU_DEFAULTS['Frequency']}

# Camera-to-body (IMU) extrinsic — wide cam aligned with device frame
IMU.T_b_c1: !!opencv-matrix
  rows: 4
  cols: 4
  dt: d
  data: [{tbc[0,0]}, {tbc[0,1]}, {tbc[0,2]}, {tbc[0,3]},
         {tbc[1,0]}, {tbc[1,1]}, {tbc[1,2]}, {tbc[1,3]},
         {tbc[2,0]}, {tbc[2,1]}, {tbc[2,2]}, {tbc[2,3]},
         {tbc[3,0]}, {tbc[3,1]}, {tbc[3,2]}, {tbc[3,3]}]
"""

    with open(path, "w") as f:
        f.write(content)


class SLAMSystem:
    """Monocular (or monocular-inertial) SLAM system backed by ORB-SLAM3."""

    def __init__(self, vocab_path: str | None = None, settings_path: str | None = None,
                 width: int = 640, height: int = 480, focal: float | None = None,
                 use_imu: bool = False):
        self._tmpdir = tempfile.mkdtemp(prefix="mono_slam_")
        self._width = width
        self._height = height
        self._use_imu = use_imu

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
            write_settings_yaml(self._settings_path, width, height,
                                focal=focal, use_imu=use_imu)
            if settings_path:
                print(f"Warning: settings file '{settings_path}' not found, "
                      "using generated defaults")

        sensor = (orbslam3.Sensor.IMU_MONOCULAR if use_imu
                  else orbslam3.Sensor.MONOCULAR)
        self._system = orbslam3.System(
            self._vocab_path,
            self._settings_path,
            sensor,
        )
        self._system.set_use_viewer(False)
        self._system.initialize()

        self._frame_count = 0
        self._last_result = None
        mode_str = "monocular-inertial" if use_imu else "monocular"
        print(f"ORB-SLAM3 initialized ({mode_str}, {width}x{height})")

    def process(self, gray: np.ndarray, timestamp: float,
                imu_measurements: list | None = None) -> orbslam3.TrackingResult:
        """Process a grayscale frame (with optional IMU data) through ORB-SLAM3.

        imu_measurements: list of (ax, ay, az, gx, gy, gz, t) tuples
            covering the interval since the previous frame. Only used when
            the system was initialized with use_imu=True.
        """
        h, w = gray.shape[:2]
        if (w, h) != (self._width, self._height):
            gray = cv2.resize(gray, (self._width, self._height))

        if self._use_imu and imu_measurements:
            result = self._system.process_mono_imu(
                gray, timestamp, imu_measurements)
        else:
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
