"""
SLAM Subscriber — subscribes to camera frames (and optionally IMU data) over
Zenoh, runs ORB-SLAM3 monocular or monocular-inertial tracking, and streams
results to Rerun web viewer.  Publishes camera-to-world poses on slam/pose.

Usage:
    uv run slam-sub                          # monocular, Rerun on :9090
    uv run slam-sub --imu                    # monocular-inertial (needs IMU on slam/imu)
    uv run slam-sub --vocab ORBvoc.txt       # custom vocabulary
"""

import argparse
import queue
import struct
import threading

import cv2
import numpy as np
import rerun as rr
import zenoh

from mono_slam.backends.orbslam3_backend import ORBSLAM3Backend

FRAME_TOPIC = "slam/camera/frame"
IMU_TOPIC = "slam/imu"
POSE_TOPIC = "slam/pose"

# Costmap parameters
COSTMAP_SIZE = 200         # grid cells per side
COSTMAP_RESOLUTION = 0.05  # meters per cell (5 cm)
COSTMAP_RADIUS = 3         # inflation radius in cells
COSTMAP_HALF = COSTMAP_SIZE // 2
COSTMAP_Y_MIN = -0.5
COSTMAP_Y_MAX = 2.0
COSTMAP_MAX_PTS = 50_000


# ---------------------------------------------------------------------------
# Message encoding / decoding
# ---------------------------------------------------------------------------

def encode_pose(timestamp: float, pose_cw: np.ndarray, state: str) -> bytes:
    """Encode a camera-to-world 4x4 pose into a binary message.

    Format: [8-byte float64 timestamp | 128-byte float64[16] row-major 4x4
    camera-to-world pose | UTF-8 state string].
    """
    header = struct.pack("<d", timestamp)
    return header + pose_cw.astype(np.float64).tobytes() + state.encode("utf-8")


def decode_frame(payload: bytes) -> tuple[float, np.ndarray]:
    """Decode a binary message back into (timestamp, grayscale frame)."""
    header_size = struct.calcsize("<diiq")
    timestamp, h, w, _seq = struct.unpack("<diiq", payload[:header_size])
    pixels = np.frombuffer(payload[header_size:], dtype=np.uint8).reshape(h, w)
    return timestamp, pixels


def decode_imu(payload: bytes) -> list[tuple[float, float, float, float, float, float, float]]:
    """Decode a binary IMU message into a list of (ax, ay, az, gx, gy, gz, t).

    Format: N repetitions of [7 x float64] = 56 bytes each.
    """
    sample_size = struct.calcsize("<7d")
    n = len(payload) // sample_size
    samples = []
    for i in range(n):
        offset = i * sample_size
        ax, ay, az, gx, gy, gz, t = struct.unpack(
            "<7d", payload[offset:offset + sample_size])
        samples.append((ax, ay, az, gx, gy, gz, t))
    return samples


# ---------------------------------------------------------------------------
# Costmap
# ---------------------------------------------------------------------------

def build_costmap(pts: np.ndarray, cam_pos: np.ndarray | None = None) -> np.ndarray:
    """Project 3D map points onto XZ ground plane as a 2D costmap."""
    grid = np.zeros((COSTMAP_SIZE, COSTMAP_SIZE, 3), dtype=np.uint8)

    origin_x = cam_pos[0] if cam_pos is not None else 0.0
    origin_z = cam_pos[2] if cam_pos is not None else 0.0

    if cam_pos is not None:
        cam_y = cam_pos[1]
        y_rel = pts[:, 1] - cam_y
        height_mask = (y_rel >= COSTMAP_Y_MIN) & (y_rel <= COSTMAP_Y_MAX)
        pts = pts[height_mask]

    if len(pts) == 0:
        return grid

    gx = ((pts[:, 0] - origin_x) / COSTMAP_RESOLUTION + COSTMAP_HALF).astype(np.int32)
    gz = ((pts[:, 2] - origin_z) / COSTMAP_RESOLUTION + COSTMAP_HALF).astype(np.int32)

    mask = (gx >= 0) & (gx < COSTMAP_SIZE) & (gz >= 0) & (gz < COSTMAP_SIZE)
    gx, gz = gx[mask], gz[mask]

    occ = np.zeros((COSTMAP_SIZE, COSTMAP_SIZE), dtype=np.uint8)
    occ[gz, gx] = 255
    if COSTMAP_RADIUS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * COSTMAP_RADIUS + 1, 2 * COSTMAP_RADIUS + 1),
        )
        occ = cv2.dilate(occ, kernel)

    grid[:, :, 0] = occ

    if cam_pos is not None:
        cv2.circle(grid, (COSTMAP_HALF, COSTMAP_HALF), 3, (0, 255, 0), -1)

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SLAM subscriber (Zenoh + ORB-SLAM3 + Rerun)")
    parser.add_argument("--imu", action="store_true",
                        help="Enable monocular-inertial mode (subscribes to slam/imu)")
    parser.add_argument("--vocab", type=str, default=None,
                        help="Path to ORB vocabulary file (ORBvoc.txt)")
    parser.add_argument("--settings", type=str, default=None,
                        help="Path to ORB-SLAM3 YAML settings file")
    parser.add_argument("--connect", type=str, default=None,
                        help="Zenoh router endpoint (e.g. tcp/localhost:7447)")
    parser.add_argument("--web-port", type=int, default=9090,
                        help="Rerun web viewer port (default: 9090)")
    parser.add_argument("--grpc-port", type=int, default=9876,
                        help="Rerun gRPC server port (default: 9876)")
    parser.add_argument("--web-connect", type=str, default=None,
                        help="Optional explicit rerun+http://... URI for web viewer connection")
    parser.add_argument("--rerun-connect-host", type=str, default="127.0.0.1",
                        help="Host embedded in rerun+http://<host>:<grpc>/proxy (default: 127.0.0.1)")
    parser.add_argument("--init-frame-step", type=int, default=3,
                        help="Process every Nth frame while NOT_INITIALIZED (default: 3)")
    parser.add_argument("--focal", type=float, default=None,
                        help="Override focal length (default: auto from comma camera intrinsics)")
    parser.add_argument("--drop-nonmonotonic", action="store_true",
                        help="Drop frames whose timestamps go backwards or repeat")
    args = parser.parse_args()

    # ---- Rerun setup ----
    rr.init("mono_slam", spawn=False)
    server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
    connect_uri = (
        args.web_connect
        if args.web_connect
        else f"rerun+http://{args.rerun_connect_host}:{args.grpc_port}/proxy"
    )
    rr.serve_web_viewer(open_browser=False, web_port=args.web_port)
    viewer_url = f"http://0.0.0.0:{args.web_port}/?url={connect_uri}"
    print(f"Rerun web viewer at {viewer_url}")
    print(f"Rerun gRPC server at {server_uri}")
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # ---- Zenoh setup ----
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", f'["{args.connect}"]')
    session = zenoh.open(conf)

    frame_queue: queue.Queue = queue.Queue()
    pose_pub = session.declare_publisher(POSE_TOPIC)

    def _on_frame(sample):
        payload = sample.payload.to_bytes()
        timestamp, gray = decode_frame(payload)
        frame_queue.put((timestamp, gray))

    # IMU buffer: holds samples between frames, protected by a lock
    imu_buffer: list = []
    imu_lock = threading.Lock()

    def _on_imu(sample):
        payload = sample.payload.to_bytes()
        samples = decode_imu(payload)
        with imu_lock:
            imu_buffer.extend(samples)

    mode = "monocular-inertial" if args.imu else "monocular"
    print(f"Mode: {mode}")
    print(f"Subscribing to '{FRAME_TOPIC}' — waiting for frames...")
    if args.imu:
        print(f"Subscribing to '{IMU_TOPIC}' for IMU data")
    print(f"Publishing poses on '{POSE_TOPIC}'")

    frame_sub = session.declare_subscriber(FRAME_TOPIC, _on_frame)
    imu_sub = session.declare_subscriber(IMU_TOPIC, _on_imu) if args.imu else None

    backend = None
    frame_count = 0
    dropped_nonmonotonic = 0
    last_timestamp = None
    last_state_name = "NOT_INITIALIZED"
    accumulated_pts = np.empty((0, 3), dtype=np.float32)
    last_num_map_points = 0

    try:
        while True:
            try:
                timestamp, gray = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if args.drop_nonmonotonic and last_timestamp is not None and timestamp <= last_timestamp:
                dropped_nonmonotonic += 1
                if dropped_nonmonotonic <= 5 or dropped_nonmonotonic % 30 == 0:
                    print(
                        "Dropping non-monotonic frame: "
                        f"ts={timestamp:.6f} <= last={last_timestamp:.6f} "
                        f"(dropped={dropped_nonmonotonic})"
                    )
                continue
            last_timestamp = timestamp

            # Drain IMU samples accumulated since last frame
            imu_samples = None
            if args.imu:
                with imu_lock:
                    if imu_buffer:
                        imu_samples = list(imu_buffer)
                        imu_buffer.clear()

            # Lazy-init backend from first frame dimensions
            if backend is None:
                h, w = gray.shape[:2]
                print(f"First frame received: {w}x{h}, initializing ORB-SLAM3 ({mode})...")
                backend = ORBSLAM3Backend(
                    width=w,
                    height=h,
                    vocab_path=args.vocab,
                    settings_path=args.settings,
                    focal=args.focal,
                    use_imu=args.imu,
                )
                from mono_slam.slam import NATIVE_FX, NATIVE_W
                focal = args.focal if args.focal else NATIVE_FX * (float(w) / NATIVE_W)
                cx, cy = w / 2.0, h / 2.0

            # Frame skipping during initialization
            should_process = True
            if frame_count > 0 and args.init_frame_step > 1:
                if "NOT_INITIALIZED" in last_state_name and (frame_count % args.init_frame_step) != 0:
                    should_process = False

            if not should_process:
                rr.set_time("frame", sequence=frame_count + 1)
                rr.set_time("timestamp", timestamp=timestamp)
                frame_count += 1
                continue

            result = backend.process(gray, timestamp,
                                     imu_measurements=imu_samples)
            frame_count += 1

            rr.set_time("frame", sequence=frame_count)
            rr.set_time("timestamp", timestamp=timestamp)

            # ---- Log tracking state ----
            state_name = str(result.state)
            last_state_name = state_name
            rr.log("slam/state", rr.TextLog(
                f"{state_name} | features: {result.num_features_tracked}/{result.num_features_detected} "
                f"| keyframes: {result.num_keyframes} | points: {result.num_map_points} "
                f"| {result.processing_time_ms:.1f}ms"
            ))

            # ---- Log scalar metrics ----
            rr.log("slam/metrics/features_tracked", rr.Scalars(result.num_features_tracked))
            rr.log("slam/metrics/map_points", rr.Scalars(result.num_map_points))
            rr.log("slam/metrics/keyframes", rr.Scalars(result.num_keyframes))
            rr.log("slam/metrics/processing_ms", rr.Scalars(result.processing_time_ms))

            # ---- Log 3D map points ----
            pts = result.points
            if pts is not None:
                pts = np.asarray(pts, dtype=np.float32)
                if pts.ndim == 2 and pts.shape[1] >= 3:
                    rr.log("world/map_points", rr.Points3D(
                        pts[:, :3],
                        radii=0.01,
                        colors=np.full((len(pts), 3), [255, 255, 255], dtype=np.uint8),
                    ))

            # ---- Log camera pose + publish over Zenoh ----
            pose = result.pose
            if pose is not None and pose.shape == (4, 4):
                R = pose[:3, :3]
                t = pose[:3, 3]
                R_wc = R.T
                t_wc = -R.T @ t
                rr.log("world/camera", rr.Transform3D(
                    translation=t_wc,
                    mat3x3=R_wc,
                ))
                rr.log("world/camera/image", rr.Pinhole(
                    focal_length=[focal, focal],
                    principal_point=[cx, cy],
                    resolution=[w, h],
                    camera_xyz=rr.ViewCoordinates.RDF,
                ))
                rr.log("world/camera/image", rr.Image(gray))

                T_wc = np.eye(4, dtype=np.float64)
                T_wc[:3, :3] = R_wc
                T_wc[:3, 3] = t_wc
                pose_pub.put(encode_pose(timestamp, T_wc, state_name))

            # ---- Costmap ----
            cur_num_map_points = result.num_map_points
            if last_num_map_points > 200 and cur_num_map_points < 50:
                accumulated_pts = np.empty((0, 3), dtype=np.float32)
                print(f"[{frame_count}] Map reset detected — costmap cleared")
            last_num_map_points = cur_num_map_points

            if pts is not None and pts.ndim == 2 and pts.shape[0] > 0:
                all_pts = (
                    np.vstack([accumulated_pts, pts[:, :3]])
                    if accumulated_pts.shape[0] > 0
                    else pts[:, :3].copy()
                )
                voxels = np.round(all_pts / COSTMAP_RESOLUTION).astype(np.int32)
                _, unique_idx = np.unique(voxels, axis=0, return_index=True)
                accumulated_pts = all_pts[unique_idx]
                if accumulated_pts.shape[0] > COSTMAP_MAX_PTS:
                    accumulated_pts = accumulated_pts[-COSTMAP_MAX_PTS:]
            if accumulated_pts.shape[0] > 0:
                cam_pos = None
                if pose is not None and pose.shape == (4, 4):
                    cam_pos = -pose[:3, :3].T @ pose[:3, 3]
                costmap = build_costmap(accumulated_pts, cam_pos)
                rr.log("costmap", rr.Image(costmap))

            if frame_count % 30 == 0:
                print(f"[{frame_count}] state={state_name} "
                      f"features={result.num_features_tracked}/{result.num_features_detected} "
                      f"keyframes={result.num_keyframes} points={result.num_map_points} "
                      f"({result.processing_time_ms:.1f}ms)")

    except KeyboardInterrupt:
        print("\nStopping subscriber")
    finally:
        if backend:
            backend.shutdown()
        frame_sub.undeclare()
        if imu_sub:
            imu_sub.undeclare()
        pose_pub.undeclare()
        session.close()
        if dropped_nonmonotonic:
            print(f"Dropped {dropped_nonmonotonic} non-monotonic frames")
        print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
