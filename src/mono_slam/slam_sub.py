"""
SLAM Subscriber — subscribes to camera frames over Zenoh, runs ORB-SLAM3
monocular tracking, and streams results to Rerun web viewer.

Usage:
    uv run slam-sub                          # Rerun web viewer on :9090
    uv run slam-sub --web-port 8080          # custom port
    uv run slam-sub --vocab ORBvoc.txt       # custom vocabulary
"""

import argparse
import queue
import struct

import cv2
import numpy as np
import rerun as rr
import zenoh

from mono_slam.backends import AVAILABLE_BACKENDS, get_backend

TOPIC = "slam/camera/frame"
POSE_TOPIC = "slam/pose"

# Costmap parameters
COSTMAP_SIZE = 200         # grid cells per side
COSTMAP_RESOLUTION = 0.05  # meters per cell (5 cm) — adjust to match SLAM scale
COSTMAP_RADIUS = 3         # inflation radius in cells
COSTMAP_HALF = COSTMAP_SIZE // 2
# Only include points within this Y range (relative to camera height) as obstacles.
# Monocular SLAM has no metric scale, so these are in SLAM units. Tune as needed.
COSTMAP_Y_MIN = -0.5       # exclude points far below camera (floor plane noise)
COSTMAP_Y_MAX = 2.0        # exclude points far above camera (ceiling features)
# Maximum accumulated unique points to keep (deduplicated via voxel snap)
COSTMAP_MAX_PTS = 50_000


def build_costmap(pts: np.ndarray, cam_pos: np.ndarray | None = None) -> np.ndarray:
    """Project 3D map points onto XZ ground plane as a 2D costmap.

    The grid is re-centred on the camera position so the robot is always visible.
    Returns an RGB image: black=free, red=occupied (inflated), green=camera.
    """
    grid = np.zeros((COSTMAP_SIZE, COSTMAP_SIZE, 3), dtype=np.uint8)

    # Re-centre on camera so the robot stays in the middle of the map.
    origin_x = cam_pos[0] if cam_pos is not None else 0.0
    origin_z = cam_pos[2] if cam_pos is not None else 0.0

    # Filter by height to remove ceiling/floor noise before projecting.
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

    # Rerun expects RGB — red obstacles in channel 0 only.
    grid[:, :, 0] = occ

    # Mark camera position in green at the grid centre.
    if cam_pos is not None:
        cv2.circle(grid, (COSTMAP_HALF, COSTMAP_HALF), 3, (0, 255, 0), -1)

    return grid


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
    # Sequence id is currently used only for stream sanity checks.
    # It is decoded to keep message format compatibility explicit.
    _seq: int
    timestamp, h, w, _seq = struct.unpack("<diiq", payload[:header_size])
    pixels = np.frombuffer(payload[header_size:], dtype=np.uint8).reshape(h, w)
    return timestamp, pixels


def main():
    parser = argparse.ArgumentParser(description="SLAM subscriber (Zenoh + ORB-SLAM3 + Rerun)")
    parser.add_argument("--backend", type=str, default="orb",
                        help=f"SLAM backend: {', '.join(AVAILABLE_BACKENDS)} (default: orb)")
    parser.add_argument("--strict-backend", action="store_true",
                        help="Do not fallback to ORB if backend init/process fails")
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
                        help="Process every Nth frame while ORB is NOT_INITIALIZED (default: 3)")
    parser.add_argument("--focal", type=float, default=None,
                        help="Override focal length (default: auto-estimate from frame width)")
    parser.add_argument("--drop-nonmonotonic", action="store_true",
                        help="Drop frames whose timestamps go backwards or repeat")
    parser.add_argument("--depth-every", type=int, default=5,
                        help="Run depth completion every N frames for prompt-da backend (default: 5)")
    args = parser.parse_args()

    # init Rerun with gRPC server + web viewer
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

    # Set up 3D world coordinate system (right-handed, Y-down like OpenCV)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # open Zenoh session
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", f'["{args.connect}"]')
    session = zenoh.open(conf)

    # Use a queue so frames arriving during backend init are buffered, not lost
    frame_queue = queue.Queue()

    def _on_sample(sample):
        payload = sample.payload.to_bytes()
        timestamp, gray = decode_frame(payload)
        frame_queue.put((timestamp, gray))

    pose_pub = session.declare_publisher(POSE_TOPIC)

    print(f"Subscribing to '{TOPIC}' — waiting for frames...")
    print(f"Publishing poses on '{POSE_TOPIC}'")
    print(f"Selected backend: {args.backend}")
    sub = session.declare_subscriber(TOPIC, _on_sample)
    active_backend_name = args.backend

    backend = None
    frame_count = 0
    dropped_nonmonotonic = 0
    last_timestamp = None
    last_state_name = "NOT_INITIALIZED"
    accumulated_pts = np.empty((0, 3), dtype=np.float32)
    # Track the previous map-point count to detect ORB-SLAM3 map resets.
    # A reset drops the map to near-zero; flush accumulated_pts so ghost
    # points from the dead map don't pollute the costmap.
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

            # lazy-init SLAM from first frame's actual dimensions
            if backend is None:
                h, w = gray.shape[:2]
                print(f"First frame received: {w}x{h}, initializing '{args.backend}' backend...")
                buffered = frame_queue.qsize()
                backend_kwargs = dict(
                    width=w,
                    height=h,
                    vocab_path=args.vocab,
                    settings_path=args.settings,
                    focal=args.focal,
                    depth_every=args.depth_every,
                )
                try:
                    backend_cls = get_backend(active_backend_name)
                    backend = backend_cls(**backend_kwargs)
                except Exception as exc:
                    if active_backend_name != "orb" and not args.strict_backend:
                        print(
                            f"Backend '{active_backend_name}' failed to initialize: {exc}. "
                            "Falling back to 'orb'."
                        )
                        active_backend_name = "orb"
                        backend_cls = get_backend(active_backend_name)
                        backend = backend_cls(**backend_kwargs)
                    else:
                        raise
                buffered = frame_queue.qsize() - buffered
                if buffered > 0:
                    print(f"Buffered {buffered} frames during backend init")
                focal = args.focal if args.focal else float(w) * 0.55
                cx, cy = w / 2.0, h / 2.0

            should_process = True
            if frame_count > 0 and args.init_frame_step > 1:
                if "NOT_INITIALIZED" in last_state_name and (frame_count % args.init_frame_step) != 0:
                    should_process = False

            if not should_process:
                rr.set_time("frame", sequence=frame_count + 1)
                rr.set_time("timestamp", timestamp=timestamp)
                frame_count += 1
                continue

            try:
                result = backend.process(gray, timestamp)
            except RuntimeError as exc:
                if active_backend_name != "orb" and not args.strict_backend:
                    h, w = gray.shape[:2]
                    print(
                        f"Backend '{active_backend_name}' failed at runtime: {exc}. "
                        "Falling back to 'orb'."
                    )
                    backend.shutdown()
                    active_backend_name = "orb"
                    backend_cls = get_backend(active_backend_name)
                    backend = backend_cls(**backend_kwargs)
                    result = backend.process(gray, timestamp)
                else:
                    raise
            frame_count += 1

            rr.set_time("frame", sequence=frame_count)
            rr.set_time("timestamp", timestamp=timestamp)

            # log tracking state as text
            state_name = str(result.state)
            last_state_name = state_name
            rr.log("slam/state", rr.TextLog(
                f"{state_name} | features: {result.num_features_tracked}/{result.num_features_detected} "
                f"| keyframes: {result.num_keyframes} | points: {result.num_map_points} "
                f"| {result.processing_time_ms:.1f}ms"
            ))

            # log scalar metrics
            rr.log("slam/metrics/features_tracked", rr.Scalars(result.num_features_tracked))
            rr.log("slam/metrics/map_points", rr.Scalars(result.num_map_points))
            rr.log("slam/metrics/keyframes", rr.Scalars(result.num_keyframes))
            rr.log("slam/metrics/processing_ms", rr.Scalars(result.processing_time_ms))

            # log 3D map points
            pts = result.points
            if pts is not None:
                pts = np.asarray(pts, dtype=np.float32)
                if pts.ndim == 2 and pts.shape[1] >= 3:
                    rr.log("world/map_points", rr.Points3D(
                        pts[:, :3],
                        radii=0.01,
                        colors=np.full((len(pts), 3), [255, 255, 255], dtype=np.uint8),
                    ))

            # log camera pose as a 3D transform + pinhole so Rerun shows a
            # camera frustum in the 3D view with the current image projected
            pose = result.pose
            if pose is not None and pose.shape == (4, 4):
                R = pose[:3, :3]
                t = pose[:3, 3]
                # ORB-SLAM3 returns world-to-camera (Tcw); invert for camera-to-world
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

                # publish camera-to-world pose over Zenoh
                T_wc = np.eye(4, dtype=np.float64)
                T_wc[:3, :3] = R_wc
                T_wc[:3, 3] = t_wc
                pose_pub.put(encode_pose(timestamp, T_wc, state_name))

            # Detect an ORB-SLAM3 map reset: point count collapses to near-zero
            # after having had a substantial map. Clear stale ghost points.
            cur_num_map_points = result.num_map_points
            if last_num_map_points > 200 and cur_num_map_points < 50:
                accumulated_pts = np.empty((0, 3), dtype=np.float32)
                print(f"[{frame_count}] Map reset detected ({last_num_map_points} → "
                      f"{cur_num_map_points} points) — costmap cleared")
            last_num_map_points = cur_num_map_points

            # log bird's-eye costmap (XZ projection of map points)
            if pts is not None and pts.ndim == 2 and pts.shape[0] > 0:
                # Merge new points into accumulated set, deduplicate by voxel-snap
                # with numpy unique (O(N log N)) to avoid unbounded growth.
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

            # Log dense depth if backend provides it (e.g. prompt-da)
            if result.depth_mm is not None:
                rr.log("world/camera/depth", rr.DepthImage(result.depth_mm, meter=1000))

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
        sub.undeclare()
        pose_pub.undeclare()
        session.close()
        if dropped_nonmonotonic:
            print(f"Dropped {dropped_nonmonotonic} non-monotonic frames")
        print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
