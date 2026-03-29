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

# Costmap parameters
COSTMAP_SIZE = 200       # grid cells per side
COSTMAP_RESOLUTION = 0.05  # meters per cell (5cm)
COSTMAP_RADIUS = 3       # inflation radius in cells
COSTMAP_HALF = COSTMAP_SIZE // 2


def build_costmap(pts: np.ndarray, cam_pos: np.ndarray | None = None) -> np.ndarray:
    """Project 3D map points onto XZ ground plane as a 2D costmap.

    Returns an RGB image: black=free, red=occupied (inflated), green=camera.
    """
    grid = np.zeros((COSTMAP_SIZE, COSTMAP_SIZE, 3), dtype=np.uint8)

    # Project points onto XZ plane, centred on the map origin
    gx = (pts[:, 0] / COSTMAP_RESOLUTION + COSTMAP_HALF).astype(np.int32)
    gz = (pts[:, 2] / COSTMAP_RESOLUTION + COSTMAP_HALF).astype(np.int32)

    mask = (gx >= 0) & (gx < COSTMAP_SIZE) & (gz >= 0) & (gz < COSTMAP_SIZE)
    gx, gz = gx[mask], gz[mask]

    # Build occupancy channel then inflate
    occ = np.zeros((COSTMAP_SIZE, COSTMAP_SIZE), dtype=np.uint8)
    occ[gz, gx] = 255
    if COSTMAP_RADIUS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * COSTMAP_RADIUS + 1, 2 * COSTMAP_RADIUS + 1),
        )
        occ = cv2.dilate(occ, kernel)

    # Red channel = cost
    grid[:, :, 2] = occ  # BGR: red is channel 2... but Rerun expects RGB
    grid[:, :, 0] = occ  # So put cost in R channel

    # Mark camera position in green
    if cam_pos is not None:
        cx = int(cam_pos[0] / COSTMAP_RESOLUTION + COSTMAP_HALF)
        cz = int(cam_pos[2] / COSTMAP_RESOLUTION + COSTMAP_HALF)
        if 0 <= cx < COSTMAP_SIZE and 0 <= cz < COSTMAP_SIZE:
            cv2.circle(grid, (cx, cz), 3, (0, 255, 0), -1)

    return grid


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

    print(f"Subscribing to '{TOPIC}' — waiting for frames...")
    print(f"Selected backend: {args.backend}")
    sub = session.declare_subscriber(TOPIC, _on_sample)
    active_backend_name = args.backend

    backend = None
    frame_count = 0
    dropped_nonmonotonic = 0
    last_timestamp = None
    last_state_name = "NOT_INITIALIZED"

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
                try:
                    backend_cls = get_backend(active_backend_name)
                    backend = backend_cls(
                        width=w,
                        height=h,
                        vocab_path=args.vocab,
                        settings_path=args.settings,
                        focal=args.focal,
                    )
                except Exception as exc:
                    if active_backend_name != "orb" and not args.strict_backend:
                        print(
                            f"Backend '{active_backend_name}' failed to initialize: {exc}. "
                            "Falling back to 'orb'."
                        )
                        active_backend_name = "orb"
                        backend_cls = get_backend(active_backend_name)
                        backend = backend_cls(
                            width=w,
                            height=h,
                            vocab_path=args.vocab,
                            settings_path=args.settings,
                            focal=args.focal,
                        )
                    else:
                        raise
                buffered = frame_queue.qsize() - buffered
                if buffered > 0:
                    print(f"Buffered {buffered} frames during backend init")
                # OS04C10 intrinsics: native 1344x760, focal=425.25
                focal = args.focal if args.focal else 425.25 * (float(w) / 1344.0)
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
                    backend = backend_cls(
                        width=w,
                        height=h,
                        vocab_path=args.vocab,
                        settings_path=args.settings,
                        focal=args.focal,
                    )
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

            # log bird's-eye costmap (XZ projection of map points)
            if pts is not None and pts.ndim == 2 and pts.shape[0] > 0:
                cam_pos = None
                if pose is not None and pose.shape == (4, 4):
                    cam_pos = -pose[:3, :3].T @ pose[:3, 3]
                costmap = build_costmap(pts[:, :3], cam_pos)
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
        session.close()
        if dropped_nonmonotonic:
            print(f"Dropped {dropped_nonmonotonic} non-monotonic frames")
        print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
