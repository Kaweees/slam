#!/usr/bin/env python3
"""
Prompt Depth Anything + SLAM example.

Runs the ``prompt-da`` backend (ORB-SLAM3 tracking + Prompt Depth Anything
dense depth completion) on a video and streams everything to Rerun web viewer:
RGB, 3D point cloud, camera frustum, dense depth, and costmap.

Based on https://github.com/rerun-io/prompt-da

Requirements:
    uv pip install mono-slam[depth]

Usage:
    python examples/prompt_da_slam.py videos/videoplayback.mp4
    python examples/prompt_da_slam.py videos/videoplayback.mp4 --depth-model large
    python examples/prompt_da_slam.py videos/videoplayback.mp4 --web-port 8080
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import rerun as rr

from mono_slam.backends import get_backend
from mono_slam.slam_sub import build_costmap


def main():
    parser = argparse.ArgumentParser(description="Prompt-DA + SLAM example (Rerun web)")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--depth-model", default="large", choices=["large"],
                        help="Prompt-DA model size (default: large)")
    parser.add_argument("--depth-every", type=int, default=5,
                        help="Run depth completion every N frames (default: 5)")
    parser.add_argument("--focal", type=float, default=None,
                        help="Focal length override (default: 0.55 * width)")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Max frames to process (default: 300)")
    parser.add_argument("--web-port", type=int, default=9090,
                        help="Rerun web viewer port (default: 9090)")
    parser.add_argument("--grpc-port", type=int, default=9876,
                        help="Rerun gRPC server port (default: 9876)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w > 640:
        scale = 640 / w
        w, h = 640, int(h * scale)
    else:
        scale = 1.0

    focal = args.focal if args.focal else w * 0.55
    cx, cy = w / 2.0, h / 2.0

    # Init Rerun web viewer
    rr.init("prompt_da_slam", spawn=False)
    server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
    connect_uri = f"rerun+http://127.0.0.1:{args.grpc_port}/proxy"
    rr.serve_web_viewer(open_browser=False, web_port=args.web_port)
    viewer_url = f"http://0.0.0.0:{args.web_port}/?url={connect_uri}"
    print(f"Rerun web viewer at {viewer_url}")
    print(f"Rerun gRPC server at {server_uri}")
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # Init prompt-da backend (ORB-SLAM3 + Prompt Depth Anything)
    print("Initializing prompt-da backend...")
    backend_cls = get_backend("prompt-da")
    backend = backend_cls(
        width=w, height=h,
        focal=focal,
        depth_model=args.depth_model,
        depth_every=args.depth_every,
    )

    frame_idx = 0
    ok_frames = 0
    t_start = time.time()

    print(f"Processing {args.video} ({w}x{h}, focal={focal:.0f})...")

    try:
        while cap.isOpened() and frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if scale != 1.0:
                frame = cv2.resize(frame, (w, h))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = frame_idx / 30.0

            result = backend.process(gray, timestamp)
            frame_idx += 1

            rr.set_time("frame", sequence=frame_idx)

            # Tracking state + metrics
            rr.log("slam/state", rr.TextLog(
                f"{result.state} | pts={result.num_map_points} | kf={result.num_keyframes}"
            ))
            rr.log("slam/metrics/map_points", rr.Scalars(result.num_map_points))
            rr.log("slam/metrics/processing_ms", rr.Scalars(result.processing_time_ms))

            # 3D map points
            pts = result.points
            if pts is not None and pts.ndim == 2 and pts.shape[1] >= 3:
                rr.log("world/map_points", rr.Points3D(
                    pts[:, :3], radii=0.01,
                    colors=np.full((len(pts), 3), [255, 255, 255], dtype=np.uint8),
                ))

            # Camera pose + image
            pose = result.pose
            if pose is not None and pose.shape == (4, 4):
                R_wc = pose[:3, :3].T
                t_wc = -pose[:3, :3].T @ pose[:3, 3]
                rr.log("world/camera", rr.Transform3D(
                    translation=t_wc, mat3x3=R_wc,
                ))
                rr.log("world/camera/image", rr.Pinhole(
                    focal_length=[focal, focal],
                    principal_point=[cx, cy],
                    resolution=[w, h],
                    camera_xyz=rr.ViewCoordinates.RDF,
                ))
                rr.log("world/camera/image", rr.Image(gray))
                ok_frames += 1

            # Dense depth (from Prompt-DA via backend)
            if result.depth_mm is not None:
                rr.log("world/camera/depth", rr.DepthImage(result.depth_mm, meter=1000))

            # Bird's-eye costmap
            if pts is not None and pts.ndim == 2 and pts.shape[0] > 0:
                cam_pos = None
                if pose is not None and pose.shape == (4, 4):
                    cam_pos = -pose[:3, :3].T @ pose[:3, 3]
                rr.log("costmap", rr.Image(build_costmap(pts[:, :3], cam_pos)))

            if frame_idx % 30 == 0:
                elapsed = time.time() - t_start
                has_depth = "+" if result.depth_mm is not None else "-"
                print(
                    f"[{frame_idx}] state={result.state} "
                    f"pts={result.num_map_points} ok={ok_frames} "
                    f"depth={has_depth} ({elapsed:.1f}s)"
                )

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        backend.shutdown()
        cap.release()

    elapsed = time.time() - t_start
    print(f"\nDone: {frame_idx} frames, {ok_frames} tracked, {elapsed:.1f}s")
    print(f"Viewer still running at {viewer_url} — Ctrl+C to exit")

    # Keep process alive so the web viewer stays up
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
