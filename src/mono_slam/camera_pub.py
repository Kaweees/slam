"""
Camera Publisher — captures frames from webcam, video file, or comma device
and publishes them over Zenoh on topic "slam/camera/frame".

Each message is raw bytes: [8-byte float64 timestamp | 4-byte int32 height |
4-byte int32 width | 8-byte int64 sequence | raw grayscale pixel data].

Usage:
    python -m mono_slam.camera_pub                  # webcam
    python -m mono_slam.camera_pub video.mp4        # video file
    python -m mono_slam.camera_pub --width 640      # custom resolution
    python -m mono_slam.camera_pub --comma 192.168.1.10   # comma device
    python -m mono_slam.camera_pub --comma 192.168.1.10 --comma-camera wide  # wide road cam
"""

import argparse
import struct
import time

import cv2
import numpy as np
import zenoh

TOPIC = "slam/camera/frame"

COMMA_STREAM_MAP = {
    "road": "VISION_STREAM_ROAD",
    "wide": "VISION_STREAM_WIDE_ROAD",
    "driver": "VISION_STREAM_DRIVER",
}


def encode_frame(timestamp: float, frame: np.ndarray, seq: int) -> bytes:
    """Encode a grayscale frame into a compact binary message."""
    h, w = frame.shape[:2]
    header = struct.pack("<diiq", timestamp, h, w, int(seq))
    return header + frame.tobytes()


def _compute_scale(src_w: int, src_h: int, target_w: int):
    """Return (scale, w, h) for resizing."""
    if src_w > target_w:
        scale = target_w / src_w
        return scale, target_w, int(src_h * scale)
    return 1.0, src_w, src_h


def _run_comma(args, pub):
    """Capture loop for comma device via VisionIPC."""
    from cereal.visionipc import VisionIpcClient, VisionStreamType

    stream_name = COMMA_STREAM_MAP[args.comma_camera]
    stream_type = getattr(VisionStreamType, stream_name)

    print(f"Connecting to comma device at {args.comma} "
          f"(stream: {args.comma_camera})...")
    vipc = VisionIpcClient("camerad", stream_type, True, addr=args.comma)

    # wait for connection
    while not vipc.connect(True):
        time.sleep(0.1)
    vipc.recv()  # first frame to get dimensions

    src_w, src_h = vipc.width, vipc.height
    scale, w, h = _compute_scale(src_w, src_h, args.width)

    dt = 1.0 / args.fps
    seq = 0
    print(f"Publishing frames on '{TOPIC}' — {w}x{h} @ {args.fps} fps")
    print(f"Source: comma device {args.comma} ({args.comma_camera} camera)")
    print("Press Ctrl+C to stop")

    try:
        while True:
            t0 = time.monotonic()
            buf = vipc.recv()
            if buf is None:
                continue

            frame = np.frombuffer(buf.data, dtype=np.uint8).reshape(
                (src_h, src_w, 3))

            if scale != 1.0:
                frame = cv2.resize(frame, (w, h))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = buf.timestamp_eof / 1e9  # ns → seconds

            pub.put(encode_frame(timestamp, gray, seq))
            seq += 1

            if args.show:
                cv2.imshow("Camera Publisher", gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping publisher")
    finally:
        cv2.destroyAllWindows()


def _run_opencv(args, pub):
    """Capture loop for webcam / video file via OpenCV."""
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale, w, h = _compute_scale(src_w, src_h, args.width)

    dt = 1.0 / args.fps
    seq = 0
    print(f"Publishing frames on '{TOPIC}' — {w}x{h} @ {args.fps} fps")
    print(f"Source: {'webcam' if isinstance(source, int) else source}")
    print("Press Ctrl+C to stop")

    try:
        while cap.isOpened():
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, int):
                    continue  # transient webcam failure
                if args.loop and not isinstance(source, int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break  # end of video file

            if scale != 1.0:
                frame = cv2.resize(frame, (w, h))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = time.time()

            pub.put(encode_frame(timestamp, gray, seq))
            seq += 1

            if args.show:
                cv2.imshow("Camera Publisher", gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # rate-limit
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping publisher")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Camera frame publisher (Zenoh)")
    parser.add_argument("source", nargs="?", default="0",
                        help="Video source: device index (0) or file path "
                             "(ignored when --comma is set)")
    parser.add_argument("--width", type=int, default=640, help="Target frame width")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Max publish rate (fps)")
    parser.add_argument("--connect", type=str, default=None,
                        help="Zenoh router endpoint (e.g. tcp/localhost:7447)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop video file continuously (ignored for webcam)")
    parser.add_argument("--show", action="store_true",
                        help="Show local preview window (requires display)")
    parser.add_argument("--comma", type=str, default=None, metavar="ADDR",
                        help="Connect to comma device at this IP/hostname "
                             "(requires cereal package)")
    parser.add_argument("--comma-camera", type=str, default="road",
                        choices=list(COMMA_STREAM_MAP),
                        help="Which comma camera to use (default: road)")
    args = parser.parse_args()

    # open Zenoh session
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", f'["{args.connect}"]')
    session = zenoh.open(conf)
    pub = session.declare_publisher(TOPIC)

    try:
        if args.comma:
            _run_comma(args, pub)
        else:
            _run_opencv(args, pub)
    finally:
        pub.undeclare()
        session.close()


if __name__ == "__main__":
    main()
