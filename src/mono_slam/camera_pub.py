"""
Camera Publisher — captures frames from webcam or video file and publishes
them over Zenoh on topic "slam/camera/frame".

Each message is raw bytes: [8-byte float64 timestamp | 4-byte int32 height |
4-byte int32 width | 8-byte int64 sequence | raw grayscale pixel data].

Usage:
    python -m mono_slam.camera_pub                  # webcam
    python -m mono_slam.camera_pub video.mp4        # video file
    python -m mono_slam.camera_pub --width 640      # custom resolution
"""

import argparse
import struct
import time

import cv2
import numpy as np
import zenoh

TOPIC = "slam/camera/frame"


def encode_frame(timestamp: float, frame: np.ndarray, seq: int) -> bytes:
    """Encode a grayscale frame into a compact binary message."""
    h, w = frame.shape[:2]
    header = struct.pack("<diiq", timestamp, h, w, int(seq))
    return header + frame.tobytes()


def main():
    parser = argparse.ArgumentParser(description="Camera frame publisher (Zenoh)")
    parser.add_argument("source", nargs="?", default="0",
                        help="Video source: device index (0) or file path")
    parser.add_argument("--width", type=int, default=640, help="Target frame width")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Max publish rate (fps)")
    parser.add_argument("--connect", type=str, default=None,
                        help="Zenoh router endpoint (e.g. tcp/localhost:7447)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop video file continuously (ignored for webcam)")
    parser.add_argument("--show", action="store_true",
                        help="Show local preview window (requires display)")
    args = parser.parse_args()

    # parse source — integer means device index
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if src_w > args.width:
        scale = args.width / src_w
        w = args.width
        h = int(src_h * scale)
    else:
        scale = 1.0
        w, h = src_w, src_h

    # open Zenoh session
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", f'["{args.connect}"]')
    session = zenoh.open(conf)
    pub = session.declare_publisher(TOPIC)

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
        pub.undeclare()
        session.close()


if __name__ == "__main__":
    main()
