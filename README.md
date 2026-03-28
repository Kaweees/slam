# slam-pipeline

```sh
uv run camera-pub
uv run camera-pub ./videos/drive.mp4
```

```sh
uv run slam-sub
uv run slam-sub --backend orb
uv run slam-sub --backend orb --drop-nonmonotonic --init-frame-step 3
uv run slam-sub --backend dpvo
uv run slam-sub --backend droid
uv run slam-sub --backend mast3r
uv run slam-sub --backend dino
uv run slam-sub --backend dpvo --strict-backend
```

## Remote / SSH Rerun

If you forward ports over SSH, keep both web and gRPC forwarded and use
the local loopback connect host in the subscriber:

```sh
uv run slam-sub --backend orb --web-port 9090 --grpc-port 9876 --rerun-connect-host 127.0.0.1
```

Important: run only one `camera-pub` process per topic. Multiple concurrent
publishers on `slam/camera/frame` can interleave frames and prevent
monocular initialization.

## Backends

- `orb`: ORB-SLAM3 backend (default, fully integrated)
- `dpvo`: DPVO/DPV-SLAM backend (experimental adapter)
- `droid`: DROID-SLAM backend (experimental adapter)
- `mast3r`: MASt3R-SLAM backend (experimental adapter)
- `dino`: DINO-VO backend (experimental adapter)

Experimental deep backends currently validate imports and reserve the shared API,
but require project-specific model/config wiring for live frame-by-frame tracking.

By default, if a selected deep backend fails to initialize or process frames,
the subscriber falls back to ORB-SLAM3 automatically. Use --strict-backend to
disable fallback and fail immediately.

```sh
uv run slam-sub --backend orb --drop-nonmonotonic --init-frame-step 5 --rerun-connect-host 127.0.0.1
```

```sh
uv run camera-pub ./videos/videoplayback.mp4 --fps 15 --loop
```
