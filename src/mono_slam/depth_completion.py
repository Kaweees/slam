"""
Depth completion using Prompt Depth Anything.

Takes an RGB frame + sparse depth from SLAM map points and produces a dense
metric depth map. Requires ``monopriors`` (pip install monopriors) and a
CUDA-capable GPU.

Usage from CLI:
    uv run depth-complete --help

The module also exposes helpers that ``slam_sub`` can call when
``--depth-completion`` is passed.
"""

from __future__ import annotations

import numpy as np

# Prompt depth input resolution expected by the model
PROMPT_H, PROMPT_W = 192, 256


def sparse_points_to_prompt_depth(
    points_3d: np.ndarray,
    pose_cw: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Project 3D world points into a low-res prompt depth map (uint16, mm).

    Parameters
    ----------
    points_3d : (N, 3) float32 — map points in world frame.
    pose_cw   : (4, 4) float64 — world-to-camera transform (Tcw).
    fx, fy, cx, cy : camera intrinsics at *img_w × img_h*.
    img_w, img_h   : full-resolution image size.

    Returns
    -------
    prompt_depth : (PROMPT_H, PROMPT_W) uint16 — depth in millimetres,
                   zero where unknown.
    """
    R = pose_cw[:3, :3]
    t = pose_cw[:3, 3]

    # Transform to camera frame
    pts_cam = (R @ points_3d.T).T + t  # (N, 3)
    z = pts_cam[:, 2]

    # Keep only points in front of the camera
    valid = z > 0.01
    pts_cam = pts_cam[valid]
    z = z[valid]

    if len(z) == 0:
        return np.zeros((PROMPT_H, PROMPT_W), dtype=np.uint16)

    # Project to pixel coordinates at full resolution
    u = (fx * pts_cam[:, 0] / z + cx).astype(np.float32)
    v = (fy * pts_cam[:, 1] / z + cy).astype(np.float32)

    # Scale to prompt resolution
    sx = PROMPT_W / img_w
    sy = PROMPT_H / img_h
    pu = (u * sx).astype(np.int32)
    pv = (v * sy).astype(np.int32)

    in_bounds = (pu >= 0) & (pu < PROMPT_W) & (pv >= 0) & (pv < PROMPT_H)
    pu, pv, z = pu[in_bounds], pv[in_bounds], z[in_bounds]

    depth_mm = (z * 1000.0).astype(np.uint16)

    prompt = np.zeros((PROMPT_H, PROMPT_W), dtype=np.uint16)
    # If multiple points land on the same pixel, keep the closest
    for i in range(len(pu)):
        cur = prompt[pv[i], pu[i]]
        if cur == 0 or depth_mm[i] < cur:
            prompt[pv[i], pu[i]] = depth_mm[i]

    return prompt


class DepthCompleter:
    """Wraps Prompt Depth Anything for dense depth from RGB + sparse depth."""

    def __init__(self, model_type: str = "large", device: str = "cuda", max_size: int = 518):
        try:
            from monopriors.depth_completion_models.prompt_da import PromptDAPredictor
        except ImportError as exc:
            raise ImportError(
                "Depth completion requires 'monopriors'. Install with:\n"
                "  pip install monopriors\n"
                "See https://github.com/rerun-io/prompt-da for details."
            ) from exc

        self._predictor = PromptDAPredictor(
            device=device,
            model_type=model_type,
            max_size=max_size,
        )
        print(f"Prompt-DA loaded (model_type={model_type}, device={device})")

    def predict(self, rgb_hw3: np.ndarray, prompt_depth: np.ndarray) -> np.ndarray:
        """Run depth completion.

        Parameters
        ----------
        rgb_hw3      : (H, W, 3) uint8 RGB image.
        prompt_depth : (192, 256) uint16 depth in millimetres.

        Returns
        -------
        depth_mm : (H, W) uint16 dense metric depth in millimetres.
        """
        result = self._predictor(rgb=rgb_hw3, prompt_depth=prompt_depth)
        return result.depth_mm
