"""SLAM backend registry — ORB-SLAM3 only."""

AVAILABLE_BACKENDS = ["orb"]


def get_backend(name: str = "orb"):
    """Import and return the ORB-SLAM3 backend class."""
    normalized = (name or "").strip().lower()

    if normalized in {"orb", "orbslam3"}:
        from mono_slam.backends.orbslam3_backend import ORBSLAM3Backend

        return ORBSLAM3Backend

    raise ValueError(
        f"Unknown backend '{name}'. Available: {AVAILABLE_BACKENDS}"
    )
