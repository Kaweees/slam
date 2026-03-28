"""SLAM backend registry and loader."""

AVAILABLE_BACKENDS = ["orb", "prompt-da", "dpvo", "droid", "mast3r", "dino"]


def get_backend(name: str):
    """Import and return the backend class by name."""
    normalized = (name or "").strip().lower()

    if normalized in {"orb", "orbslam3"}:
        from mono_slam.backends.orbslam3_backend import ORBSLAM3Backend

        return ORBSLAM3Backend

    if normalized in {"prompt-da", "promptda", "pda"}:
        from mono_slam.backends.prompt_da import PromptDABackend

        return PromptDABackend

    if normalized in {"dpvo", "dpv", "dpv-slam"}:
        from mono_slam.backends.dpvo import DPVOBackend

        return DPVOBackend

    if normalized in {"droid", "droid-slam"}:
        from mono_slam.backends.droid_slam import DROIDSLAMBackend

        return DROIDSLAMBackend

    if normalized in {"mast3r", "mast3r-slam"}:
        from mono_slam.backends.mast3r_slam import MASt3RSLAMBackend

        return MASt3RSLAMBackend

    if normalized in {"dino", "dino-vo"}:
        from mono_slam.backends.dino_vo import DINOVOBackend

        return DINOVOBackend

    raise ValueError(
        f"Unknown backend '{name}'. Available: {AVAILABLE_BACKENDS} "
        "(aliases: orbslam3, dpv, droid-slam, mast3r-slam, dino-vo)"
    )
