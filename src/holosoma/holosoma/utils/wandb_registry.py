"""WandB artifact download utilities for motion/terrain data and checkpoints.

Pure WandB SDK calls with no
framework dependencies.
"""

from __future__ import annotations

import pathlib

import wandb


def resolve_tag(registry_name: str, default_tag: str = "latest") -> str:
    """Ensure the registry name includes a tag. If not, append the default tag."""
    if ":" not in registry_name:
        registry_name = f"{registry_name}:{default_tag}"
    return registry_name


def download_motion_and_terrain(
    registry: str | None = None,
    artifact: wandb.Artifact | None = None,
) -> tuple[list[pathlib.Path] | None, list[pathlib.Path] | None]:
    """Download motion and terrain artifacts given a full registry name.

    If a registry name does not include a ``":"`` tag, ``":latest"`` is appended
    automatically.

    Returns ``(motion_paths, terrain_paths)`` or ``(None, None)`` if not provided
    or download failed.
    """

    def _download_from_registry(registry_name: str | None) -> pathlib.Path | None:
        if artifact is not None:
            art = artifact
        elif not registry_name:
            return None
        elif registry_name.startswith("file://"):
            # Local filesystem bypass: skip wandb and read directly from disk.
            local_path = pathlib.Path(registry_name[len("file://") :]).expanduser().resolve()
            if not local_path.exists():
                print(f"[WARN] Local registry path does not exist: {local_path}")
                return None
            print(f"[INFO] Using local registry directory: {local_path}")
            return local_path
        else:
            try:
                api = wandb.Api()
                art = api.artifact(registry_name)
            except Exception as e:
                print(f"[WARN] Failed to load artifact '{registry_name}': {e}")
                return None

        # Use artifact's cached directory if available, otherwise download once
        default_root = getattr(art, "_default_root", None)
        if default_root is not None and pathlib.Path(default_root()).exists():
            print(f"[INFO] Using cached artifact for '{registry_name}'")
            root = pathlib.Path(default_root())
        else:
            root = pathlib.Path(art.download())
        return root

    root = _download_from_registry(registry)

    motion: list[pathlib.Path] | None = None
    terrain: list[pathlib.Path] | None = None

    if root:
        motion_matches = sorted(
            [p for p in root.rglob("*") if "motion" in p.name.lower()],
            key=lambda p: p.name,
        )
        terrain_matches = sorted(
            [p for p in root.rglob("*") if "terrain" in p.name.lower()],
            key=lambda p: p.name,
        )
        motion = motion_matches if motion_matches else None
        terrain = terrain_matches if terrain_matches else None

    if motion:
        print(f"[INFO] Motion file/artifact: {motion}")
    else:
        print("[WARN] No motion artifact downloaded.")
    if terrain:
        print(f"[INFO] Terrain file/artifact: {terrain}")
    else:
        print("[WARN] No terrain artifact downloaded.")

    return motion, terrain


def get_wandb_run_and_file(wandb_path: str) -> tuple[wandb.apis.public.Run, str, str]:
    """Resolve a wandb path to a Run object and model filename.

    Parameters
    ----------
    wandb_path : str
        Either ``"entity/project/run_id"`` or
        ``"entity/project/run_id/model_xxx.pt"``.

    Returns
    -------
    tuple
        ``(run, run_path, file_name)``
    """
    api = wandb.Api()
    # Check if the last segment looks like a filename (has an extension)
    last_segment = wandb_path.rsplit("/", 1)[-1]
    has_explicit_file = "." in last_segment

    if has_explicit_file:
        run_path = "/".join(wandb_path.split("/")[:-1])
        file_name: str | None = last_segment
    else:
        run_path = wandb_path
        file_name = None

    run = api.run(run_path)

    if file_name is None:
        # Collect .pt checkpoint files and pick the one with the largest index
        files = [f.name for f in run.files() if f.name.endswith(".pt")]
        if not files:
            raise FileNotFoundError(
                f"No .pt checkpoint files found in run {run_path}. Available files: {[f.name for f in run.files()]}"
            )
        try:
            file_name = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        except Exception:
            # Fallback to lexicographic max if naming differs
            file_name = max(files)
    assert file_name is not None
    return run, run_path, file_name


def download_checkpoint_from_run(
    run: wandb.apis.public.Run,
    file_name: str,
    base_dir: str = "./logs/temp/checkpoints",
    replace: bool = True,
) -> pathlib.Path:
    """Download the given file from the run into ``base_dir/<run_id>/``."""
    run_id = run.id
    download_dir = pathlib.Path(base_dir) / run_id
    download_dir.mkdir(parents=True, exist_ok=True)

    wandb_file = run.file(str(file_name))
    wandb_file.download(str(download_dir), replace=replace)

    resume_path = download_dir / file_name
    if not resume_path.exists():
        raise FileNotFoundError(f"Downloaded checkpoint not found at {resume_path}")
    return resume_path


def download_checkpoint_motion_and_terrain(
    wandb_path: str,
    download_base: str = "./logs/temp/checkpoints",
) -> tuple[pathlib.Path | None, list[pathlib.Path] | None, list[pathlib.Path] | None]:
    """Download checkpoint, motion and terrain artifacts from a WandB run.

    Returns ``(resume_checkpoint_path, motion_paths, terrain_paths)``.
    """
    run, run_path, model_file = get_wandb_run_and_file(wandb_path)
    resume_path = download_checkpoint_from_run(run, model_file, base_dir=download_base)

    artifact = next((a for a in run.used_artifacts() if a.type == "terrains-motions"), None)

    if artifact is not None:
        motion, terrain = download_motion_and_terrain(registry=None, artifact=artifact)
    else:
        # No linked artifact — fall back to the registry_name stored in the run's config
        registry_name = (run.config or {}).get("training", {}).get("registry_name")
        if registry_name:
            registry_name = resolve_tag(registry_name)
            print(
                f"[INFO] No terrains-motions artifact linked; falling back to run config registry_name: {registry_name}"
            )
            motion, terrain = download_motion_and_terrain(registry=registry_name)
        else:
            motion, terrain = None, None

    print(f"[INFO] Loaded checkpoint: {resume_path}")
    if motion:
        print(f"[INFO] Motion file/artifact: {motion}")
    else:
        print("[WARN] No motion artifact found.")
    if terrain:
        print(f"[INFO] Terrain file/artifact: {terrain}")
    else:
        print("[WARN] No terrain artifact found.")

    return resume_path, motion, terrain


def download_checkpoint(
    wandb_path: str,
    download_base: str = "./logs/temp/checkpoints",
) -> pathlib.Path:
    """Download just the checkpoint from a WandB run."""
    run, _, model_file = get_wandb_run_and_file(wandb_path)
    return download_checkpoint_from_run(run, model_file, base_dir=download_base)
