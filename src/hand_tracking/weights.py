"""Pinned model acquisition for the HumanEgo hand-tracking reproduction."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .models import HandTrackingConfig, HandTrackingError

HAMER_REPO = "Leo-TX/hamer"
VITPOSE_REPO = "JunkyByte/easy_ViTPose"
MEDIAPIPE_REPO = "Leo-TX/mediapipe-hand"
WILOR_REPO = "warmshao/WiLoR-mini"

_VITPOSE_FILENAMES = {
    "h": "torch/wholebody/vitpose-h-wholebody.pth",
    "l": "torch/wholebody/vitpose-l-wholebody.pth",
    "b": "torch/wholebody/vitpose-b-wholebody.pth",
    "s": "torch/wholebody/vitpose-s-wholebody.pth",
}


@dataclass(frozen=True, slots=True)
class HandTrackingWeights:
    """Resolved local files needed by detector and reconstruction backends."""

    hamer_root: Path
    hamer_checkpoint: Path
    hamer_model_config: Path
    mano_model: Path
    vitpose_model: Path | None
    yolo_model: Path | None
    mediapipe_model: Path | None
    manifest: Path


def ensure_hand_tracking_weights(config: HandTrackingConfig) -> HandTrackingWeights:
    """Materialize only the models selected by config and record their hashes."""

    root = config.model_directory
    hamer_root = root / "hamer"
    hamer_checkpoint = hamer_root / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    hamer_model_config = hamer_root / "hamer_ckpts" / "model_config.yaml"
    hamer_dataset_config = hamer_root / "hamer_ckpts" / "dataset_config.yaml"
    mano_mean_params = hamer_root / "data" / "mano_mean_params.npz"
    mano_model = hamer_root / "data" / "mano" / "MANO_RIGHT.pkl"

    required: list[tuple[str, str, str, Path]] = [
        (
            HAMER_REPO,
            "hamer.ckpt",
            config.sources.hamer_weights_revision,
            hamer_checkpoint,
        ),
        (
            HAMER_REPO,
            "model_config.yaml",
            config.sources.hamer_weights_revision,
            hamer_model_config,
        ),
        (
            HAMER_REPO,
            "dataset_config.yaml",
            config.sources.hamer_weights_revision,
            hamer_dataset_config,
        ),
        (
            HAMER_REPO,
            "mano_mean_params.npz",
            config.sources.hamer_weights_revision,
            mano_mean_params,
        ),
        (
            WILOR_REPO,
            "pretrained_models/MANO_RIGHT.pkl",
            config.sources.mano_weights_revision,
            mano_model,
        ),
    ]

    vitpose_model: Path | None = None
    yolo_model: Path | None = None
    if config.detector == "vitpose":
        vitpose_filename = _VITPOSE_FILENAMES[config.vitpose_variant]
        vitpose_model = root / "vitpose" / Path(vitpose_filename).name
        yolo_model = root / "vitpose" / "yolov8s.pt"
        required.extend(
            (
                (
                    VITPOSE_REPO,
                    vitpose_filename,
                    config.sources.vitpose_weights_revision,
                    vitpose_model,
                ),
                (
                    VITPOSE_REPO,
                    "yolov8/yolov8s.pt",
                    config.sources.vitpose_weights_revision,
                    yolo_model,
                ),
            )
        )

    mediapipe_model: Path | None = None
    if config.detector == "mediapipe" or config.fallback_detector == "mediapipe":
        mediapipe_model = root / "mediapipe" / "hand_landmarker.task"
        required.append(
            (
                MEDIAPIPE_REPO,
                "hand_landmarker.task",
                config.sources.mediapipe_weights_revision,
                mediapipe_model,
            )
        )

    missing = [destination for _, _, _, destination in required if not destination.is_file()]
    if missing and not config.download_models:
        rendered = ", ".join(str(path) for path in missing)
        raise HandTrackingError(f"hand tracking model files are missing: {rendered}")

    for repo_id, filename, revision, destination in required:
        if not destination.is_file():
            _download_file(repo_id, filename, revision, destination)

    manifest_path = root / "model-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "sources": config.sources.model_dump(mode="json"),
        "files": [
            {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "relative_path": destination.relative_to(root).as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
            for repo_id, filename, revision, destination in required
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return HandTrackingWeights(
        hamer_root=hamer_root,
        hamer_checkpoint=hamer_checkpoint,
        hamer_model_config=hamer_model_config,
        mano_model=mano_model,
        vitpose_model=vitpose_model,
        yolo_model=yolo_model,
        mediapipe_model=mediapipe_model,
        manifest=manifest_path,
    )


def _download_file(repo_id: str, filename: str, revision: str, destination: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download

        cached_path = Path(
            hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, destination)
    except Exception as exc:
        raise HandTrackingError(
            f"failed to download {repo_id}@{revision}:{filename}"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
