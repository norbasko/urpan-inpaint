from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexConfig:
    dataset_root: Path = Path("/mnt/vision/data/kaust")
    output_root: Path = Path("/mnt/vision/data/kaust/inpaint")
    sequence_glob: str = "GS*"
    manifest_filename: str = "gps-fixed.csv"
    min_valid_frames: int = 3
    dry_run: bool = False
    compute_checksums: bool = True
    cube_face_size: int = 1536
    cube_overlap_px: int = 64
    cache_cubemap_faces: bool = True
    semantic_model_id: str = "facebook/mask2former-swin-tiny-cityscapes-semantic"
    semantic_device: str = "auto"
    semantic_local_files_only: bool = False
    semantic_save_logits: bool = False
    semantic_save_confidence: bool = True
    semantic_attempt_panoptic: bool = True
