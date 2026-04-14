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
