from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_GROUNDING_DINO_PROMPTS = (
    "person",
    "pedestrian",
    "rider",
    "bicyclist",
    "bicycle",
    "motorcycle",
    "scooter",
    "car",
    "van",
    "pickup truck",
    "truck",
    "bus",
    "trailer",
    "caravan",
)



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
    grounding_model_id: str = "IDEA-Research/grounding-dino-tiny"
    grounding_device: str = "auto"
    grounding_local_files_only: bool = False
    grounding_prompts: tuple[str, ...] = DEFAULT_GROUNDING_DINO_PROMPTS
    grounding_box_threshold: float = 0.25
    grounding_text_threshold: float = 0.25
    grounding_nms_iou_threshold: float = 0.7
