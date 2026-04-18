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

DEFAULT_SAM2_SEMANTIC_PROMPT_CLASSES = (
    "sky",
    "person",
    "rider",
    "bicycle",
    "motorcycle",
    "car",
    "truck",
    "bus",
    "train",
    "trailer",
    "building",
    "tree",
    "vegetation",
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
    sam2_model_id: str = "facebook/sam2.1-hiera-tiny"
    sam2_device: str = "auto"
    sam2_local_files_only: bool = False
    sam2_mask_threshold: float = 0.0
    sam2_min_mask_area_px: int = 16
    sam2_refine_grounding: bool = True
    sam2_refine_semantic: bool = True
    sam2_refine_roof: bool = True
    sam2_semantic_prompt_classes: tuple[str, ...] = DEFAULT_SAM2_SEMANTIC_PROMPT_CLASSES
    sam2_roof_box_fraction: float = 0.55
    sam2_roof_prior_margin_fraction: float = 0.15
    sam2_roof_temporal_window: int = 1
    sam2_roof_temporal_disagreement_iou_threshold: float = 0.4
    sky_mask_top_seed_fraction: float = 0.12
    sky_mask_sam2_boundary_margin_px: int = 3
    sky_mask_obstacle_dilation_px: int = 1
    sky_mask_erp_smoothing_iterations: int = 1
    dyn_min_component_area_px: int = 64
    roof_min_component_area_px: int = 256
    dyn_dilate_px: int = 3
    roof_dilate_px: int = 5
    dyn_erode_after_dilate_px: int = 0
    roof_erode_after_dilate_px: int = 0
    sam2_temporal_propagation: bool = True
    sam2_temporal_iou_threshold: float = 0.45
    sam2_temporal_area_ratio_min: float = 0.5
    sam2_temporal_area_ratio_max: float = 2.0
    sam2_temporal_max_gap: int = 1
