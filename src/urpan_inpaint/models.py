from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REQUIRED_CSV_FIELDS = (
    "path",
    "ts",
    "lat",
    "lon",
    "alt",
    "speed",
    "speed3d",
    "date",
)


@dataclass(frozen=True)
class FrameRecord:
    sequence_id: str
    csv_row_number: int
    source_index: str
    csv_path: Path
    source_path: str
    resolved_fixed_path: Path
    frame_name: str
    frame_stem: str
    frame_number: Optional[int]
    ts: str
    lat: str
    lon: str
    alt: str
    speed: str
    speed3d: str
    date: str
    file_exists: bool
    processing_status: str
    skip_reason: str
    valid_order_index: Optional[int]
    rgb_output_path: Path
    rgba_output_path: Path
    dynamic_mask_path: Path
    roof_mask_path: Path
    sky_mask_path: Path
    inpaint_mask_path: Path
    union_debug_path: Path
    overlay_output_path: Path
    cubemap_cache_dir: Path
    erp_width: Optional[int] = None
    erp_height: Optional[int] = None
    erp_channels: Optional[int] = None
    erp_dtype: str = ""
    erp_source_mode: str = ""
    erp_file_sha256: str = ""
    erp_horizontal_wrap_mode: str = "circular"
    erp_normalization_status: str = "pending"
    erp_normalization_error: str = ""
    cubemap_face_size: Optional[int] = None
    cubemap_overlap_px: Optional[int] = None
    cubemap_total_face_size: Optional[int] = None
    cubemap_face_cache_format: str = ""
    cubemap_faces_cached: bool = False
    cubemap_metadata_path: Optional[Path] = None
    cubemap_projection_status: str = "pending"
    cubemap_projection_error: str = ""
    semantic_model_id: str = ""
    semantic_output_dir: Optional[Path] = None
    semantic_has_panoptic: bool = False
    semantic_has_confidence: bool = False
    semantic_has_logits: bool = False
    semantic_parse_status: str = "pending"
    semantic_parse_error: str = ""
    grounding_model_id: str = ""
    grounding_output_dir: Optional[Path] = None
    grounding_box_count: Optional[int] = None
    grounding_detect_status: str = "pending"
    grounding_detect_error: str = ""

    def to_manifest_row(self) -> dict[str, str]:
        return {
            "sequence_id": self.sequence_id,
            "csv_row_number": str(self.csv_row_number),
            "source_index": self.source_index,
            "csv_path": str(self.csv_path),
            "source_path": self.source_path,
            "resolved_fixed_path": str(self.resolved_fixed_path),
            "frame_name": self.frame_name,
            "frame_stem": self.frame_stem,
            "frame_number": "" if self.frame_number is None else str(self.frame_number),
            "ts": self.ts,
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "speed": self.speed,
            "speed3d": self.speed3d,
            "date": self.date,
            "file_exists": "1" if self.file_exists else "0",
            "processing_status": self.processing_status,
            "skip_reason": self.skip_reason,
            "valid_order_index": "" if self.valid_order_index is None else str(self.valid_order_index),
            "rgb_output_path": str(self.rgb_output_path),
            "rgba_output_path": str(self.rgba_output_path),
            "dynamic_mask_path": str(self.dynamic_mask_path),
            "roof_mask_path": str(self.roof_mask_path),
            "sky_mask_path": str(self.sky_mask_path),
            "inpaint_mask_path": str(self.inpaint_mask_path),
            "union_debug_path": str(self.union_debug_path),
            "overlay_output_path": str(self.overlay_output_path),
            "cubemap_cache_dir": str(self.cubemap_cache_dir),
            "erp_width": "" if self.erp_width is None else str(self.erp_width),
            "erp_height": "" if self.erp_height is None else str(self.erp_height),
            "erp_channels": "" if self.erp_channels is None else str(self.erp_channels),
            "erp_dtype": self.erp_dtype,
            "erp_source_mode": self.erp_source_mode,
            "erp_file_sha256": self.erp_file_sha256,
            "erp_horizontal_wrap_mode": self.erp_horizontal_wrap_mode,
            "erp_normalization_status": self.erp_normalization_status,
            "erp_normalization_error": self.erp_normalization_error,
            "cubemap_face_size": "" if self.cubemap_face_size is None else str(self.cubemap_face_size),
            "cubemap_overlap_px": "" if self.cubemap_overlap_px is None else str(self.cubemap_overlap_px),
            "cubemap_total_face_size": "" if self.cubemap_total_face_size is None else str(self.cubemap_total_face_size),
            "cubemap_face_cache_format": self.cubemap_face_cache_format,
            "cubemap_faces_cached": "1" if self.cubemap_faces_cached else "0",
            "cubemap_metadata_path": "" if self.cubemap_metadata_path is None else str(self.cubemap_metadata_path),
            "cubemap_projection_status": self.cubemap_projection_status,
            "cubemap_projection_error": self.cubemap_projection_error,
            "semantic_model_id": self.semantic_model_id,
            "semantic_output_dir": "" if self.semantic_output_dir is None else str(self.semantic_output_dir),
            "semantic_has_panoptic": "1" if self.semantic_has_panoptic else "0",
            "semantic_has_confidence": "1" if self.semantic_has_confidence else "0",
            "semantic_has_logits": "1" if self.semantic_has_logits else "0",
            "semantic_parse_status": self.semantic_parse_status,
            "semantic_parse_error": self.semantic_parse_error,
            "grounding_model_id": self.grounding_model_id,
            "grounding_output_dir": "" if self.grounding_output_dir is None else str(self.grounding_output_dir),
            "grounding_box_count": "" if self.grounding_box_count is None else str(self.grounding_box_count),
            "grounding_detect_status": self.grounding_detect_status,
            "grounding_detect_error": self.grounding_detect_error,
        }


@dataclass
class SequenceManifest:
    sequence_id: str
    sequence_dir: Path
    csv_path: Path
    output_dir: Path
    status: str
    failure_reason: Optional[str]
    total_csv_rows: int
    valid_frames: int
    skipped_frames: int
    rows: list[FrameRecord]

    def to_summary_dict(self) -> dict[str, object]:
        first_valid = next((row.frame_name for row in self.rows if row.file_exists), None)
        last_valid = next((row.frame_name for row in reversed(self.rows) if row.file_exists), None)
        missing_frames = [row.frame_name for row in self.rows if not row.file_exists]
        normalized_frames = sum(1 for row in self.rows if row.erp_normalization_status == "normalized")
        normalization_failed_frames = [
            row.frame_name for row in self.rows if row.erp_normalization_status == "failed"
        ]
        projected_frames = sum(1 for row in self.rows if row.cubemap_projection_status == "projected")
        projection_failed_frames = [
            row.frame_name for row in self.rows if row.cubemap_projection_status == "failed"
        ]
        parsed_frames = sum(1 for row in self.rows if row.semantic_parse_status == "parsed")
        semantic_failed_frames = [
            row.frame_name for row in self.rows if row.semantic_parse_status == "failed"
        ]
        detected_frames = sum(1 for row in self.rows if row.grounding_detect_status == "detected")
        grounding_failed_frames = [
            row.frame_name for row in self.rows if row.grounding_detect_status == "failed"
        ]
        return {
            "sequence_id": self.sequence_id,
            "sequence_dir": str(self.sequence_dir),
            "csv_path": str(self.csv_path),
            "output_dir": str(self.output_dir),
            "status": self.status,
            "failure_reason": self.failure_reason,
            "total_csv_rows": self.total_csv_rows,
            "valid_frames": self.valid_frames,
            "skipped_frames": self.skipped_frames,
            "missing_frames": missing_frames,
            "first_valid_frame": first_valid,
            "last_valid_frame": last_valid,
            "normalized_frames": normalized_frames,
            "normalization_failed_frames": normalization_failed_frames,
            "projected_frames": projected_frames,
            "projection_failed_frames": projection_failed_frames,
            "parsed_frames": parsed_frames,
            "semantic_failed_frames": semantic_failed_frames,
            "detected_frames": detected_frames,
            "grounding_failed_frames": grounding_failed_frames,
            "required_csv_fields": list(REQUIRED_CSV_FIELDS),
        }
