from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.models import FrameRecord, REQUIRED_CSV_FIELDS, SequenceManifest


FRAME_NUMBER_RE = re.compile(r"frame-(\d+)$")


def discover_sequence_dirs(dataset_root: Path, sequence_glob: str = "GS*") -> list[Path]:
    return sorted(
        path
        for path in dataset_root.glob(sequence_glob)
        if path.is_dir() and path.name.startswith("GS")
    )


def filter_sequence_dirs(sequence_dirs: Iterable[Path], sequence_ids: Optional[Iterable[str]]) -> list[Path]:
    if not sequence_ids:
        return list(sequence_dirs)
    allowed = set(sequence_ids)
    return [path for path in sequence_dirs if path.name in allowed]


def sequence_output_dirs(output_root: Path, sequence_id: str) -> dict[str, Path]:
    base = output_root / sequence_id
    return {
        "base": base,
        "rgb": base / "rgb",
        "rgba": base / "rgba",
        "masks_dynamic": base / "masks" / "dynamic",
        "masks_roof": base / "masks" / "roof",
        "masks_sky": base / "masks" / "sky",
        "masks_inpaint": base / "masks" / "inpaint",
        "masks_union_debug": base / "masks" / "union_debug",
        "cubemap": base / "cubemap",
        "qa_overlays": base / "qa" / "overlays",
        "qa_contact_sheets": base / "qa" / "contact_sheets",
        "manifests": base / "manifests",
    }


def ensure_sequence_output_dirs(output_dirs: dict[str, Path]) -> None:
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)


def resolve_csv_frame_path(sequence_dir: Path, raw_path: str) -> Path:
    raw_path = raw_path.strip()
    source_path = Path(raw_path).expanduser()
    candidate = source_path if source_path.is_absolute() else (sequence_dir / source_path)
    if "fixed" in candidate.parts:
        return candidate.resolve(strict=False)
    return (sequence_dir / "fixed" / candidate.name).resolve(strict=False)


def parse_frame_number(frame_stem: str) -> Optional[int]:
    match = FRAME_NUMBER_RE.search(frame_stem)
    if not match:
        return None
    return int(match.group(1))


def build_output_paths(frame_stem: str, output_dirs: dict[str, Path]) -> dict[str, Path]:
    filename = f"{frame_stem}.png"
    return {
        "rgb_output_path": output_dirs["rgb"] / filename,
        "rgba_output_path": output_dirs["rgba"] / filename,
        "dynamic_mask_path": output_dirs["masks_dynamic"] / filename,
        "roof_mask_path": output_dirs["masks_roof"] / filename,
        "sky_mask_path": output_dirs["masks_sky"] / filename,
        "inpaint_mask_path": output_dirs["masks_inpaint"] / filename,
        "union_debug_path": output_dirs["masks_union_debug"] / filename,
        "overlay_output_path": output_dirs["qa_overlays"] / f"{frame_stem}.overlay.png",
        "cubemap_cache_dir": output_dirs["cubemap"] / frame_stem,
    }


def read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, fieldnames


def validate_required_fields(fieldnames: Iterable[str]) -> list[str]:
    normalized_fields = {name.strip() for name in fieldnames if name}
    return [field for field in REQUIRED_CSV_FIELDS if field not in normalized_fields]


def index_sequence(sequence_dir: Path, config: IndexConfig) -> SequenceManifest:
    sequence_id = sequence_dir.name
    output_dirs = sequence_output_dirs(config.output_root, sequence_id)
    csv_path = sequence_dir / config.manifest_filename

    if not csv_path.is_file():
        manifest = SequenceManifest(
            sequence_id=sequence_id,
            sequence_dir=sequence_dir,
            csv_path=csv_path,
            output_dir=output_dirs["base"],
            status="failed_missing_manifest",
            failure_reason=f"Missing {config.manifest_filename}",
            total_csv_rows=0,
            valid_frames=0,
            skipped_frames=0,
            rows=[],
        )
        if not config.dry_run:
            ensure_sequence_output_dirs(output_dirs)
            write_sequence_manifest(manifest)
        return manifest

    rows, fieldnames = read_csv_rows(csv_path)
    missing_fields = validate_required_fields(fieldnames)
    if missing_fields:
        manifest = SequenceManifest(
            sequence_id=sequence_id,
            sequence_dir=sequence_dir,
            csv_path=csv_path,
            output_dir=output_dirs["base"],
            status="failed_invalid_manifest",
            failure_reason=f"Missing required columns: {', '.join(missing_fields)}",
            total_csv_rows=len(rows),
            valid_frames=0,
            skipped_frames=len(rows),
            rows=[],
        )
        if not config.dry_run:
            ensure_sequence_output_dirs(output_dirs)
            write_sequence_manifest(manifest)
        return manifest

    manifest_rows: list[FrameRecord] = []
    valid_order_index = 0
    for csv_row_number, row in enumerate(rows):
        source_path = row["path"]
        resolved_fixed_path = resolve_csv_frame_path(sequence_dir, source_path)
        frame_name = resolved_fixed_path.name
        frame_stem = resolved_fixed_path.stem
        frame_number = parse_frame_number(frame_stem)
        file_exists = resolved_fixed_path.is_file()
        processing_status = "ready" if file_exists else "missing_source"
        skip_reason = "" if file_exists else "Referenced fixed frame is missing"
        output_paths = build_output_paths(frame_stem, output_dirs)

        manifest_rows.append(
            FrameRecord(
                sequence_id=sequence_id,
                csv_row_number=csv_row_number,
                source_index=row.get("", str(csv_row_number)),
                csv_path=csv_path,
                source_path=source_path,
                resolved_fixed_path=resolved_fixed_path,
                frame_name=frame_name,
                frame_stem=frame_stem,
                frame_number=frame_number,
                ts=row["ts"],
                lat=row["lat"],
                lon=row["lon"],
                alt=row["alt"],
                speed=row["speed"],
                speed3d=row["speed3d"],
                date=row["date"],
                file_exists=file_exists,
                processing_status=processing_status,
                skip_reason=skip_reason,
                valid_order_index=valid_order_index if file_exists else None,
                **output_paths,
            )
        )
        if file_exists:
            valid_order_index += 1

    valid_frames = sum(1 for row in manifest_rows if row.file_exists)
    skipped_frames = len(manifest_rows) - valid_frames
    status = "ready"
    failure_reason = None
    if valid_frames < config.min_valid_frames:
        status = "failed_insufficient_valid_frames"
        failure_reason = (
            f"Only {valid_frames} valid frames remain after validation; "
            f"minimum is {config.min_valid_frames}"
        )

    manifest = SequenceManifest(
        sequence_id=sequence_id,
        sequence_dir=sequence_dir,
        csv_path=csv_path,
        output_dir=output_dirs["base"],
        status=status,
        failure_reason=failure_reason,
        total_csv_rows=len(manifest_rows),
        valid_frames=valid_frames,
        skipped_frames=skipped_frames,
        rows=manifest_rows,
    )

    if not config.dry_run:
        ensure_sequence_output_dirs(output_dirs)
        write_sequence_manifest(manifest)

    return manifest


def write_sequence_manifest(manifest: SequenceManifest) -> None:
    output_dirs = sequence_output_dirs(manifest.output_dir.parent, manifest.sequence_id)
    ensure_sequence_output_dirs(output_dirs)

    frames_csv_path = output_dirs["manifests"] / "frames.csv"
    with frames_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(manifest.rows[0].to_manifest_row().keys()) if manifest.rows else [
            "sequence_id",
            "csv_row_number",
            "source_index",
            "csv_path",
            "source_path",
            "resolved_fixed_path",
            "frame_name",
            "frame_stem",
            "frame_number",
            "ts",
            "lat",
            "lon",
            "alt",
            "speed",
            "speed3d",
            "date",
            "file_exists",
            "processing_status",
            "skip_reason",
            "valid_order_index",
            "rgb_output_path",
            "rgba_output_path",
            "dynamic_mask_path",
            "roof_mask_path",
            "sky_mask_path",
            "inpaint_mask_path",
            "union_debug_path",
            "overlay_output_path",
            "cubemap_cache_dir",
            "erp_width",
            "erp_height",
            "erp_channels",
            "erp_dtype",
            "erp_source_mode",
            "erp_file_sha256",
            "erp_horizontal_wrap_mode",
            "erp_normalization_status",
            "erp_normalization_error",
            "cubemap_face_size",
            "cubemap_overlap_px",
            "cubemap_total_face_size",
            "cubemap_face_cache_format",
            "cubemap_faces_cached",
            "cubemap_metadata_path",
            "cubemap_projection_status",
            "cubemap_projection_error",
            "semantic_model_id",
            "semantic_output_dir",
            "semantic_has_panoptic",
            "semantic_has_confidence",
            "semantic_has_logits",
            "semantic_parse_status",
            "semantic_parse_error",
            "grounding_model_id",
            "grounding_output_dir",
            "grounding_box_count",
            "grounding_detect_status",
            "grounding_detect_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest.rows:
            writer.writerow(row.to_manifest_row())

    summary_path = output_dirs["manifests"] / "sequence_summary.json"
    summary = manifest.to_summary_dict()
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_indexing(config: IndexConfig, sequence_ids: Optional[Iterable[str]] = None) -> list[SequenceManifest]:
    sequence_dirs = discover_sequence_dirs(config.dataset_root, config.sequence_glob)
    selected_sequence_dirs = filter_sequence_dirs(sequence_dirs, sequence_ids)
    return [index_sequence(sequence_dir, config) for sequence_dir in selected_sequence_dirs]
