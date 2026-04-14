from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.cubemap import erp_to_cubemap, save_cubemap_projection
from urpan_inpaint.discovery import run_indexing, write_sequence_manifest
from urpan_inpaint.erp import load_erp_rgb
from urpan_inpaint.models import FrameRecord, SequenceManifest


def project_frame_record(frame: FrameRecord, config: IndexConfig) -> FrameRecord:
    if not frame.file_exists:
        return replace(
            frame,
            erp_normalization_status="skipped_missing_source",
            erp_normalization_error="Referenced fixed frame is missing",
            cubemap_projection_status="skipped_missing_source",
            cubemap_projection_error="Referenced fixed frame is missing",
        )

    try:
        erp = load_erp_rgb(frame.resolved_fixed_path, compute_checksum=config.compute_checksums)
    except Exception as exc:
        error = str(exc)
        return replace(
            frame,
            erp_normalization_status="failed",
            erp_normalization_error=error,
            cubemap_projection_status="failed",
            cubemap_projection_error=error,
        )

    normalized_frame = replace(
        frame,
        erp_width=erp.width,
        erp_height=erp.height,
        erp_channels=erp.channels,
        erp_dtype=erp.dtype,
        erp_source_mode=erp.source_mode,
        erp_file_sha256=erp.file_sha256,
        erp_horizontal_wrap_mode="circular",
        erp_normalization_status="normalized",
        erp_normalization_error="",
    )

    try:
        projection = erp_to_cubemap(
            erp,
            face_size=config.cube_face_size,
            overlap_px=config.cube_overlap_px,
        )
        metadata_path = None
        if not config.dry_run and config.cache_cubemap_faces:
            metadata_path = save_cubemap_projection(projection, normalized_frame.cubemap_cache_dir)
    except Exception as exc:
        return replace(
            normalized_frame,
            cubemap_face_size=config.cube_face_size,
            cubemap_overlap_px=config.cube_overlap_px,
            cubemap_total_face_size=config.cube_face_size + (2 * config.cube_overlap_px),
            cubemap_projection_status="failed",
            cubemap_projection_error=str(exc),
        )

    return replace(
        normalized_frame,
        cubemap_face_size=config.cube_face_size,
        cubemap_overlap_px=config.cube_overlap_px,
        cubemap_total_face_size=projection.total_face_size,
        cubemap_face_cache_format="npz" if config.cache_cubemap_faces else "",
        cubemap_faces_cached=config.cache_cubemap_faces and not config.dry_run,
        cubemap_metadata_path=metadata_path,
        cubemap_projection_status="projected",
        cubemap_projection_error="",
    )


def project_sequence_cubemap(sequence_manifest: SequenceManifest, config: IndexConfig) -> SequenceManifest:
    if sequence_manifest.status != "ready":
        return sequence_manifest

    projected_rows = [project_frame_record(frame, config) for frame in sequence_manifest.rows]
    failed_rows = [row for row in projected_rows if row.cubemap_projection_status == "failed"]
    status = sequence_manifest.status
    failure_reason = sequence_manifest.failure_reason
    if failed_rows:
        status = "failed_cubemap_projection"
        failure_reason = f"{len(failed_rows)} frame(s) failed cubemap projection"

    projected_manifest = SequenceManifest(
        sequence_id=sequence_manifest.sequence_id,
        sequence_dir=sequence_manifest.sequence_dir,
        csv_path=sequence_manifest.csv_path,
        output_dir=sequence_manifest.output_dir,
        status=status,
        failure_reason=failure_reason,
        total_csv_rows=sequence_manifest.total_csv_rows,
        valid_frames=sequence_manifest.valid_frames,
        skipped_frames=sequence_manifest.skipped_frames,
        rows=projected_rows,
    )

    if not config.dry_run:
        write_sequence_manifest(projected_manifest)

    return projected_manifest


def run_cubemap_projection(
    config: IndexConfig,
    sequence_ids: Optional[Iterable[str]] = None,
) -> list[SequenceManifest]:
    indexed_manifests = run_indexing(config, sequence_ids=sequence_ids)
    return [project_sequence_cubemap(manifest, config) for manifest in indexed_manifests]
