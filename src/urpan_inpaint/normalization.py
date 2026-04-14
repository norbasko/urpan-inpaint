from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.discovery import run_indexing, write_sequence_manifest
from urpan_inpaint.erp import load_erp_rgb
from urpan_inpaint.models import FrameRecord, SequenceManifest


def normalize_frame_record(frame: FrameRecord, compute_checksums: bool = True) -> FrameRecord:
    if not frame.file_exists:
        return replace(
            frame,
            erp_normalization_status="skipped_missing_source",
            erp_normalization_error="Referenced fixed frame is missing",
        )

    try:
        erp = load_erp_rgb(frame.resolved_fixed_path, compute_checksum=compute_checksums)
    except Exception as exc:
        return replace(
            frame,
            erp_normalization_status="failed",
            erp_normalization_error=str(exc),
        )

    return replace(
        frame,
        erp_width=erp.width,
        erp_height=erp.height,
        erp_channels=erp.channels,
        erp_dtype=erp.dtype,
        erp_source_mode=erp.source_mode,
        erp_file_sha256=erp.file_sha256,
        erp_normalization_status="normalized",
        erp_normalization_error="",
    )


def normalize_sequence_erp(sequence_manifest: SequenceManifest, config: IndexConfig) -> SequenceManifest:
    if sequence_manifest.status != "ready":
        return sequence_manifest

    normalized_rows = [
        normalize_frame_record(frame, compute_checksums=config.compute_checksums)
        for frame in sequence_manifest.rows
    ]
    failed_rows = [row for row in normalized_rows if row.erp_normalization_status == "failed"]

    status = sequence_manifest.status
    failure_reason = sequence_manifest.failure_reason
    if failed_rows:
        status = "failed_erp_normalization"
        failure_reason = f"{len(failed_rows)} frame(s) failed ERP normalization"

    normalized_manifest = SequenceManifest(
        sequence_id=sequence_manifest.sequence_id,
        sequence_dir=sequence_manifest.sequence_dir,
        csv_path=sequence_manifest.csv_path,
        output_dir=sequence_manifest.output_dir,
        status=status,
        failure_reason=failure_reason,
        total_csv_rows=sequence_manifest.total_csv_rows,
        valid_frames=sequence_manifest.valid_frames,
        skipped_frames=sequence_manifest.skipped_frames,
        rows=normalized_rows,
    )

    if not config.dry_run:
        write_sequence_manifest(normalized_manifest)

    return normalized_manifest


def run_erp_normalization(
    config: IndexConfig,
    sequence_ids: Optional[Iterable[str]] = None,
) -> list[SequenceManifest]:
    indexed_manifests = run_indexing(config, sequence_ids=sequence_ids)
    return [normalize_sequence_erp(manifest, config) for manifest in indexed_manifests]
