from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.discovery import run_indexing, sequence_output_dirs, write_sequence_manifest
from urpan_inpaint.models import FrameRecord, SequenceManifest


FRAME_METRIC_FIELDS = (
    "sequence_id",
    "frame_name",
    "frame_stem",
    "valid_order_index",
    "qa_status",
    "qa_error",
    "sky_ratio",
    "dynamic_ratio",
    "roof_ratio",
    "inpaint_ratio",
    "mask_components_dynamic",
    "mask_components_roof",
    "seam_delta_rgb",
    "seam_delta_alpha",
    "used_propainter",
    "used_lama_fallback",
    "processing_time_sec",
    "overlay_output_path",
)


def _load_rgb(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_rgba(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def _load_mask(path: Path, expected_shape: tuple[int, int], label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} mask: {path}")
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if mask.shape != expected_shape:
        raise ValueError(f"{label} mask shape {mask.shape} does not match ERP shape {expected_shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _mask_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0


def _count_components(mask: np.ndarray) -> int:
    foreground = mask > 0
    if not foreground.any():
        return 0

    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    components = 0
    start_y, start_x = np.nonzero(foreground)

    for y0_raw, x0_raw in zip(start_y, start_x):
        y0 = int(y0_raw)
        x0 = int(x0_raw)
        if visited[y0, x0]:
            continue
        components += 1
        visited[y0, x0] = True
        stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            y_min = max(0, y - 1)
            y_max = min(height, y + 2)
            x_min = max(0, x - 1)
            x_max = min(width, x + 2)
            for ny in range(y_min, y_max):
                for nx in range(x_min, x_max):
                    if foreground[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return components


def _seam_delta_rgb(rgb: np.ndarray) -> float:
    left = rgb[:, 0, :].astype(np.int16)
    right = rgb[:, -1, :].astype(np.int16)
    return float(np.abs(left - right).mean())


def _seam_delta_alpha(rgba: np.ndarray) -> float:
    left = rgba[:, 0, 3].astype(np.int16)
    right = rgba[:, -1, 3].astype(np.int16)
    return float(np.abs(left - right).mean())


def _has_png_artifacts(path: Optional[Path]) -> bool:
    if path is None or not path.is_dir():
        return False
    return any(path.rglob("*.png"))


def _infer_lama_fallback(frame: FrameRecord) -> bool:
    if frame.lama_status == "inpainted" or frame.single_frame_fallback_reason:
        return True
    if _has_png_artifacts(frame.lama_output_dir):
        return True
    return _has_png_artifacts(frame.cubemap_cache_dir / "lama_fallback")


def _infer_propainter(frame: FrameRecord, used_lama_fallback: bool) -> bool:
    if used_lama_fallback:
        return False
    if frame.propainter_status == "inpainted":
        return True
    if _has_png_artifacts(frame.propainter_output_dir):
        return True
    return _has_png_artifacts(frame.cubemap_cache_dir / "propainter")


def _sample_frame_indices(rows: list[FrameRecord], sample_count: int) -> set[int]:
    candidates = [index for index, row in enumerate(rows) if row.file_exists]
    if sample_count <= 0 or not candidates:
        return set()
    if len(candidates) <= sample_count:
        return set(candidates)

    selected = {
        candidates[int(round(position))]
        for position in np.linspace(0, len(candidates) - 1, sample_count)
    }
    if len(selected) < sample_count:
        for candidate in candidates:
            selected.add(candidate)
            if len(selected) >= sample_count:
                break
    return selected


def _resize_panel(image: Image.Image, panel_width: int) -> Image.Image:
    panel_width = max(1, panel_width)
    width, height = image.size
    if width == panel_width:
        return image
    panel_height = max(1, round(height * (panel_width / width)))
    return image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)


def _with_label(image: Image.Image, label: str) -> Image.Image:
    label_height = 24
    output = Image.new("RGB", (image.width, image.height + label_height), (18, 18, 18))
    output.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(output)
    draw.text((8, 5), label, fill=(245, 245, 245))
    return output


def _mask_panel(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    panel = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    panel[mask > 0] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(panel, mode="RGB")


def _checkerboard(shape: tuple[int, int], cell_px: int = 16) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices((height, width))
    checks = ((yy // cell_px) + (xx // cell_px)) % 2
    values = np.where(checks == 0, 220, 170).astype(np.uint8)
    return np.repeat(values[:, :, None], 3, axis=2)


def _rgba_preview(rgba: np.ndarray) -> Image.Image:
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    foreground = rgba[:, :, :3].astype(np.float32)
    background = _checkerboard(rgba.shape[:2]).astype(np.float32)
    preview = foreground * alpha + background * (1.0 - alpha)
    return Image.fromarray(np.clip(preview, 0, 255).astype(np.uint8), mode="RGB")


def _write_diagnostic_panel(
    frame: FrameRecord,
    original_rgb: np.ndarray,
    dynamic_mask: np.ndarray,
    roof_mask: np.ndarray,
    sky_mask: np.ndarray,
    inpaint_mask: np.ndarray,
    final_rgb: np.ndarray,
    final_rgba: np.ndarray,
    config: IndexConfig,
) -> None:
    panels = [
        ("Original RGB", Image.fromarray(original_rgb, mode="RGB")),
        ("Dynamic Mask", _mask_panel(dynamic_mask, (245, 74, 63))),
        ("Roof Mask", _mask_panel(roof_mask, (255, 191, 73))),
        ("Sky Mask", _mask_panel(sky_mask, (76, 157, 255))),
        ("Inpaint Mask", _mask_panel(inpaint_mask, (255, 102, 204))),
        ("Final RGB", Image.fromarray(final_rgb, mode="RGB")),
        ("RGBA Preview", _rgba_preview(final_rgba)),
    ]
    labelled = [
        _with_label(_resize_panel(panel, config.qa_diagnostic_panel_width_px), label)
        for label, panel in panels
    ]
    height = max(panel.height for panel in labelled)
    width = sum(panel.width for panel in labelled)
    contact = Image.new("RGB", (width, height), (18, 18, 18))
    x_offset = 0
    for panel in labelled:
        contact.paste(panel, (x_offset, 0))
        x_offset += panel.width

    frame.overlay_output_path.parent.mkdir(parents=True, exist_ok=True)
    contact.save(frame.overlay_output_path, format="PNG")


def _empty_metric_row(frame: FrameRecord, status: str, error: str, elapsed: float) -> dict[str, object]:
    used_lama_fallback = _infer_lama_fallback(frame)
    used_propainter = _infer_propainter(frame, used_lama_fallback)
    return {
        "sequence_id": frame.sequence_id,
        "frame_name": frame.frame_name,
        "frame_stem": frame.frame_stem,
        "valid_order_index": "" if frame.valid_order_index is None else frame.valid_order_index,
        "qa_status": status,
        "qa_error": error,
        "sky_ratio": "",
        "dynamic_ratio": "",
        "roof_ratio": "",
        "inpaint_ratio": "",
        "mask_components_dynamic": "",
        "mask_components_roof": "",
        "seam_delta_rgb": "",
        "seam_delta_alpha": "",
        "used_propainter": "1" if used_propainter else "0",
        "used_lama_fallback": "1" if used_lama_fallback else "0",
        "processing_time_sec": f"{elapsed:.8f}",
        "overlay_output_path": str(frame.overlay_output_path),
    }


def _measure_frame(
    frame_index: int,
    frame: FrameRecord,
    sampled_frame_indices: set[int],
    config: IndexConfig,
) -> tuple[FrameRecord, dict[str, object]]:
    started_at = time.perf_counter()
    if not frame.file_exists:
        elapsed = time.perf_counter() - started_at
        used_lama_fallback = _infer_lama_fallback(frame)
        used_propainter = _infer_propainter(frame, used_lama_fallback)
        return (
            replace(
                frame,
                qa_used_propainter=used_propainter,
                qa_used_lama_fallback=used_lama_fallback,
                qa_status="skipped",
                qa_error=frame.skip_reason,
                qa_processing_time_sec=elapsed,
            ),
            _empty_metric_row(frame, "skipped", frame.skip_reason, elapsed),
        )

    try:
        final_rgb = _load_rgb(frame.rgb_output_path, "final RGB output")
        final_rgba = _load_rgba(frame.rgba_output_path, "final RGBA output")
        original_rgb = _load_rgb(frame.resolved_fixed_path, "source ERP frame")
        expected_shape = final_rgb.shape[:2]
        if final_rgba.shape[:2] != expected_shape:
            raise ValueError(f"RGBA shape {final_rgba.shape[:2]} does not match RGB shape {expected_shape}")
        if original_rgb.shape != final_rgb.shape:
            raise ValueError(f"source RGB shape {original_rgb.shape} does not match final RGB shape {final_rgb.shape}")

        dynamic_mask = _load_mask(frame.dynamic_mask_path, expected_shape, "DYN")
        roof_mask = _load_mask(frame.roof_mask_path, expected_shape, "ROOF")
        sky_mask = _load_mask(frame.sky_mask_path, expected_shape, "SKY")
        inpaint_mask = _load_mask(frame.inpaint_mask_path, expected_shape, "INPAINT")

        used_lama_fallback = _infer_lama_fallback(frame)
        used_propainter = _infer_propainter(frame, used_lama_fallback)
        sky_ratio = _mask_ratio(sky_mask)
        dynamic_ratio = _mask_ratio(dynamic_mask)
        roof_ratio = _mask_ratio(roof_mask)
        inpaint_ratio = _mask_ratio(inpaint_mask)
        dynamic_components = _count_components(dynamic_mask)
        roof_components = _count_components(roof_mask)
        seam_rgb = _seam_delta_rgb(final_rgb)
        seam_alpha = _seam_delta_alpha(final_rgba)

        if frame_index in sampled_frame_indices and not config.dry_run:
            _write_diagnostic_panel(
                frame=frame,
                original_rgb=original_rgb,
                dynamic_mask=dynamic_mask,
                roof_mask=roof_mask,
                sky_mask=sky_mask,
                inpaint_mask=inpaint_mask,
                final_rgb=final_rgb,
                final_rgba=final_rgba,
                config=config,
            )

        elapsed = time.perf_counter() - started_at
        propainter_output_dir = frame.propainter_output_dir
        lama_output_dir = frame.lama_output_dir
        propainter_status = frame.propainter_status
        lama_status = frame.lama_status
        if used_propainter and propainter_output_dir is None:
            propainter_output_dir = frame.cubemap_cache_dir / "propainter"
        if used_lama_fallback and lama_output_dir is None:
            lama_output_dir = frame.cubemap_cache_dir / "lama_fallback"
        if used_propainter and propainter_status == "pending":
            propainter_status = "inpainted"
        if used_lama_fallback and lama_status == "pending":
            lama_status = "inpainted"
        if used_lama_fallback and propainter_status == "pending":
            propainter_status = "fallback_lama"

        updated_frame = replace(
            frame,
            erp_width=final_rgb.shape[1],
            erp_height=final_rgb.shape[0],
            erp_channels=final_rgb.shape[2],
            propainter_output_dir=propainter_output_dir,
            propainter_status=propainter_status,
            lama_output_dir=lama_output_dir,
            lama_status=lama_status,
            qa_sky_ratio=sky_ratio,
            qa_dynamic_ratio=dynamic_ratio,
            qa_roof_ratio=roof_ratio,
            qa_inpaint_ratio=inpaint_ratio,
            qa_mask_components_dynamic=dynamic_components,
            qa_mask_components_roof=roof_components,
            qa_seam_delta_rgb=seam_rgb,
            qa_seam_delta_alpha=seam_alpha,
            qa_used_propainter=used_propainter,
            qa_used_lama_fallback=used_lama_fallback,
            qa_processing_time_sec=elapsed,
            qa_status="measured",
            qa_error="",
        )
        metric_row = {
            "sequence_id": frame.sequence_id,
            "frame_name": frame.frame_name,
            "frame_stem": frame.frame_stem,
            "valid_order_index": "" if frame.valid_order_index is None else frame.valid_order_index,
            "qa_status": "measured",
            "qa_error": "",
            "sky_ratio": f"{sky_ratio:.8f}",
            "dynamic_ratio": f"{dynamic_ratio:.8f}",
            "roof_ratio": f"{roof_ratio:.8f}",
            "inpaint_ratio": f"{inpaint_ratio:.8f}",
            "mask_components_dynamic": dynamic_components,
            "mask_components_roof": roof_components,
            "seam_delta_rgb": f"{seam_rgb:.8f}",
            "seam_delta_alpha": f"{seam_alpha:.8f}",
            "used_propainter": "1" if used_propainter else "0",
            "used_lama_fallback": "1" if used_lama_fallback else "0",
            "processing_time_sec": f"{elapsed:.8f}",
            "overlay_output_path": str(frame.overlay_output_path),
        }
        return updated_frame, metric_row
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        error = str(exc)
        used_lama_fallback = _infer_lama_fallback(frame)
        used_propainter = _infer_propainter(frame, used_lama_fallback)
        return (
            replace(
                frame,
                qa_used_propainter=used_propainter,
                qa_used_lama_fallback=used_lama_fallback,
                qa_status="failed",
                qa_error=error,
                qa_processing_time_sec=elapsed,
            ),
            _empty_metric_row(frame, "failed", error, elapsed),
        )


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _percentiles(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    data = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
    }


def _count_stage_failures(rows: list[FrameRecord]) -> dict[str, int]:
    return {
        "erp_normalization": sum(1 for row in rows if row.erp_normalization_status == "failed"),
        "cubemap_projection": sum(1 for row in rows if row.cubemap_projection_status == "failed"),
        "semantic_parse": sum(1 for row in rows if row.semantic_parse_status == "failed"),
        "grounding_detect": sum(1 for row in rows if row.grounding_detect_status == "failed"),
        "sam2_refine": sum(1 for row in rows if row.sam2_refine_status == "failed"),
        "mask_fusion": sum(1 for row in rows if row.mask_fusion_status == "failed"),
        "propainter": sum(1 for row in rows if row.propainter_status == "failed"),
        "lama": sum(1 for row in rows if row.lama_status == "failed"),
        "qa": sum(1 for row in rows if row.qa_status == "failed"),
    }


def _sequence_metrics(
    manifest: SequenceManifest,
    rows: list[FrameRecord],
) -> dict[str, object]:
    measured = [row for row in rows if row.qa_status == "measured"]
    return {
        "sequence_id": manifest.sequence_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frames_processed": len(measured),
        "frames_skipped": sum(1 for row in rows if row.qa_status == "skipped"),
        "frames_with_fallback": sum(1 for row in rows if row.qa_used_lama_fallback),
        "median_mask_ratios": {
            "sky": _median([row.qa_sky_ratio for row in measured if row.qa_sky_ratio is not None]),
            "dynamic": _median([row.qa_dynamic_ratio for row in measured if row.qa_dynamic_ratio is not None]),
            "roof": _median([row.qa_roof_ratio for row in measured if row.qa_roof_ratio is not None]),
            "inpaint": _median([row.qa_inpaint_ratio for row in measured if row.qa_inpaint_ratio is not None]),
        },
        "percentile_seam_errors": {
            "rgb": _percentiles([row.qa_seam_delta_rgb for row in measured if row.qa_seam_delta_rgb is not None]),
            "alpha": _percentiles([row.qa_seam_delta_alpha for row in measured if row.qa_seam_delta_alpha is not None]),
        },
        "failure_counts_by_stage": _count_stage_failures(rows),
    }


def _write_frame_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FRAME_METRIC_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_sequence_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_sequence_qa(sequence_manifest: SequenceManifest, config: IndexConfig) -> SequenceManifest:
    if sequence_manifest.status != "ready":
        return sequence_manifest

    sampled_frame_indices = _sample_frame_indices(sequence_manifest.rows, config.qa_sample_count)
    updated_rows: list[FrameRecord] = []
    metric_rows: list[dict[str, object]] = []
    for frame_index, frame in enumerate(sequence_manifest.rows):
        updated_frame, metric_row = _measure_frame(frame_index, frame, sampled_frame_indices, config)
        updated_rows.append(updated_frame)
        metric_rows.append(metric_row)

    failed_count = sum(1 for row in updated_rows if row.qa_status == "failed")
    status = "ready" if failed_count == 0 else "failed_qa"
    failure_reason = None if failed_count == 0 else f"{failed_count} frame(s) failed QA measurement"
    qa_manifest = SequenceManifest(
        sequence_id=sequence_manifest.sequence_id,
        sequence_dir=sequence_manifest.sequence_dir,
        csv_path=sequence_manifest.csv_path,
        output_dir=sequence_manifest.output_dir,
        status=status,
        failure_reason=failure_reason,
        total_csv_rows=sequence_manifest.total_csv_rows,
        valid_frames=sequence_manifest.valid_frames,
        skipped_frames=sequence_manifest.skipped_frames,
        rows=updated_rows,
    )

    if not config.dry_run:
        output_dirs = sequence_output_dirs(config.output_root, sequence_manifest.sequence_id)
        _write_frame_metrics(output_dirs["qa_metrics"] / "frame_metrics.csv", metric_rows)
        _write_sequence_metrics(
            output_dirs["qa_metrics"] / "sequence_metrics.json",
            _sequence_metrics(qa_manifest, updated_rows),
        )
        write_sequence_manifest(qa_manifest)

    return qa_manifest


def run_qa(
    config: IndexConfig,
    sequence_ids: Optional[Iterable[str]] = None,
) -> list[SequenceManifest]:
    manifests = run_indexing(config, sequence_ids=sequence_ids)
    return [run_sequence_qa(manifest, config) for manifest in manifests]
