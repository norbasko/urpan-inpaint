from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.cubemap import FACE_ORDER, cubemap_face_mask_to_erp
from urpan_inpaint.discovery import run_indexing, write_sequence_manifest
from urpan_inpaint.models import FrameRecord, SequenceManifest
from urpan_inpaint.projection import load_or_create_cubemap_face_rgbs
from urpan_inpaint.refinement import DYNAMIC_PROMPT_CLASSES, sam2_output_dir_for_frame
from urpan_inpaint.semantic import semantic_output_dir_for_frame


PARSER_DYNAMIC_CLASSES = {
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
    "train",
    "trailer",
    "caravan",
}


@dataclass(frozen=True)
class FrameFusionInputs:
    frame: FrameRecord
    dyn_coarse: np.ndarray
    dyn_ovd: np.ndarray
    roof_refined: np.ndarray
    sky_refined: np.ndarray
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameFusionMasks:
    dynamic: np.ndarray
    roof: np.ndarray
    sky: np.ndarray
    inpaint: np.ndarray


def _normalize_class_text(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())


def _empty_erp_mask(frame: FrameRecord) -> np.ndarray:
    if frame.erp_width is None or frame.erp_height is None:
        raise RuntimeError("Cannot fuse masks without ERP dimensions")
    return np.zeros((frame.erp_height, frame.erp_width), dtype=np.uint8)


def _load_mask_png(path: Path, shape: tuple[int, int]) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    mask = np.asarray(Image.open(path).convert("L"))
    if mask.shape != shape:
        return None
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _binary_dilate(mask: np.ndarray, radius: int, wrap_x: bool = False) -> np.ndarray:
    result = mask > 0
    for _ in range(max(0, int(radius))):
        padded = np.pad(result, ((1, 1), (0, 0)), mode="edge")
        up = padded[:-2, :]
        down = padded[2:, :]
        left = np.roll(result, 1, axis=1) if wrap_x else np.pad(result[:, :-1], ((0, 0), (1, 0)), mode="constant")
        right = np.roll(result, -1, axis=1) if wrap_x else np.pad(result[:, 1:], ((0, 0), (0, 1)), mode="constant")
        result = result | up | down | left | right
    return result


def _binary_erode(mask: np.ndarray, radius: int, wrap_x: bool = False) -> np.ndarray:
    result = mask > 0
    for _ in range(max(0, int(radius))):
        padded = np.pad(result, ((1, 1), (0, 0)), mode="edge")
        up = padded[:-2, :]
        down = padded[2:, :]
        left = np.roll(result, 1, axis=1) if wrap_x else np.pad(result[:, :-1], ((0, 0), (1, 0)), mode="constant")
        right = np.roll(result, -1, axis=1) if wrap_x else np.pad(result[:, 1:], ((0, 0), (0, 1)), mode="constant")
        result = result & up & down & left & right
    return result


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0
    try:
        from scipy import ndimage

        return np.where(ndimage.binary_fill_holes(binary), 255, 0).astype(np.uint8)
    except Exception:
        pass

    height, width = binary.shape
    exterior = np.zeros_like(binary)
    stack: list[tuple[int, int]] = []
    border = np.zeros_like(binary)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    ys, xs = np.where(border & ~binary)
    stack.extend(zip(ys.tolist(), xs.tolist()))

    while stack:
        y, x = stack.pop()
        if exterior[y, x] or binary[y, x]:
            continue
        exterior[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < height:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < width:
            stack.append((y, x + 1))

    filled = binary | (~binary & ~exterior)
    return np.where(filled, 255, 0).astype(np.uint8)


def _suppress_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    binary = mask > 0
    min_area = max(0, int(min_area_px))
    if min_area <= 1 or int(binary.sum()) == 0:
        return np.where(binary, 255, 0).astype(np.uint8)

    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=4)
        keep_ids = np.where(stats[:, cv2.CC_STAT_AREA] >= min_area)[0]
        keep_ids = keep_ids[keep_ids != 0]
        return np.where(np.isin(labels, keep_ids), 255, 0).astype(np.uint8)
    except Exception:
        pass

    try:
        from scipy import ndimage

        labels, count = ndimage.label(binary)
        areas = np.bincount(labels.ravel())
        keep_ids = np.where(areas >= min_area)[0]
        keep_ids = keep_ids[keep_ids != 0]
        return np.where(np.isin(labels, keep_ids), 255, 0).astype(np.uint8)
    except Exception:
        pass

    height, width = binary.shape
    visited = np.zeros_like(binary)
    result = np.zeros_like(binary)
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not binary[y, x]:
                continue
            stack = [(y, x)]
            component: list[tuple[int, int]] = []
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if visited[ny, nx] or not binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            if len(component) >= min_area:
                ys, xs = zip(*component)
                result[np.asarray(ys), np.asarray(xs)] = True
    return np.where(result, 255, 0).astype(np.uint8)


def _smooth_seam(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask > 0
    for _ in range(max(0, int(iterations))):
        vertical = np.pad(result, ((1, 1), (0, 0)), mode="edge")
        rows = [vertical[:-2, :], result, vertical[2:, :]]
        neighbor_count = np.zeros(result.shape, dtype=np.uint8)
        for row in rows:
            neighbor_count += np.roll(row, 1, axis=1).astype(np.uint8)
            neighbor_count += row.astype(np.uint8)
            neighbor_count += np.roll(row, -1, axis=1).astype(np.uint8)
        result = result & (neighbor_count >= 2)
    return np.where(result, 255, 0).astype(np.uint8)


def _morph_dynamic_or_roof(
    mask: np.ndarray,
    min_area_px: int,
    dilate_px: int,
    erode_after_dilate_px: int,
) -> np.ndarray:
    result = _fill_holes(mask)
    result = _suppress_small_components(result, min_area_px)
    if dilate_px > 0:
        result = np.where(_binary_dilate(result, dilate_px), 255, 0).astype(np.uint8)
    if erode_after_dilate_px > 0:
        result = np.where(_binary_erode(result, erode_after_dilate_px), 255, 0).astype(np.uint8)
    return result


def _temporal_consistency(masks: list[np.ndarray]) -> list[np.ndarray]:
    consistent: list[np.ndarray] = []
    for index, current in enumerate(masks):
        result = current > 0
        prev_mask = masks[index - 1] > 0 if index > 0 and masks[index - 1].shape == current.shape else None
        next_mask = masks[index + 1] > 0 if index + 1 < len(masks) and masks[index + 1].shape == current.shape else None
        if prev_mask is not None and next_mask is not None:
            result = result | (prev_mask & next_mask)
        consistent.append(np.where(result, 255, 0).astype(np.uint8))
    return consistent


def _project_face_mask(frame: FrameRecord, face_name: str, mask: np.ndarray) -> np.ndarray:
    if frame.erp_width is None or frame.erp_height is None:
        raise RuntimeError("Cannot reproject masks without ERP dimensions")
    if frame.cubemap_face_size is None or frame.cubemap_overlap_px is None:
        raise RuntimeError("Cannot reproject masks without cubemap geometry")
    return cubemap_face_mask_to_erp(
        mask,
        face_name=face_name,
        erp_width=frame.erp_width,
        erp_height=frame.erp_height,
        face_size=frame.cubemap_face_size,
        overlap_px=frame.cubemap_overlap_px,
    )


def _load_semantic_dynamic_erp(frame: FrameRecord) -> tuple[np.ndarray, int]:
    result = _empty_erp_mask(frame)
    face_count = 0
    for face_name in FACE_ORDER:
        face_path = semantic_output_dir_for_frame(frame) / f"{face_name}.npz"
        if not face_path.is_file():
            continue
        with np.load(face_path) as payload:
            if "target_masks" not in payload or "target_class_names" not in payload:
                continue
            masks = payload["target_masks"]
            names = [str(item) for item in payload["target_class_names"]]
        face_union: Optional[np.ndarray] = None
        for index, name in enumerate(names):
            if _normalize_class_text(name) not in PARSER_DYNAMIC_CLASSES or index >= len(masks):
                continue
            mask = np.where(masks[index] > 0, 255, 0).astype(np.uint8)
            face_union = mask if face_union is None else np.maximum(face_union, mask).astype(np.uint8)
        if face_union is None:
            continue
        result = np.maximum(result, _project_face_mask(frame, face_name, face_union))
        face_count += 1
    return result, face_count


def _load_sam2_grounding_dynamic_erp(frame: FrameRecord) -> tuple[np.ndarray, int]:
    result = _empty_erp_mask(frame)
    mask_count = 0
    output_dir = sam2_output_dir_for_frame(frame)
    dynamic_classes = {_normalize_class_text(item) for item in DYNAMIC_PROMPT_CLASSES}
    for face_name in FACE_ORDER:
        face_path = output_dir / f"{face_name}.npz"
        if not face_path.is_file():
            continue
        with np.load(face_path) as payload:
            if "masks" not in payload or "class_text" not in payload or "source" not in payload:
                continue
            masks = payload["masks"]
            class_texts = [str(item) for item in payload["class_text"]]
            sources = [str(item) for item in payload["source"]]
        face_union: Optional[np.ndarray] = None
        for index, class_text in enumerate(class_texts):
            source = sources[index] if index < len(sources) else ""
            class_name = _normalize_class_text(class_text)
            if class_name not in dynamic_classes:
                continue
            if source not in {"grounding_box", "temporal_prior"}:
                continue
            if index >= len(masks):
                continue
            mask = np.where(masks[index] > 0, 255, 0).astype(np.uint8)
            face_union = mask if face_union is None else np.maximum(face_union, mask).astype(np.uint8)
            mask_count += 1
        if face_union is None:
            continue
        result = np.maximum(result, _project_face_mask(frame, face_name, face_union))
    return result, mask_count


def _load_frame_fusion_inputs(frame: FrameRecord, config: IndexConfig) -> FrameFusionInputs:
    projected_frame, _, _ = load_or_create_cubemap_face_rgbs(frame, config)
    shape = (projected_frame.erp_height, projected_frame.erp_width)
    if shape[0] is None or shape[1] is None:
        raise RuntimeError("Cannot fuse masks without ERP geometry")

    warnings: list[str] = []
    dyn_coarse, coarse_faces = _load_semantic_dynamic_erp(projected_frame)
    dyn_ovd, ovd_masks = _load_sam2_grounding_dynamic_erp(projected_frame)
    if coarse_faces == 0:
        warnings.append("missing semantic dynamic masks")
    if ovd_masks == 0:
        warnings.append("missing Grounding DINO SAM 2 dynamic masks")

    erp_shape = (int(shape[0]), int(shape[1]))
    roof_refined = _load_mask_png(projected_frame.roof_mask_path, erp_shape)
    if roof_refined is None:
        roof_refined = _empty_erp_mask(projected_frame)
        warnings.append("missing roof_refined mask")

    sky_refined = _load_mask_png(projected_frame.sky_mask_path, erp_shape)
    if sky_refined is None:
        sky_refined = _empty_erp_mask(projected_frame)
        warnings.append("missing sky_refined mask")

    return FrameFusionInputs(
        frame=projected_frame,
        dyn_coarse=dyn_coarse,
        dyn_ovd=dyn_ovd,
        roof_refined=roof_refined,
        sky_refined=sky_refined,
        warnings=tuple(warnings),
    )


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8)).save(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_union_debug(path: Path, masks: FrameFusionMasks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    debug = np.zeros((*masks.dynamic.shape, 3), dtype=np.uint8)
    debug[..., 0] = np.where(masks.dynamic > 0, 255, 0).astype(np.uint8)
    debug[..., 1] = np.where(masks.roof > 0, 255, 0).astype(np.uint8)
    debug[..., 2] = np.where(masks.sky > 0, 255, 0).astype(np.uint8)
    Image.fromarray(debug, mode="RGB").save(path)


def _write_fused_frame_masks(inputs: FrameFusionInputs, masks: FrameFusionMasks, config: IndexConfig) -> FrameRecord:
    frame = inputs.frame
    if not config.dry_run:
        _write_mask(frame.dynamic_mask_path, masks.dynamic)
        _write_mask(frame.roof_mask_path, masks.roof)
        _write_mask(frame.sky_mask_path, masks.sky)
        _write_mask(frame.inpaint_mask_path, masks.inpaint)
        _write_union_debug(frame.union_debug_path, masks)
        base_payload = {
            "frame_name": frame.frame_name,
            "warnings": list(inputs.warnings),
            "morphology": {
                "order": ["hole_filling", "small_component_suppression", "edge_dilation", "optional_erosion"],
                "dyn_min_component_area_px": config.dyn_min_component_area_px,
                "roof_min_component_area_px": config.roof_min_component_area_px,
                "dyn_dilate_px": config.dyn_dilate_px,
                "roof_dilate_px": config.roof_dilate_px,
                "dyn_erode_after_dilate_px": config.dyn_erode_after_dilate_px,
                "roof_erode_after_dilate_px": config.roof_erode_after_dilate_px,
            },
        }
        _write_json(
            frame.dynamic_mask_path.with_suffix(".json"),
            {
                **base_payload,
                "source": "temporal_consistency(dilate(DYN_COARSE OR DYN_OVD))",
                "area_px": int((masks.dynamic > 0).sum()),
            },
        )
        _write_json(
            frame.roof_mask_path.with_suffix(".json"),
            {
                **base_payload,
                "source": "temporal_consistency(roof_refined)",
                "area_px": int((masks.roof > 0).sum()),
            },
        )
        _write_json(
            frame.sky_mask_path.with_suffix(".json"),
            {
                **base_payload,
                "source": "seam_smooth(sky_refined)",
                "area_px": int((masks.sky > 0).sum()),
            },
        )
        _write_json(
            frame.inpaint_mask_path.with_suffix(".json"),
            {
                **base_payload,
                "source": "(DYN OR ROOF) AND NOT SKY",
                "area_px": int((masks.inpaint > 0).sum()),
            },
        )

    dynamic_area = int((masks.dynamic > 0).sum())
    roof_area = int((masks.roof > 0).sum())
    sky_area = int((masks.sky > 0).sum())
    inpaint_area = int((masks.inpaint > 0).sum())
    warning_text = "; ".join(inputs.warnings)
    return replace(
        frame,
        dynamic_mask_status="fused",
        dynamic_mask_source="parser_dynamic_or_grounding_sam2",
        dynamic_mask_area_px=dynamic_area,
        dynamic_mask_error=warning_text,
        roof_mask_status="fused",
        roof_mask_source="temporal_consistency(roof_refined)",
        roof_mask_area_px=roof_area,
        roof_mask_error=warning_text if "missing roof_refined mask" in inputs.warnings else "",
        sky_mask_status="fused",
        sky_mask_source="seam_smooth(sky_refined)",
        sky_mask_area_px=sky_area,
        sky_mask_error=warning_text if "missing sky_refined mask" in inputs.warnings else "",
        inpaint_mask_status="fused",
        inpaint_mask_area_px=inpaint_area,
        inpaint_mask_error="",
        mask_fusion_status="fused",
        mask_fusion_error=warning_text,
    )


def fuse_sequence_masks(sequence_manifest: SequenceManifest, config: IndexConfig) -> SequenceManifest:
    if sequence_manifest.status != "ready":
        return sequence_manifest

    inputs_by_frame: list[Optional[FrameFusionInputs]] = []
    failed_rows: dict[int, FrameRecord] = {}
    for frame_index, frame in enumerate(sequence_manifest.rows):
        try:
            inputs_by_frame.append(_load_frame_fusion_inputs(frame, config))
        except Exception as exc:
            failed_rows[frame_index] = replace(
                frame,
                mask_fusion_status="failed",
                mask_fusion_error=str(exc),
                dynamic_mask_status="failed",
                dynamic_mask_error=str(exc),
                inpaint_mask_status="failed",
                inpaint_mask_error=str(exc),
            )
            inputs_by_frame.append(None)

    dyn_base = [
        _empty_erp_mask(inputs.frame) if inputs is not None else np.zeros((0, 0), dtype=np.uint8)
        for inputs in inputs_by_frame
    ]
    roof_base = [
        _empty_erp_mask(inputs.frame) if inputs is not None else np.zeros((0, 0), dtype=np.uint8)
        for inputs in inputs_by_frame
    ]
    sky_masks = [
        _empty_erp_mask(inputs.frame) if inputs is not None else np.zeros((0, 0), dtype=np.uint8)
        for inputs in inputs_by_frame
    ]

    for index, inputs in enumerate(inputs_by_frame):
        if inputs is None:
            continue
        dyn_union = np.maximum(inputs.dyn_coarse, inputs.dyn_ovd)
        dyn_base[index] = _morph_dynamic_or_roof(
            dyn_union,
            min_area_px=config.dyn_min_component_area_px,
            dilate_px=config.dyn_dilate_px,
            erode_after_dilate_px=config.dyn_erode_after_dilate_px,
        )
        roof_base[index] = _morph_dynamic_or_roof(
            inputs.roof_refined,
            min_area_px=config.roof_min_component_area_px,
            dilate_px=config.roof_dilate_px,
            erode_after_dilate_px=config.roof_erode_after_dilate_px,
        )
        sky_masks[index] = _smooth_seam(inputs.sky_refined, iterations=config.sky_mask_erp_smoothing_iterations)

    dyn_temporal = _temporal_consistency(dyn_base)
    roof_temporal = _temporal_consistency(roof_base)

    fused_rows: list[FrameRecord] = []
    for frame_index, inputs in enumerate(inputs_by_frame):
        if inputs is None:
            fused_rows.append(failed_rows[frame_index])
            continue
        dynamic = dyn_temporal[frame_index]
        roof = roof_temporal[frame_index]
        sky = sky_masks[frame_index]
        inpaint = np.where(((dynamic > 0) | (roof > 0)) & ~(sky > 0), 255, 0).astype(np.uint8)
        masks = FrameFusionMasks(dynamic=dynamic, roof=roof, sky=sky, inpaint=inpaint)
        fused_rows.append(_write_fused_frame_masks(inputs, masks, config))

    fusion_failures = [row for row in fused_rows if row.mask_fusion_status == "failed"]
    status = sequence_manifest.status
    failure_reason = sequence_manifest.failure_reason
    if fusion_failures:
        status = "failed_mask_fusion"
        failure_reason = f"{len(fusion_failures)} frame(s) failed mask fusion"

    fused_manifest = SequenceManifest(
        sequence_id=sequence_manifest.sequence_id,
        sequence_dir=sequence_manifest.sequence_dir,
        csv_path=sequence_manifest.csv_path,
        output_dir=sequence_manifest.output_dir,
        status=status,
        failure_reason=failure_reason,
        total_csv_rows=sequence_manifest.total_csv_rows,
        valid_frames=sequence_manifest.valid_frames,
        skipped_frames=sequence_manifest.skipped_frames,
        rows=fused_rows,
    )

    if not config.dry_run:
        write_sequence_manifest(fused_manifest)

    return fused_manifest


def run_mask_fusion(
    config: IndexConfig,
    sequence_ids: Optional[Iterable[str]] = None,
) -> list[SequenceManifest]:
    indexed_manifests = run_indexing(config, sequence_ids=sequence_ids)
    return [fuse_sequence_masks(manifest, config) for manifest in indexed_manifests]
