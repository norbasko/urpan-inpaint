from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import numpy as np
from PIL import Image

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.cubemap import FACE_ORDER, CubemapProjection
from urpan_inpaint.discovery import run_indexing, write_sequence_manifest
from urpan_inpaint.erp import load_erp_rgb
from urpan_inpaint.models import FrameRecord, SequenceManifest
from urpan_inpaint.projection import load_or_create_cubemap_projection
from urpan_inpaint.windowing import (
    InpaintWindow,
    build_inpaint_windows,
    build_sequence_inpaint_windows,
    reconcile_window_predictions,
)


class ProPainterRuntimeError(RuntimeError):
    """Raised when ProPainter inference cannot be executed in the selected environment."""


class ProPainterOutOfMemoryError(ProPainterRuntimeError):
    """Raised after ProPainter exhausts all chunk-size retry levels due to OOM."""


class LaMaRuntimeError(RuntimeError):
    """Raised when LaMa fallback inference cannot be executed in the selected environment."""


class ProPainterBackend(Protocol):
    model_id: str

    def inpaint_clip(
        self,
        face_name: str,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        config: IndexConfig,
    ) -> list[np.ndarray]:
        """Return inpainted RGB frames for one temporally ordered face clip."""


class LaMaBackend(Protocol):
    model_id: str

    def inpaint_image(
        self,
        face_name: str,
        image: np.ndarray,
        mask: np.ndarray,
        config: IndexConfig,
    ) -> np.ndarray:
        """Return one LaMa-inpainted RGB face image."""


@dataclass(frozen=True)
class FaceWindowChunk:
    window_index: int
    chunk_index: int
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class SequenceInpaintArtifacts:
    output_dir: Path
    windows: list[InpaintWindow]
    chunks: list[FaceWindowChunk]


@dataclass(frozen=True)
class FrameErpMasks:
    dynamic: np.ndarray
    roof: np.ndarray
    sky: np.ndarray
    inpaint: np.ndarray


class ExternalProPainterBackend:
    def __init__(self, command_template: str, model_id: str = "ProPainter") -> None:
        if not command_template:
            raise ProPainterRuntimeError(
                "ProPainter inference requires --propainter-command or an injected ProPainterBackend. "
                "The command template may use {frames_dir}, {masks_dir}, {output_dir}, {face_name}, and {device}."
            )
        self.command_template = command_template
        self.model_id = model_id

    @classmethod
    def from_config(cls, config: IndexConfig) -> "ExternalProPainterBackend":
        return cls(command_template=config.propainter_command, model_id=config.propainter_model_id)

    def inpaint_clip(
        self,
        face_name: str,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        config: IndexConfig,
    ) -> list[np.ndarray]:
        if len(frames) != len(masks):
            raise ProPainterRuntimeError("Frame/mask count mismatch for ProPainter clip")
        if not frames:
            return []

        with tempfile.TemporaryDirectory(prefix="urpan-propainter-") as temp_name:
            temp_dir = Path(temp_name)
            frames_dir = temp_dir / "frames"
            masks_dir = temp_dir / "masks"
            output_dir = temp_dir / "output"
            frames_dir.mkdir()
            masks_dir.mkdir()
            output_dir.mkdir()
            for index, (frame, mask) in enumerate(zip(frames, masks)):
                Image.fromarray(frame.astype(np.uint8), mode="RGB").save(frames_dir / f"{index:06d}.png")
                Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8)).save(masks_dir / f"{index:06d}.png")

            command = self.command_template.format(
                frames_dir=str(frames_dir),
                masks_dir=str(masks_dir),
                output_dir=str(output_dir),
                face_name=face_name,
                device=config.propainter_device,
            )
            completed = subprocess.run(
                shlex.split(command),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise ProPainterRuntimeError(
                    f"ProPainter command failed with exit code {completed.returncode}: {completed.stderr.strip()}"
                )

            outputs: list[np.ndarray] = []
            for index in range(len(frames)):
                output_path = output_dir / f"{index:06d}.png"
                if not output_path.is_file():
                    raise ProPainterRuntimeError(f"ProPainter command did not write {output_path.name}")
                output = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.uint8)
                if output.shape != frames[index].shape:
                    raise ProPainterRuntimeError(
                        f"ProPainter output shape {output.shape} does not match input shape {frames[index].shape}"
                    )
                outputs.append(output)
            return outputs


class PassthroughProPainterBackend:
    model_id = "ProPainter/dry-run-passthrough"

    def inpaint_clip(
        self,
        face_name: str,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        config: IndexConfig,
    ) -> list[np.ndarray]:
        return [frame.copy() for frame in frames]


class ExternalLaMaBackend:
    def __init__(self, command_template: str, model_id: str = "LaMa") -> None:
        if not command_template:
            raise LaMaRuntimeError(
                "LaMa fallback requires --lama-command or an injected LaMaBackend. "
                "The command template may use {image_path}, {mask_path}, {output_path}, "
                "{output_dir}, {face_name}, and {device}."
            )
        self.command_template = command_template
        self.model_id = model_id

    @classmethod
    def from_config(cls, config: IndexConfig) -> "ExternalLaMaBackend":
        return cls(command_template=config.lama_command, model_id=config.lama_model_id)

    def inpaint_image(
        self,
        face_name: str,
        image: np.ndarray,
        mask: np.ndarray,
        config: IndexConfig,
    ) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="urpan-lama-") as temp_name:
            temp_dir = Path(temp_name)
            image_path = temp_dir / "image.png"
            mask_path = temp_dir / "mask.png"
            output_dir = temp_dir / "output"
            output_path = output_dir / "output.png"
            output_dir.mkdir()
            Image.fromarray(image.astype(np.uint8), mode="RGB").save(image_path)
            Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8)).save(mask_path)

            command = self.command_template.format(
                image_path=str(image_path),
                mask_path=str(mask_path),
                output_path=str(output_path),
                output_dir=str(output_dir),
                face_name=face_name,
                device=config.lama_device,
            )
            completed = subprocess.run(
                shlex.split(command),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise LaMaRuntimeError(
                    f"LaMa command failed with exit code {completed.returncode}: {completed.stderr.strip()}"
                )
            if not output_path.is_file():
                raise LaMaRuntimeError(f"LaMa command did not write {output_path}")
            output = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.uint8)
            if output.shape != image.shape:
                raise LaMaRuntimeError(f"LaMa output shape {output.shape} does not match input shape {image.shape}")
            return output


class PassthroughLaMaBackend:
    model_id = "LaMa/dry-run-passthrough"

    def inpaint_image(
        self,
        face_name: str,
        image: np.ndarray,
        mask: np.ndarray,
        config: IndexConfig,
    ) -> np.ndarray:
        return image.copy()


def propainter_output_dir_for_frame(frame: FrameRecord) -> Path:
    return frame.cubemap_cache_dir / "propainter"


def sequence_propainter_output_dir(manifest: SequenceManifest) -> Path:
    return manifest.output_dir / "propainter"


def lama_output_dir_for_frame(frame: FrameRecord) -> Path:
    return frame.cubemap_cache_dir / "lama_fallback"


def sequence_lama_output_dir(manifest: SequenceManifest) -> Path:
    return manifest.output_dir / "lama_fallback"


def _load_required_erp_mask(path: Path, expected_shape: tuple[int, int], label: str) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"Missing final ERP {label} mask: {path}")
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if mask.shape != expected_shape:
        raise RuntimeError(f"{label} mask shape {mask.shape} does not match ERP shape {expected_shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _load_final_erp_masks(frame: FrameRecord) -> FrameErpMasks:
    if frame.erp_width is None or frame.erp_height is None:
        raise RuntimeError("Cannot load final ERP masks without ERP geometry")
    expected_shape = (frame.erp_height, frame.erp_width)
    return FrameErpMasks(
        dynamic=_load_required_erp_mask(frame.dynamic_mask_path, expected_shape, "DYN"),
        roof=_load_required_erp_mask(frame.roof_mask_path, expected_shape, "ROOF"),
        sky=_load_required_erp_mask(frame.sky_mask_path, expected_shape, "SKY"),
        inpaint=_load_required_erp_mask(frame.inpaint_mask_path, expected_shape, "INPAINT"),
    )


def _sample_erp_mask_to_face(mask: np.ndarray, erp_x: np.ndarray, erp_y: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    xs = np.mod(np.rint(erp_x).astype(np.int32), width)
    ys = np.clip(np.rint(erp_y).astype(np.int32), 0, height - 1)
    return np.where(mask[ys, xs] > 0, 255, 0).astype(np.uint8)


def _compose_masked_prediction(original: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if prediction.shape != original.shape:
        raise RuntimeError(
            f"ProPainter output shape {prediction.shape} does not match input shape {original.shape}"
        )
    binary = (mask > 0)[..., None]
    return np.where(binary, prediction, original).astype(np.uint8)


def _alpha_from_sky_mask(sky_mask: np.ndarray) -> np.ndarray:
    return np.where(sky_mask == 255, 0, 255).astype(np.uint8)


def _chunk_frame_indices(frame_indices: tuple[int, ...], chunk_size: int) -> list[tuple[int, ...]]:
    size = max(1, int(chunk_size))
    return [frame_indices[start : start + size] for start in range(0, len(frame_indices), size)]


def _face_feather_weights(shape: tuple[int, int], overlap_px: int, feather_px: int) -> np.ndarray:
    height, width = shape
    if feather_px <= 0:
        return np.ones((height, width), dtype=np.float32)
    yy, xx = np.ogrid[:height, :width]
    horizontal = np.minimum(xx + 1, width - xx)
    vertical = np.minimum(yy + 1, height - yy)
    distance = np.minimum(horizontal, vertical).astype(np.float32)
    effective = max(1, min(int(feather_px), max(1, int(overlap_px))))
    return np.clip(distance / float(effective), 0.0, 1.0).astype(np.float32)


def _reproject_inpainted_faces_to_erp(
    projection: CubemapProjection,
    face_rgbs: dict[str, np.ndarray],
    original_erp_rgb: np.ndarray,
    config: IndexConfig,
) -> np.ndarray:
    accum = np.zeros((*original_erp_rgb.shape[:2], 3), dtype=np.float32)
    weights = np.zeros(original_erp_rgb.shape[:2], dtype=np.float32)
    erp_height, erp_width = original_erp_rgb.shape[:2]
    feather_px = config.propainter_face_feather_px

    for face_name in FACE_ORDER:
        face = projection.faces[face_name]
        rgb = face_rgbs[face_name]
        weight = _face_feather_weights(rgb.shape[:2], projection.overlap_px, feather_px)
        xs = np.mod(np.rint(face.erp_x).astype(np.int32), erp_width)
        ys = np.clip(np.rint(face.erp_y).astype(np.int32), 0, erp_height - 1)
        flat_y = ys.ravel()
        flat_x = xs.ravel()
        flat_weight = weight.ravel()
        for channel in range(3):
            np.add.at(
                accum[..., channel],
                (flat_y, flat_x),
                rgb[..., channel].ravel().astype(np.float32) * flat_weight,
            )
        np.add.at(weights, (flat_y, flat_x), flat_weight)

    result = original_erp_rgb.astype(np.float32)
    valid = weights > 1e-6
    result[valid] = accum[valid] / weights[valid, None]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _write_face_artifacts(
    output_dir: Path,
    face_name: str,
    frame: FrameRecord,
    rgb: np.ndarray,
    mask: np.ndarray,
    overlap_px: int,
    config: IndexConfig,
) -> None:
    if config.dry_run:
        return
    face_dir = output_dir / face_name
    face_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(face_dir / f"{frame.frame_stem}.with_overlap.png")
    crop = (
        rgb
        if overlap_px <= 0
        else rgb[overlap_px : rgb.shape[0] - overlap_px, overlap_px : rgb.shape[1] - overlap_px]
    )
    Image.fromarray(crop.astype(np.uint8), mode="RGB").save(face_dir / f"{frame.frame_stem}.png")
    Image.fromarray(mask.astype(np.uint8)).save(face_dir / f"{frame.frame_stem}.mask.png")


def _write_frame_outputs(frame: FrameRecord, rgb: np.ndarray, sky_mask: np.ndarray, config: IndexConfig) -> None:
    if config.dry_run:
        return
    if frame.erp_width is None or frame.erp_height is None:
        raise RuntimeError("Cannot write final products without ERP geometry")
    expected_shape = (frame.erp_height, frame.erp_width)
    if rgb.shape != (*expected_shape, 3):
        raise RuntimeError(f"Final RGB shape {rgb.shape} does not match ERP shape {expected_shape}")
    if sky_mask.shape != expected_shape:
        raise RuntimeError(f"Final SKY mask shape {sky_mask.shape} does not match ERP shape {expected_shape}")
    if frame.rgb_output_path.suffix.lower() != ".png" or frame.rgba_output_path.suffix.lower() != ".png":
        raise RuntimeError("Final RGB/RGBA products must be PNG files")

    frame.rgb_output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.rgba_output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(frame.rgb_output_path)
    alpha = _alpha_from_sky_mask(sky_mask)
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(frame.rgba_output_path)


def _write_sequence_metadata(
    output_dir: Path,
    artifacts: SequenceInpaintArtifacts,
    backend: ProPainterBackend,
    config: IndexConfig,
) -> None:
    if config.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": getattr(backend, "model_id", config.propainter_model_id),
        "face_order": list(FACE_ORDER),
        "window_size": config.inpaint_window_size,
        "window_stride": config.inpaint_window_stride,
        "chunk_size": config.propainter_chunk_size,
        "face_feather_px": config.propainter_face_feather_px,
        "windows": [window.to_dict() for window in artifacts.windows],
        "chunks": [
            {
                "window_index": chunk.window_index,
                "chunk_index": chunk.chunk_index,
                "frame_indices": list(chunk.frame_indices),
            }
            for chunk in artifacts.chunks
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_lama_sequence_metadata(
    output_dir: Path,
    fallback_reason: str,
    backend: LaMaBackend,
    config: IndexConfig,
    windows: list[InpaintWindow],
    processed_frame_indices: list[int],
) -> None:
    if config.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": getattr(backend, "model_id", config.lama_model_id),
        "fallback_reason": fallback_reason,
        "face_order": list(FACE_ORDER),
        "window_size": config.inpaint_window_size,
        "window_stride": config.inpaint_window_stride,
        "propainter_min_window_frames": config.propainter_min_window_frames,
        "single_frame_min_mask_area_px": config.single_frame_min_mask_area_px,
        "face_feather_px": config.propainter_face_feather_px,
        "windows": [window.to_dict() for window in windows],
        "processed_frame_indices": processed_frame_indices,
    }
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "oom" in text or "cuda memory" in text


def _propainter_retry_chunk_sizes(initial_chunk_size: int) -> list[int]:
    size = max(1, int(initial_chunk_size))
    sizes: list[int] = []
    while size >= 1:
        if size not in sizes:
            sizes.append(size)
        if size == 1:
            break
        size = max(1, size // 2)
    return sizes


def _clone_face_rgb_mapping(
    face_rgbs_by_frame: dict[int, dict[str, np.ndarray]],
) -> dict[int, dict[str, np.ndarray]]:
    return {frame_index: dict(face_rgbs) for frame_index, face_rgbs in face_rgbs_by_frame.items()}


def _select_single_frame_fallback_reason(
    config: IndexConfig,
    windows: list[InpaintWindow],
    erp_masks_by_frame: dict[int, FrameErpMasks],
) -> str:
    if config.force_single_frame_fallback:
        return "force_single_frame_fallback"

    max_window_size = max((window.size for window in windows), default=0)
    min_window_size = max(1, int(config.propainter_min_window_frames))
    if max_window_size < min_window_size:
        return f"window_length_below_minimum_viable_size(max={max_window_size}, min={min_window_size})"

    mask_threshold = max(0, int(config.single_frame_min_mask_area_px))
    if mask_threshold > 0 and erp_masks_by_frame:
        max_mask_area = max(int((masks.inpaint > 0).sum()) for masks in erp_masks_by_frame.values())
        if max_mask_area < mask_threshold:
            return f"mask_area_too_small(max={max_mask_area}, threshold={mask_threshold})"

    return ""


def _process_face_stream(
    face_name: str,
    windows: list[InpaintWindow],
    face_rgbs_by_frame: dict[int, dict[str, np.ndarray]],
    face_masks_by_frame: dict[int, dict[str, np.ndarray]],
    backend: ProPainterBackend,
    config: IndexConfig,
) -> tuple[dict[int, np.ndarray], list[FaceWindowChunk]]:
    predictions_by_window: dict[int, dict[int, np.ndarray]] = {}
    chunks: list[FaceWindowChunk] = []
    for window in windows:
        window_predictions: dict[int, np.ndarray] = {}
        chunk_ranges = _chunk_frame_indices(window.frame_indices, config.propainter_chunk_size)
        for chunk_index, frame_indices in enumerate(chunk_ranges):
            frames = [face_rgbs_by_frame[frame_index][face_name] for frame_index in frame_indices]
            masks = [face_masks_by_frame[frame_index][face_name] for frame_index in frame_indices]
            predicted = backend.inpaint_clip(face_name, frames, masks, config)
            if len(predicted) != len(frames):
                raise RuntimeError(
                    f"ProPainter returned {len(predicted)} frame(s) for {len(frames)} input frame(s)"
                )
            for frame_index, original, mask, prediction in zip(frame_indices, frames, masks, predicted):
                window_predictions[frame_index] = _compose_masked_prediction(original, prediction, mask)
            chunks.append(
                FaceWindowChunk(
                    window_index=window.window_index,
                    chunk_index=chunk_index,
                    frame_indices=tuple(frame_indices),
                )
            )
        predictions_by_window[window.window_index] = window_predictions

    reconciled = reconcile_window_predictions(
        windows=windows,
        predictions_by_window=predictions_by_window,
        n_frames=max((window.end for window in windows), default=0),
    )
    frame_indices = [frame_index for window in windows for frame_index in window.frame_indices]
    ordered_frame_indices = sorted(set(frame_indices), key=frame_indices.index)
    return dict(zip(ordered_frame_indices, reconciled)), chunks


def _run_propainter_face_streams_with_retries(
    windows: list[InpaintWindow],
    face_rgbs_by_frame: dict[int, dict[str, np.ndarray]],
    face_masks_by_frame: dict[int, dict[str, np.ndarray]],
    backend: ProPainterBackend,
    config: IndexConfig,
) -> tuple[dict[int, dict[str, np.ndarray]], list[FaceWindowChunk]]:
    last_oom: Optional[BaseException] = None
    for chunk_size in _propainter_retry_chunk_sizes(config.propainter_chunk_size):
        attempt_config = replace(config, propainter_chunk_size=chunk_size)
        attempt_face_rgbs = _clone_face_rgb_mapping(face_rgbs_by_frame)
        all_chunks: list[FaceWindowChunk] = []
        try:
            for face_name in FACE_ORDER:
                face_outputs, chunks = _process_face_stream(
                    face_name=face_name,
                    windows=windows,
                    face_rgbs_by_frame=attempt_face_rgbs,
                    face_masks_by_frame=face_masks_by_frame,
                    backend=backend,
                    config=attempt_config,
                )
                all_chunks.extend(chunks)
                for frame_index, rgb in face_outputs.items():
                    attempt_face_rgbs[frame_index][face_name] = rgb
            return attempt_face_rgbs, all_chunks
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            last_oom = exc

    raise ProPainterOutOfMemoryError(f"out-of-memory during all retry levels: {last_oom}")


def _mark_single_frame_fallback_unavailable(
    sequence_manifest: SequenceManifest,
    projected_rows: list[FrameRecord],
    processing_windows: list[InpaintWindow],
    failed_rows: dict[int, FrameRecord],
    config: IndexConfig,
    fallback_reason: str,
    propainter_error: str = "",
) -> SequenceManifest:
    error = f"LaMa fallback unavailable: {fallback_reason}"
    if propainter_error:
        error = f"{error}; ProPainter error: {propainter_error}"
    rows: list[FrameRecord] = []
    for frame_index, row in enumerate(projected_rows):
        if frame_index in failed_rows:
            rows.append(failed_rows[frame_index])
            continue
        rows.append(
            replace(
                row,
                propainter_window_count=len(processing_windows),
                propainter_chunk_count=0,
                propainter_status="failed" if row.file_exists else row.propainter_status,
                propainter_error=error if row.file_exists else row.propainter_error,
                single_frame_fallback_reason=fallback_reason if row.file_exists else row.single_frame_fallback_reason,
            )
        )
    return _finish_inpaint_manifest(sequence_manifest, rows, config)


def _run_lama_fallback_sequence(
    sequence_manifest: SequenceManifest,
    projected_rows: list[FrameRecord],
    face_rgbs_by_frame: dict[int, dict[str, np.ndarray]],
    face_masks_by_frame: dict[int, dict[str, np.ndarray]],
    erp_masks_by_frame: dict[int, FrameErpMasks],
    projections_by_frame: dict[int, CubemapProjection],
    original_erp_by_frame: dict[int, np.ndarray],
    failed_rows: dict[int, FrameRecord],
    processing_windows: list[InpaintWindow],
    backend: LaMaBackend,
    config: IndexConfig,
    fallback_reason: str,
    propainter_error: str = "",
) -> SequenceManifest:
    processed_frame_indices = sorted(face_rgbs_by_frame)
    rows: list[FrameRecord] = []
    for frame_index, frame in enumerate(projected_rows):
        if frame_index in failed_rows:
            rows.append(
                replace(
                    failed_rows[frame_index],
                    single_frame_fallback_reason=fallback_reason,
                    lama_model_id=getattr(backend, "model_id", config.lama_model_id),
                    lama_output_dir=sequence_lama_output_dir(sequence_manifest),
                    lama_status="failed",
                    lama_error=failed_rows[frame_index].propainter_error,
                )
            )
            continue
        if frame_index not in face_rgbs_by_frame:
            rows.append(frame)
            continue

        try:
            frame_faces = dict(face_rgbs_by_frame[frame_index])
            for face_name in FACE_ORDER:
                original = frame_faces[face_name]
                mask = face_masks_by_frame[frame_index][face_name]
                prediction = backend.inpaint_image(face_name, original, mask, config)
                frame_faces[face_name] = _compose_masked_prediction(original, prediction, mask)

            projection = projections_by_frame[frame_index]
            for face_name in FACE_ORDER:
                _write_face_artifacts(
                    output_dir=lama_output_dir_for_frame(frame),
                    face_name=face_name,
                    frame=frame,
                    rgb=frame_faces[face_name],
                    mask=face_masks_by_frame[frame_index][face_name],
                    overlap_px=projection.overlap_px,
                    config=config,
                )

            reprojected_rgb = _reproject_inpainted_faces_to_erp(
                projection=projection,
                face_rgbs=frame_faces,
                original_erp_rgb=original_erp_by_frame[frame_index],
                config=config,
            )
            erp_rgb = _compose_masked_prediction(
                original_erp_by_frame[frame_index],
                reprojected_rgb,
                erp_masks_by_frame[frame_index].inpaint,
            )
            _write_frame_outputs(frame, erp_rgb, erp_masks_by_frame[frame_index].sky, config)
            rows.append(
                replace(
                    frame,
                    propainter_window_count=len(processing_windows),
                    propainter_chunk_count=0,
                    propainter_status="fallback_lama",
                    propainter_error=propainter_error,
                    lama_model_id=getattr(backend, "model_id", config.lama_model_id),
                    lama_output_dir=lama_output_dir_for_frame(frame),
                    lama_status="inpainted",
                    lama_error="",
                    single_frame_fallback_reason=fallback_reason,
                )
            )
        except Exception as exc:
            rows.append(
                replace(
                    frame,
                    propainter_window_count=len(processing_windows),
                    propainter_chunk_count=0,
                    propainter_status="failed",
                    propainter_error=propainter_error or fallback_reason,
                    lama_model_id=getattr(backend, "model_id", config.lama_model_id),
                    lama_output_dir=lama_output_dir_for_frame(frame),
                    lama_status="failed",
                    lama_error=str(exc),
                    single_frame_fallback_reason=fallback_reason,
                )
            )

    _write_lama_sequence_metadata(
        sequence_lama_output_dir(sequence_manifest),
        fallback_reason=fallback_reason,
        backend=backend,
        config=config,
        windows=processing_windows,
        processed_frame_indices=processed_frame_indices,
    )
    return _finish_inpaint_manifest(sequence_manifest, rows, config)


def inpaint_sequence_faces(
    sequence_manifest: SequenceManifest,
    config: IndexConfig,
    backend: Optional[ProPainterBackend],
    fallback_backend: Optional[LaMaBackend] = None,
    startup_fallback_reason: str = "",
) -> SequenceManifest:
    if sequence_manifest.status != "ready":
        return sequence_manifest

    windows = build_sequence_inpaint_windows(sequence_manifest, config)
    output_dir = sequence_propainter_output_dir(sequence_manifest)
    if not windows:
        return sequence_manifest

    valid_frame_indices = [frame_index for window in windows for frame_index in window.frame_indices]
    valid_frame_indices = sorted(set(valid_frame_indices), key=valid_frame_indices.index)

    projected_rows = list(sequence_manifest.rows)
    face_rgbs_by_frame: dict[int, dict[str, np.ndarray]] = {}
    face_masks_by_frame: dict[int, dict[str, np.ndarray]] = {}
    erp_masks_by_frame: dict[int, FrameErpMasks] = {}
    projections_by_frame: dict[int, CubemapProjection] = {}
    original_erp_by_frame: dict[int, np.ndarray] = {}
    failed_rows: dict[int, FrameRecord] = {}

    for frame_index in valid_frame_indices:
        frame = sequence_manifest.rows[frame_index]
        try:
            projected_frame, projection = load_or_create_cubemap_projection(frame, config)
            erp_masks = _load_final_erp_masks(projected_frame)
            face_masks = {
                face_name: _sample_erp_mask_to_face(
                    erp_masks.inpaint,
                    projection.faces[face_name].erp_x,
                    projection.faces[face_name].erp_y,
                )
                for face_name in FACE_ORDER
            }
            original_erp = load_erp_rgb(
                projected_frame.resolved_fixed_path,
                compute_checksum=False,
            ).rgb
        except Exception as exc:
            failed_rows[frame_index] = replace(
                frame,
                propainter_model_id=getattr(backend, "model_id", config.propainter_model_id),
                propainter_output_dir=output_dir,
                propainter_window_count=len(windows),
                propainter_status="failed",
                propainter_error=str(exc),
            )
            continue

        projected_rows[frame_index] = projected_frame
        face_rgbs_by_frame[frame_index] = {
            face_name: np.asarray(projection.faces[face_name].rgb, dtype=np.uint8)
            for face_name in FACE_ORDER
        }
        face_masks_by_frame[frame_index] = face_masks
        erp_masks_by_frame[frame_index] = erp_masks
        projections_by_frame[frame_index] = projection
        original_erp_by_frame[frame_index] = original_erp

    loaded_frame_indices = [frame_index for frame_index in valid_frame_indices if frame_index in face_rgbs_by_frame]
    processing_windows = build_inpaint_windows(
        n_frames=len(loaded_frame_indices),
        window_size=config.inpaint_window_size,
        window_stride=config.inpaint_window_stride,
        frame_indices=loaded_frame_indices,
    )
    failed_rows = {
        frame_index: replace(row, propainter_window_count=len(processing_windows))
        for frame_index, row in failed_rows.items()
    }

    fallback_reason = startup_fallback_reason or _select_single_frame_fallback_reason(
        config,
        processing_windows,
        erp_masks_by_frame,
    )
    if fallback_reason:
        if fallback_backend is None:
            return _mark_single_frame_fallback_unavailable(
                sequence_manifest=sequence_manifest,
                projected_rows=projected_rows,
                processing_windows=processing_windows,
                failed_rows=failed_rows,
                config=config,
                fallback_reason=fallback_reason,
            )
        return _run_lama_fallback_sequence(
            sequence_manifest=sequence_manifest,
            projected_rows=projected_rows,
            face_rgbs_by_frame=face_rgbs_by_frame,
            face_masks_by_frame=face_masks_by_frame,
            erp_masks_by_frame=erp_masks_by_frame,
            projections_by_frame=projections_by_frame,
            original_erp_by_frame=original_erp_by_frame,
            failed_rows=failed_rows,
            processing_windows=processing_windows,
            backend=fallback_backend,
            config=config,
            fallback_reason=fallback_reason,
        )

    if backend is None:
        fallback_reason = "propainter_model_load_failure"
        if fallback_backend is None:
            return _mark_single_frame_fallback_unavailable(
                sequence_manifest=sequence_manifest,
                projected_rows=projected_rows,
                processing_windows=processing_windows,
                failed_rows=failed_rows,
                config=config,
                fallback_reason=fallback_reason,
            )
        return _run_lama_fallback_sequence(
            sequence_manifest=sequence_manifest,
            projected_rows=projected_rows,
            face_rgbs_by_frame=face_rgbs_by_frame,
            face_masks_by_frame=face_masks_by_frame,
            erp_masks_by_frame=erp_masks_by_frame,
            projections_by_frame=projections_by_frame,
            original_erp_by_frame=original_erp_by_frame,
            failed_rows=failed_rows,
            processing_windows=processing_windows,
            backend=fallback_backend,
            config=config,
            fallback_reason=fallback_reason,
        )

    try:
        face_rgbs_by_frame, all_chunks = _run_propainter_face_streams_with_retries(
            windows=processing_windows,
            face_rgbs_by_frame=face_rgbs_by_frame,
            face_masks_by_frame=face_masks_by_frame,
            backend=backend,
            config=config,
        )
    except Exception as exc:
        failed_error = str(exc)
        if fallback_backend is not None:
            fallback_reason = (
                "propainter_oom_all_retry_levels"
                if isinstance(exc, ProPainterOutOfMemoryError)
                else f"propainter_inference_failed: {failed_error}"
            )
            return _run_lama_fallback_sequence(
                sequence_manifest=sequence_manifest,
                projected_rows=projected_rows,
                face_rgbs_by_frame=face_rgbs_by_frame,
                face_masks_by_frame=face_masks_by_frame,
                erp_masks_by_frame=erp_masks_by_frame,
                projections_by_frame=projections_by_frame,
                original_erp_by_frame=original_erp_by_frame,
                failed_rows=failed_rows,
                processing_windows=processing_windows,
                backend=fallback_backend,
                config=config,
                fallback_reason=fallback_reason,
                propainter_error=failed_error,
            )
        inpainted_rows = [
            replace(
                row,
                propainter_model_id=getattr(backend, "model_id", config.propainter_model_id),
                propainter_output_dir=output_dir,
                propainter_window_count=len(processing_windows),
                propainter_chunk_count=0,
                propainter_status="failed" if row.file_exists else row.propainter_status,
                propainter_error=failed_error if row.file_exists else row.propainter_error,
            )
            for row in projected_rows
        ]
        return _finish_inpaint_manifest(sequence_manifest, inpainted_rows, config)

    inpainted_rows: list[FrameRecord] = []
    for frame_index, frame in enumerate(projected_rows):
        if frame_index in failed_rows:
            inpainted_rows.append(failed_rows[frame_index])
            continue
        if frame_index not in face_rgbs_by_frame:
            inpainted_rows.append(frame)
            continue

        projection = projections_by_frame[frame_index]
        for face_name in FACE_ORDER:
            _write_face_artifacts(
                output_dir=propainter_output_dir_for_frame(frame),
                face_name=face_name,
                frame=frame,
                rgb=face_rgbs_by_frame[frame_index][face_name],
                mask=face_masks_by_frame[frame_index][face_name],
                overlap_px=projection.overlap_px,
                config=config,
            )
        reprojected_rgb = _reproject_inpainted_faces_to_erp(
            projection=projection,
            face_rgbs=face_rgbs_by_frame[frame_index],
            original_erp_rgb=original_erp_by_frame[frame_index],
            config=config,
        )
        erp_rgb = _compose_masked_prediction(
            original_erp_by_frame[frame_index],
            reprojected_rgb,
            erp_masks_by_frame[frame_index].inpaint,
        )
        _write_frame_outputs(frame, erp_rgb, erp_masks_by_frame[frame_index].sky, config)
        inpainted_rows.append(
            replace(
                frame,
                propainter_model_id=getattr(backend, "model_id", config.propainter_model_id),
                propainter_output_dir=propainter_output_dir_for_frame(frame),
                propainter_window_count=len(processing_windows),
                propainter_chunk_count=len(all_chunks),
                propainter_status="inpainted",
                propainter_error="",
            )
        )

    artifacts = SequenceInpaintArtifacts(output_dir=output_dir, windows=processing_windows, chunks=all_chunks)
    _write_sequence_metadata(output_dir, artifacts, backend, config)
    return _finish_inpaint_manifest(sequence_manifest, inpainted_rows, config)


def _finish_inpaint_manifest(
    original_manifest: SequenceManifest,
    rows: list[FrameRecord],
    config: IndexConfig,
) -> SequenceManifest:
    failed_rows = [row for row in rows if row.propainter_status == "failed"]
    status = original_manifest.status
    failure_reason = original_manifest.failure_reason
    if failed_rows:
        status = "failed_propainter_inpainting"
        failure_reason = f"{len(failed_rows)} frame(s) failed ProPainter inpainting"

    manifest = SequenceManifest(
        sequence_id=original_manifest.sequence_id,
        sequence_dir=original_manifest.sequence_dir,
        csv_path=original_manifest.csv_path,
        output_dir=original_manifest.output_dir,
        status=status,
        failure_reason=failure_reason,
        total_csv_rows=original_manifest.total_csv_rows,
        valid_frames=original_manifest.valid_frames,
        skipped_frames=original_manifest.skipped_frames,
        rows=rows,
    )
    if not config.dry_run:
        write_sequence_manifest(manifest)
    return manifest


def run_propainter_inpainting(
    config: IndexConfig,
    sequence_ids: Optional[Iterable[str]] = None,
    backend: Any = None,
    fallback_backend: Any = None,
) -> list[SequenceManifest]:
    if fallback_backend is not None:
        lama_backend = fallback_backend
    elif config.lama_command:
        lama_backend = ExternalLaMaBackend.from_config(config)
    elif config.dry_run:
        lama_backend = PassthroughLaMaBackend()
    else:
        lama_backend = None

    startup_fallback_reason = ""
    if config.force_single_frame_fallback:
        propainter_backend: Optional[ProPainterBackend] = None
    else:
        try:
            if backend is not None:
                propainter_backend = backend
            elif config.dry_run and not config.propainter_command:
                propainter_backend = PassthroughProPainterBackend()
            else:
                propainter_backend = ExternalProPainterBackend.from_config(config)
        except Exception as exc:
            if lama_backend is None:
                raise
            propainter_backend = None
            startup_fallback_reason = f"propainter_model_load_failure: {exc}"

    indexed_manifests = run_indexing(config, sequence_ids=sequence_ids)
    return [
        inpaint_sequence_faces(
            manifest,
            config,
            propainter_backend,
            fallback_backend=lama_backend,
            startup_fallback_reason=startup_fallback_reason,
        )
        for manifest in indexed_manifests
    ]
