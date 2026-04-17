from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.cubemap import FACE_ORDER
from urpan_inpaint.discovery import run_indexing, write_sequence_manifest
from urpan_inpaint.models import FrameRecord, SequenceManifest
from urpan_inpaint.projection import load_or_create_cubemap_face_rgbs


class GroundingRuntimeError(RuntimeError):
    """Raised when Grounding DINO runtime dependencies or model weights are unavailable."""


@dataclass(frozen=True)
class FaceDetection:
    box_xyxy: np.ndarray
    score: float
    text: str
    text_normalized: str


@dataclass(frozen=True)
class FaceDetections:
    face_name: str
    detections: list[FaceDetection]
    raw_detection_count: int


def _normalize_detection_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def compute_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
    bx0, by0, bx1, by1 = [float(value) for value in box_b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def class_aware_nms(detections: list[FaceDetection], iou_threshold: float) -> list[FaceDetection]:
    kept: list[FaceDetection] = []
    for class_name in sorted({det.text_normalized for det in detections}):
        class_detections = [det for det in detections if det.text_normalized == class_name]
        class_detections.sort(key=lambda det: det.score, reverse=True)
        while class_detections:
            current = class_detections.pop(0)
            kept.append(current)
            survivors: list[FaceDetection] = []
            for candidate in class_detections:
                if compute_iou_xyxy(current.box_xyxy, candidate.box_xyxy) <= iou_threshold:
                    survivors.append(candidate)
            class_detections = survivors
    kept.sort(key=lambda det: det.score, reverse=True)
    return kept


class GroundingDinoFaceDetector:
    def __init__(
        self,
        model: Any,
        processor: Any,
        torch_module: Any,
        model_id: str,
        device: str,
        prompts: tuple[str, ...],
        box_threshold: float,
        text_threshold: float,
        nms_iou_threshold: float,
    ) -> None:
        self.model = model
        self.processor = processor
        self.torch = torch_module
        self.model_id = model_id
        self.device = device
        self.prompts = prompts
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_threshold = nms_iou_threshold

    @classmethod
    def from_config(cls, config: IndexConfig) -> "GroundingDinoFaceDetector":
        try:
            import torch
            from transformers import AutoProcessor, GroundingDinoForObjectDetection
        except ImportError as exc:
            raise GroundingRuntimeError(
                "Open-vocabulary detection requires the 'torch' and 'transformers' packages in the selected environment."
            ) from exc

        device = cls._resolve_device(torch, config.grounding_device)
        processor = AutoProcessor.from_pretrained(
            config.grounding_model_id,
            local_files_only=config.grounding_local_files_only,
        )
        model = GroundingDinoForObjectDetection.from_pretrained(
            config.grounding_model_id,
            local_files_only=config.grounding_local_files_only,
        )
        model.to(device)
        model.eval()
        return cls(
            model=model,
            processor=processor,
            torch_module=torch,
            model_id=config.grounding_model_id,
            device=str(device),
            prompts=config.grounding_prompts,
            box_threshold=config.grounding_box_threshold,
            text_threshold=config.grounding_text_threshold,
            nms_iou_threshold=config.grounding_nms_iou_threshold,
        )

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> Any:
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(requested)

    def detect_face(self, face_name: str, rgb: np.ndarray) -> FaceDetections:
        image = Image.fromarray(rgb, mode="RGB")
        inputs = self.processor(images=image, text=list(self.prompts), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[rgb.shape[:2]],
            )[0]

        boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)
        text_labels = [str(text).strip() for text in result.get("text_labels", result.get("labels", []))]

        detections = [
            FaceDetection(
                box_xyxy=boxes[index],
                score=float(scores[index]),
                text=text_labels[index],
                text_normalized=_normalize_detection_text(text_labels[index]),
            )
            for index in range(len(text_labels))
            if float(scores[index]) >= self.box_threshold
        ]
        filtered = class_aware_nms(detections, self.nms_iou_threshold)
        return FaceDetections(face_name=face_name, detections=filtered, raw_detection_count=len(detections))


def grounding_output_dir_for_frame(frame: FrameRecord) -> Path:
    return frame.cubemap_cache_dir / "grounding_dino"


def _write_face_detections(output_dir: Path, face_detections: FaceDetections) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    boxes = np.asarray([det.box_xyxy for det in face_detections.detections], dtype=np.float32)
    scores = np.asarray([det.score for det in face_detections.detections], dtype=np.float32)
    text_labels = np.asarray([det.text for det in face_detections.detections], dtype="<U128")
    text_labels_normalized = np.asarray([det.text_normalized for det in face_detections.detections], dtype="<U128")

    if boxes.size == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
    np.savez_compressed(
        output_dir / f"{face_detections.face_name}.npz",
        boxes_xyxy=boxes,
        scores=scores,
        text_labels=text_labels,
        text_labels_normalized=text_labels_normalized,
    )

    sidecar = {
        "face_name": face_detections.face_name,
        "raw_detection_count": face_detections.raw_detection_count,
        "filtered_detection_count": len(face_detections.detections),
        "detections": [
            {
                "box_xyxy": [float(value) for value in det.box_xyxy.tolist()],
                "score": float(det.score),
                "text": det.text,
                "text_normalized": det.text_normalized,
            }
            for det in face_detections.detections
        ],
    }
    (output_dir / f"{face_detections.face_name}.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_grounding_metadata(output_dir: Path, detector: Any) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "model_id": getattr(detector, "model_id", ""),
        "prompts": list(getattr(detector, "prompts", ())),
        "box_threshold": _to_builtin(getattr(detector, "box_threshold", None)),
        "text_threshold": _to_builtin(getattr(detector, "text_threshold", None)),
        "nms_iou_threshold": _to_builtin(getattr(detector, "nms_iou_threshold", None)),
        "face_order": list(FACE_ORDER),
        "face_cache_format": "npz",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def detect_frame_grounding(frame: FrameRecord, config: IndexConfig, detector: Any) -> FrameRecord:
    try:
        projected_frame, _, face_rgbs = load_or_create_cubemap_face_rgbs(frame, config)
    except Exception as exc:
        return replace(
            frame,
            grounding_model_id=getattr(detector, "model_id", config.grounding_model_id),
            grounding_detect_status="failed",
            grounding_detect_error=str(exc),
        )

    grounding_output_dir = grounding_output_dir_for_frame(projected_frame)
    face_results: list[FaceDetections] = []
    try:
        for face_name in FACE_ORDER:
            face_results.append(detector.detect_face(face_name, face_rgbs[face_name]))
    except Exception as exc:
        return replace(
            projected_frame,
            grounding_model_id=getattr(detector, "model_id", config.grounding_model_id),
            grounding_output_dir=grounding_output_dir,
            grounding_detect_status="failed",
            grounding_detect_error=str(exc),
        )

    total_boxes = sum(len(face_result.detections) for face_result in face_results)
    if not config.dry_run:
        for face_result in face_results:
            _write_face_detections(grounding_output_dir, face_result)
        _write_grounding_metadata(grounding_output_dir, detector)

    return replace(
        projected_frame,
        grounding_model_id=getattr(detector, "model_id", config.grounding_model_id),
        grounding_output_dir=grounding_output_dir,
        grounding_box_count=total_boxes,
        grounding_detect_status="detected",
        grounding_detect_error="",
    )


def detect_sequence_grounding(sequence_manifest: SequenceManifest, config: IndexConfig, detector: Any) -> SequenceManifest:
    if sequence_manifest.status != "ready":
        return sequence_manifest

    detected_rows = [detect_frame_grounding(frame, config, detector) for frame in sequence_manifest.rows]
    failed_rows = [row for row in detected_rows if row.grounding_detect_status == "failed"]
    status = sequence_manifest.status
    failure_reason = sequence_manifest.failure_reason
    if failed_rows:
        status = "failed_grounding_detection"
        failure_reason = f"{len(failed_rows)} frame(s) failed Grounding DINO detection"

    detected_manifest = SequenceManifest(
        sequence_id=sequence_manifest.sequence_id,
        sequence_dir=sequence_manifest.sequence_dir,
        csv_path=sequence_manifest.csv_path,
        output_dir=sequence_manifest.output_dir,
        status=status,
        failure_reason=failure_reason,
        total_csv_rows=sequence_manifest.total_csv_rows,
        valid_frames=sequence_manifest.valid_frames,
        skipped_frames=sequence_manifest.skipped_frames,
        rows=detected_rows,
    )

    if not config.dry_run:
        write_sequence_manifest(detected_manifest)

    return detected_manifest


def run_grounding_detection(
    config: IndexConfig,
    sequence_ids: Optional[Iterable[str]] = None,
    detector: Any = None,
) -> list[SequenceManifest]:
    grounding_detector = detector if detector is not None else GroundingDinoFaceDetector.from_config(config)
    indexed_manifests = run_indexing(config, sequence_ids=sequence_ids)
    return [detect_sequence_grounding(manifest, config, grounding_detector) for manifest in indexed_manifests]
