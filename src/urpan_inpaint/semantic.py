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


TARGET_CLASS_ALIASES = {
    "sky": ("sky",),
    "person": ("person", "pedestrian"),
    "rider": ("rider",),
    "bicycle": ("bicycle", "bike"),
    "motorcycle": ("motorcycle", "motorbike"),
    "car": ("car",),
    "truck": ("truck",),
    "bus": ("bus",),
    "train": ("train", "on rails", "on rails vehicle", "rail"),
    "trailer": ("trailer", "caravan"),
    "building": ("building",),
    "tree": ("tree",),
    "vegetation": ("vegetation", "terrain vegetation"),
}


class SemanticRuntimeError(RuntimeError):
    """Raised when semantic parsing runtime dependencies or model weights are unavailable."""


@dataclass(frozen=True)
class SemanticFacePrediction:
    face_name: str
    label_map: np.ndarray
    confidence_map: Optional[np.ndarray]
    semantic_scores: Optional[np.ndarray]
    panoptic_instance_map: Optional[np.ndarray]
    panoptic_segments_info: list[dict[str, Any]]


def _normalize_label_name(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


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


def resolve_target_class_ids(id2label: dict[int, str]) -> dict[str, Optional[int]]:
    normalized = {_normalize_label_name(label): int(class_id) for class_id, label in id2label.items()}
    resolved: dict[str, Optional[int]] = {}
    for target_name, aliases in TARGET_CLASS_ALIASES.items():
        class_id = None
        for alias in aliases:
            class_id = normalized.get(_normalize_label_name(alias))
            if class_id is not None:
                break
        resolved[target_name] = class_id
    return resolved


def build_target_masks(label_map: np.ndarray, target_class_ids: dict[str, Optional[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_names: list[str] = []
    target_ids: list[int] = []
    target_masks: list[np.ndarray] = []
    for target_name in TARGET_CLASS_ALIASES:
        class_id = target_class_ids.get(target_name)
        if class_id is None:
            continue
        target_names.append(target_name)
        target_ids.append(class_id)
        target_masks.append(np.where(label_map == class_id, 255, 0).astype(np.uint8))

    if target_masks:
        stacked_masks = np.stack(target_masks, axis=0)
    else:
        stacked_masks = np.zeros((0, label_map.shape[0], label_map.shape[1]), dtype=np.uint8)
    return stacked_masks, np.asarray(target_ids, dtype=np.int32), np.asarray(target_names, dtype="<U32")


class Mask2FormerFaceParser:
    def __init__(
        self,
        model: Any,
        image_processor: Any,
        torch_module: Any,
        functional_module: Any,
        model_id: str,
        device: str,
        save_logits: bool,
        save_confidence: bool,
        attempt_panoptic: bool,
    ) -> None:
        self.model = model
        self.image_processor = image_processor
        self.torch = torch_module
        self.functional = functional_module
        self.model_id = model_id
        self.device = device
        self.save_logits = save_logits
        self.save_confidence = save_confidence
        self.attempt_panoptic = attempt_panoptic
        raw_id2label = getattr(model.config, "id2label", {})
        self.id2label = {int(class_id): str(label) for class_id, label in raw_id2label.items()}
        self.target_class_ids = resolve_target_class_ids(self.id2label)

    @classmethod
    def from_config(cls, config: IndexConfig) -> "Mask2FormerFaceParser":
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        except ImportError as exc:
            raise SemanticRuntimeError(
                "Semantic parsing requires the 'torch' and 'transformers' packages in the selected environment."
            ) from exc

        device = cls._resolve_device(torch, config.semantic_device)
        image_processor = AutoImageProcessor.from_pretrained(
            config.semantic_model_id,
            local_files_only=config.semantic_local_files_only,
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            config.semantic_model_id,
            local_files_only=config.semantic_local_files_only,
        )
        model.to(device)
        model.eval()
        return cls(
            model=model,
            image_processor=image_processor,
            torch_module=torch,
            functional_module=functional,
            model_id=config.semantic_model_id,
            device=str(device),
            save_logits=config.semantic_save_logits,
            save_confidence=config.semantic_save_confidence,
            attempt_panoptic=config.semantic_attempt_panoptic,
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

    def parse_face(self, face_name: str, rgb: np.ndarray) -> SemanticFacePrediction:
        image = Image.fromarray(rgb, mode="RGB")
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            semantic_scores = self._semantic_scores_from_outputs(outputs, rgb.shape[:2])
            confidence_map = None
            if self.save_confidence:
                confidence_map = semantic_scores.max(dim=0).values.detach().cpu().numpy().astype(np.float32)
            label_map = semantic_scores.argmax(dim=0).detach().cpu().numpy().astype(np.int32)
            logits_to_save = None
            if self.save_logits:
                logits_to_save = semantic_scores.detach().cpu().numpy().astype(np.float16)

        panoptic_instance_map = None
        panoptic_segments_info: list[dict[str, Any]] = []
        if self.attempt_panoptic and hasattr(self.image_processor, "post_process_panoptic_segmentation"):
            try:
                panoptic = self.image_processor.post_process_panoptic_segmentation(
                    outputs,
                    target_sizes=[rgb.shape[:2]],
                )[0]
                panoptic_instance_map = panoptic["segmentation"].detach().cpu().numpy().astype(np.int32)
                panoptic_segments_info = [
                    {
                        **{key: _to_builtin(value) for key, value in segment.items()},
                        "label_name": self.id2label.get(int(segment.get("label_id", -1)), ""),
                    }
                    for segment in panoptic.get("segments_info", [])
                ]
            except Exception:
                panoptic_instance_map = None
                panoptic_segments_info = []

        return SemanticFacePrediction(
            face_name=face_name,
            label_map=label_map,
            confidence_map=confidence_map,
            semantic_scores=logits_to_save,
            panoptic_instance_map=panoptic_instance_map,
            panoptic_segments_info=panoptic_segments_info,
        )

    def _semantic_scores_from_outputs(self, outputs: Any, target_hw: tuple[int, int]) -> Any:
        if not hasattr(outputs, "class_queries_logits") or not hasattr(outputs, "masks_queries_logits"):
            raise SemanticRuntimeError("Mask2Former outputs are missing class or mask query logits.")

        class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        mask_probs = outputs.masks_queries_logits.sigmoid()
        semantic_scores = self.torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        semantic_scores = self.functional.interpolate(
            semantic_scores,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        return semantic_scores[0]


def semantic_output_dir_for_frame(frame: FrameRecord) -> Path:
    return frame.cubemap_cache_dir / "semantic_mask2former"


def _write_face_prediction(
    output_dir: Path,
    prediction: SemanticFacePrediction,
    target_class_ids: dict[str, Optional[int]],
    save_logits: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_masks, target_ids, target_names = build_target_masks(prediction.label_map, target_class_ids)

    payload: dict[str, Any] = {
        "label_map": prediction.label_map.astype(np.int32),
        "target_masks": target_masks,
        "target_class_ids": target_ids,
        "target_class_names": target_names,
    }
    if prediction.confidence_map is not None:
        payload["confidence_map"] = prediction.confidence_map.astype(np.float16)
    if save_logits and prediction.semantic_scores is not None:
        payload["semantic_scores"] = prediction.semantic_scores
    if prediction.panoptic_instance_map is not None:
        payload["panoptic_instance_map"] = prediction.panoptic_instance_map.astype(np.int32)

    np.savez_compressed(output_dir / f"{prediction.face_name}.npz", **payload)

    if prediction.panoptic_segments_info:
        panoptic_path = output_dir / f"{prediction.face_name}.panoptic.json"
        panoptic_path.write_text(
            json.dumps({"segments_info": prediction.panoptic_segments_info}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_semantic_metadata(
    output_dir: Path,
    parser: Any,
    target_class_ids: dict[str, Optional[int]],
    has_panoptic: bool,
    has_confidence: bool,
    has_logits: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "model_id": getattr(parser, "model_id", ""),
        "face_order": list(FACE_ORDER),
        "target_class_ids": target_class_ids,
        "missing_target_classes": [name for name, class_id in target_class_ids.items() if class_id is None],
        "id2label": {str(class_id): label for class_id, label in getattr(parser, "id2label", {}).items()},
        "has_panoptic": has_panoptic,
        "has_confidence": has_confidence,
        "has_logits": has_logits,
        "face_cache_format": "npz",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def parse_frame_semantic(frame: FrameRecord, config: IndexConfig, parser: Any) -> FrameRecord:
    try:
        projected_frame, _, face_rgbs = load_or_create_cubemap_face_rgbs(frame, config)
    except Exception as exc:
        return replace(
            frame,
            semantic_model_id=getattr(parser, "model_id", config.semantic_model_id),
            semantic_parse_status="failed",
            semantic_parse_error=str(exc),
        )

    semantic_output_dir = semantic_output_dir_for_frame(projected_frame)
    predictions: list[SemanticFacePrediction] = []
    try:
        for face_name in FACE_ORDER:
            predictions.append(parser.parse_face(face_name, face_rgbs[face_name]))
    except Exception as exc:
        return replace(
            projected_frame,
            semantic_model_id=getattr(parser, "model_id", config.semantic_model_id),
            semantic_output_dir=semantic_output_dir,
            semantic_parse_status="failed",
            semantic_parse_error=str(exc),
        )

    has_panoptic = any(pred.panoptic_instance_map is not None for pred in predictions)
    has_confidence = any(pred.confidence_map is not None for pred in predictions)
    has_logits = any(pred.semantic_scores is not None for pred in predictions)

    if not config.dry_run:
        for prediction in predictions:
            _write_face_prediction(
                semantic_output_dir,
                prediction,
                getattr(parser, "target_class_ids", {}),
                save_logits=config.semantic_save_logits,
            )
        _write_semantic_metadata(
            semantic_output_dir,
            parser,
            getattr(parser, "target_class_ids", {}),
            has_panoptic=has_panoptic,
            has_confidence=has_confidence,
            has_logits=has_logits,
        )

    return replace(
        projected_frame,
        semantic_model_id=getattr(parser, "model_id", config.semantic_model_id),
        semantic_output_dir=semantic_output_dir,
        semantic_has_panoptic=has_panoptic,
        semantic_has_confidence=has_confidence,
        semantic_has_logits=has_logits,
        semantic_parse_status="parsed",
        semantic_parse_error="",
    )


def parse_sequence_semantic(sequence_manifest: SequenceManifest, config: IndexConfig, parser: Any) -> SequenceManifest:
    if sequence_manifest.status != "ready":
        return sequence_manifest

    parsed_rows = [parse_frame_semantic(frame, config, parser) for frame in sequence_manifest.rows]
    failed_rows = [row for row in parsed_rows if row.semantic_parse_status == "failed"]
    status = sequence_manifest.status
    failure_reason = sequence_manifest.failure_reason
    if failed_rows:
        status = "failed_semantic_parsing"
        failure_reason = f"{len(failed_rows)} frame(s) failed semantic parsing"

    parsed_manifest = SequenceManifest(
        sequence_id=sequence_manifest.sequence_id,
        sequence_dir=sequence_manifest.sequence_dir,
        csv_path=sequence_manifest.csv_path,
        output_dir=sequence_manifest.output_dir,
        status=status,
        failure_reason=failure_reason,
        total_csv_rows=sequence_manifest.total_csv_rows,
        valid_frames=sequence_manifest.valid_frames,
        skipped_frames=sequence_manifest.skipped_frames,
        rows=parsed_rows,
    )

    if not config.dry_run:
        write_sequence_manifest(parsed_manifest)

    return parsed_manifest


def run_semantic_parsing(
    config: IndexConfig,
    sequence_ids: Optional[Iterable[str]] = None,
    parser: Any = None,
) -> list[SequenceManifest]:
    semantic_parser = parser if parser is not None else Mask2FormerFaceParser.from_config(config)
    indexed_manifests = run_indexing(config, sequence_ids=sequence_ids)
    return [parse_sequence_semantic(manifest, config, semantic_parser) for manifest in indexed_manifests]
