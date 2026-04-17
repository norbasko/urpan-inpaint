from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.cubemap import FACE_ORDER, cubemap_face_mask_to_erp
from urpan_inpaint.detection import grounding_output_dir_for_frame
from urpan_inpaint.discovery import run_indexing, write_sequence_manifest
from urpan_inpaint.models import FrameRecord, SequenceManifest
from urpan_inpaint.projection import load_or_create_cubemap_face_rgbs
from urpan_inpaint.semantic import semantic_output_dir_for_frame


DYNAMIC_PROMPT_CLASSES = {
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


class Sam2RuntimeError(RuntimeError):
    """Raised when SAM 2 runtime dependencies or model weights are unavailable."""


@dataclass(frozen=True)
class Sam2Prompt:
    prompt_id: str
    face_name: str
    class_text: str
    source: str
    box_xyxy: np.ndarray
    score: float
    point_xy: Optional[np.ndarray] = None
    prior_mask: Optional[np.ndarray] = None
    used_temporal_prior: bool = False


@dataclass(frozen=True)
class RefinedMask:
    prompt_id: str
    face_name: str
    class_text: str
    source: str
    mask: np.ndarray
    box_xyxy: np.ndarray
    score: float
    prompt_box_xyxy: np.ndarray
    prompt_score: float
    used_temporal_prior: bool


@dataclass(frozen=True)
class RoofMaskResult:
    down_mask: np.ndarray
    source: str
    temporal_disagreement: bool
    error: str = ""


@dataclass(frozen=True)
class FrameRefinementResult:
    frame: FrameRecord
    masks: list[RefinedMask]
    face_shapes: dict[str, tuple[int, int]]


@dataclass
class Sam2VideoTrack:
    obj_id: int
    face_name: str
    class_text: str
    source: str
    last_box_xyxy: np.ndarray
    last_frame_index: int
    last_prompt_frame_index: int
    prompt_id: str
    prompt_box_xyxy: np.ndarray
    prompt_score: float


def sam2_output_dir_for_frame(frame: FrameRecord) -> Path:
    return frame.cubemap_cache_dir / "sam2_refined"


def _normalize_class_text(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())


def _mask_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _box_center(box_xyxy: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    return np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)


def _box_area(box_xyxy: np.ndarray) -> float:
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
    bx0, by0, bx1, by1 = [float(value) for value in box_b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = _box_area(box_a) + _box_area(box_b) - inter
    return 0.0 if union <= 0.0 else inter / union


def _clip_box(box_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    box = np.asarray(box_xyxy, dtype=np.float32).copy()
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
    if box[2] <= box[0]:
        box[2] = min(float(width), box[0] + 1.0)
    if box[3] <= box[1]:
        box[3] = min(float(height), box[1] + 1.0)
    return box


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return int(np.logical_and(a, b).sum()) / union


def _roof_prior_fraction(config: IndexConfig, expanded: bool) -> float:
    base = float(np.clip(config.sam2_roof_box_fraction, 0.05, 1.0))
    if not expanded:
        return base
    margin = max(0.0, float(config.sam2_roof_prior_margin_fraction))
    return float(np.clip(base + margin, base, 1.0))


def _roof_prior_mask(width: int, height: int, config: IndexConfig, expanded: bool = False) -> np.ndarray:
    fraction = _roof_prior_fraction(config, expanded=expanded)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    rx = max(1.0, width * fraction * 0.5)
    ry = max(1.0, height * fraction * 0.5)
    yy, xx = np.ogrid[:height, :width]
    ellipse = (((xx - cx) / rx) ** 2) + (((yy - cy) / ry) ** 2) <= 1.0
    return np.where(ellipse, 255, 0).astype(np.uint8)


def _regularize_roof_down_mask(mask: np.ndarray, config: IndexConfig) -> np.ndarray:
    height, width = mask.shape[:2]
    core_prior = _roof_prior_mask(width, height, config, expanded=False) > 0
    support = _roof_prior_mask(width, height, config, expanded=True) > 0
    regularized = np.logical_or(mask > 0, core_prior)
    regularized = np.logical_and(regularized, support)
    return np.where(regularized, 255, 0).astype(np.uint8)


def _load_grounding_prompts(frame: FrameRecord, face_name: str, width: int, height: int) -> list[Sam2Prompt]:
    face_path = grounding_output_dir_for_frame(frame) / f"{face_name}.npz"
    if not face_path.is_file():
        return []

    prompts: list[Sam2Prompt] = []
    with np.load(face_path) as payload:
        boxes = payload["boxes_xyxy"].astype(np.float32)
        scores = payload["scores"].astype(np.float32)
        labels = [str(item) for item in payload["text_labels"]]
    for index, box in enumerate(boxes):
        class_text = _normalize_class_text(labels[index])
        prompts.append(
            Sam2Prompt(
                prompt_id=f"{face_name}:grounding:{index}:{class_text}",
                face_name=face_name,
                class_text=class_text,
                source="grounding_box",
                box_xyxy=_clip_box(box, width, height),
                score=float(scores[index]),
            )
        )
    return prompts


def _load_semantic_prompts(
    frame: FrameRecord,
    face_name: str,
    width: int,
    height: int,
    config: IndexConfig,
) -> list[Sam2Prompt]:
    face_path = semantic_output_dir_for_frame(frame) / f"{face_name}.npz"
    if not face_path.is_file():
        return []

    wanted = {_normalize_class_text(item) for item in config.sam2_semantic_prompt_classes}
    prompts: list[Sam2Prompt] = []
    with np.load(face_path) as payload:
        if "target_masks" not in payload or "target_class_names" not in payload:
            return []
        masks = payload["target_masks"]
        names = [str(item) for item in payload["target_class_names"]]

    for index, mask in enumerate(masks):
        class_text = _normalize_class_text(names[index])
        if class_text not in wanted:
            continue
        binary = mask > 0
        if int(binary.sum()) < config.sam2_min_mask_area_px:
            continue
        bbox = _mask_bbox(binary.astype(np.uint8))
        if bbox is None:
            continue
        prompts.append(
            Sam2Prompt(
                prompt_id=f"{face_name}:semantic:{index}:{class_text}",
                face_name=face_name,
                class_text=class_text,
                source="semantic_region",
                box_xyxy=_clip_box(bbox, width, height),
                score=1.0,
                point_xy=_box_center(bbox),
            )
        )
    return prompts


def _roof_prompt(face_name: str, width: int, height: int, config: IndexConfig) -> list[Sam2Prompt]:
    if face_name != "down" or not config.sam2_refine_roof:
        return []
    fraction = _roof_prior_fraction(config, expanded=True)
    half_w = width * fraction * 0.5
    half_h = height * fraction * 0.5
    cx = width * 0.5
    cy = height * 0.5
    box = np.asarray([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=np.float32)
    prior_mask = _roof_prior_mask(width, height, config, expanded=False)
    return [
        Sam2Prompt(
            prompt_id="down:roof:nadir",
            face_name="down",
            class_text="roof",
            source="roof_down_prior",
            box_xyxy=_clip_box(box, width, height),
            score=1.0,
            point_xy=np.asarray([cx, cy], dtype=np.float32),
            prior_mask=prior_mask,
        )
    ]


def collect_face_prompts(
    frame: FrameRecord,
    face_name: str,
    width: int,
    height: int,
    config: IndexConfig,
) -> list[Sam2Prompt]:
    prompts: list[Sam2Prompt] = []
    if config.sam2_refine_grounding:
        prompts.extend(_load_grounding_prompts(frame, face_name, width, height))
    if config.sam2_refine_semantic:
        prompts.extend(_load_semantic_prompts(frame, face_name, width, height, config))
    prompts.extend(_roof_prompt(face_name, width, height, config))
    return prompts


@dataclass
class TemporalMaskMemory:
    masks_by_key: dict[tuple[str, str, str], RefinedMask]
    frame_index_by_key: dict[tuple[str, str, str], int]

    @classmethod
    def create(cls) -> "TemporalMaskMemory":
        return cls(masks_by_key={}, frame_index_by_key={})

    def attach_priors(
        self,
        prompts: list[Sam2Prompt],
        frame_index: int,
        config: IndexConfig,
        face_name: Optional[str] = None,
    ) -> tuple[list[Sam2Prompt], int]:
        if not config.sam2_temporal_propagation:
            return prompts, 0

        prompts_with_priors: list[Sam2Prompt] = []
        prior_count = 0
        seen_keys: set[tuple[str, str, str]] = set()
        for prompt in prompts:
            key = (prompt.face_name, prompt.class_text, prompt.source)
            seen_keys.add(key)
            prior = self.masks_by_key.get(key)
            prior_frame = self.frame_index_by_key.get(key)
            if prior is None or prior_frame is None or frame_index - prior_frame > config.sam2_temporal_max_gap:
                prompts_with_priors.append(prompt)
                continue
            if self._is_stable(prompt, prior, config):
                prior_mask = prior.mask
                if prompt.prior_mask is not None:
                    prior_mask = np.maximum(prompt.prior_mask, prior.mask).astype(np.uint8)
                prompts_with_priors.append(
                    Sam2Prompt(
                        prompt_id=prompt.prompt_id,
                        face_name=prompt.face_name,
                        class_text=prompt.class_text,
                        source=prompt.source,
                        box_xyxy=prompt.box_xyxy,
                        score=prompt.score,
                        point_xy=prompt.point_xy,
                        prior_mask=prior_mask,
                        used_temporal_prior=True,
                    )
                )
                prior_count += 1
            else:
                prompts_with_priors.append(prompt)

        for key, prior in self.masks_by_key.items():
            if face_name is not None and prior.face_name != face_name:
                continue
            if key in seen_keys:
                continue
            prior_frame = self.frame_index_by_key.get(key, -999999)
            if frame_index - prior_frame > config.sam2_temporal_max_gap:
                continue
            if prior.class_text not in {"roof", "sky"}:
                continue
            prompts_with_priors.append(
                Sam2Prompt(
                    prompt_id=f"{prior.face_name}:temporal:{prior.class_text}:{frame_index}",
                    face_name=prior.face_name,
                    class_text=prior.class_text,
                    source="temporal_prior",
                    box_xyxy=prior.box_xyxy,
                    score=prior.score,
                    point_xy=_box_center(prior.box_xyxy),
                    prior_mask=prior.mask,
                    used_temporal_prior=True,
                )
            )
            prior_count += 1
        return prompts_with_priors, prior_count

    def update(self, masks: Iterable[RefinedMask], frame_index: int) -> None:
        for mask in masks:
            key = (mask.face_name, mask.class_text, mask.source)
            self.masks_by_key[key] = mask
            self.frame_index_by_key[key] = frame_index

    @staticmethod
    def _is_stable(prompt: Sam2Prompt, prior: RefinedMask, config: IndexConfig) -> bool:
        if prompt.class_text == "roof" and prompt.face_name == "down":
            return True
        iou = _box_iou(prompt.box_xyxy, prior.box_xyxy)
        if iou < config.sam2_temporal_iou_threshold:
            return False
        prior_area = max(1.0, _box_area(prior.box_xyxy))
        ratio = _box_area(prompt.box_xyxy) / prior_area
        return config.sam2_temporal_area_ratio_min <= ratio <= config.sam2_temporal_area_ratio_max


class Sam2FaceRefiner:
    def __init__(
        self,
        model: Any,
        processor: Any,
        torch_module: Any,
        model_id: str,
        device: str,
        mask_threshold: float,
    ) -> None:
        self.model = model
        self.processor = processor
        self.torch = torch_module
        self.model_id = model_id
        self.device = device
        self.mask_threshold = mask_threshold

    @classmethod
    def from_config(cls, config: IndexConfig) -> "Sam2FaceRefiner":
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise Sam2RuntimeError(
                "SAM 2 refinement requires the 'torch' and 'transformers' packages in the selected environment."
            ) from exc

        device = cls._resolve_device(torch, config.sam2_device)
        processor = Sam2Processor.from_pretrained(
            config.sam2_model_id,
            local_files_only=config.sam2_local_files_only,
        )
        model = Sam2Model.from_pretrained(
            config.sam2_model_id,
            local_files_only=config.sam2_local_files_only,
        )
        model.to(device)
        model.eval()
        return cls(
            model=model,
            processor=processor,
            torch_module=torch,
            model_id=config.sam2_model_id,
            device=str(device),
            mask_threshold=config.sam2_mask_threshold,
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

    def refine_face(self, face_name: str, rgb: np.ndarray, prompts: list[Sam2Prompt]) -> list[RefinedMask]:
        image = Image.fromarray(rgb, mode="RGB")
        refined: list[RefinedMask] = []
        for prompt in prompts:
            input_kwargs: dict[str, Any] = {
                "images": image,
                "input_boxes": [[prompt.box_xyxy.tolist()]],
                "return_tensors": "pt",
            }
            point = prompt.point_xy
            if point is None and prompt.prior_mask is not None:
                point = _mask_centroid(prompt.prior_mask)
            if point is not None:
                input_kwargs["input_points"] = [[[point.tolist()]]]
                input_kwargs["input_labels"] = [[[1]]]

            inputs = self.processor(**input_kwargs).to(self.device)
            if prompt.prior_mask is not None:
                inputs["input_masks"] = self._prepare_prior_mask(prompt.prior_mask)
            with self.torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
                mask_threshold=self.mask_threshold,
                binarize=True,
            )[0]
            mask_array = np.asarray(masks.detach().cpu().numpy()).reshape(-1, rgb.shape[0], rgb.shape[1])
            scores = np.asarray(outputs.iou_scores.detach().cpu().numpy()).reshape(-1)
            best_index = int(scores.argmax()) if len(scores) else 0
            mask = mask_array[best_index].astype(np.uint8)
            score = float(scores[best_index]) if len(scores) else prompt.score
            bbox = _mask_bbox(mask)
            if bbox is None:
                bbox = prompt.box_xyxy
            refined.append(
                RefinedMask(
                    prompt_id=prompt.prompt_id,
                    face_name=face_name,
                    class_text=prompt.class_text,
                    source=prompt.source,
                    mask=np.where(mask > 0, 255, 0).astype(np.uint8),
                    box_xyxy=bbox.astype(np.float32),
                    score=score,
                    prompt_box_xyxy=prompt.box_xyxy.astype(np.float32),
                    prompt_score=float(prompt.score),
                    used_temporal_prior=prompt.used_temporal_prior,
                )
            )
        return refined

    def _prepare_prior_mask(self, prior_mask: np.ndarray) -> Any:
        image_size = int(getattr(self.model.config.prompt_encoder_config, "image_size", 1024))
        source = np.where(prior_mask > 0, 255, 0).astype(np.uint8)
        resized = Image.fromarray(source).resize(
            (image_size, image_size),
            resample=Image.Resampling.NEAREST,
        )
        mask = np.asarray(resized) > 0
        return self.torch.as_tensor(mask[None, :, :], dtype=self.torch.float32, device=self.device)


class Sam2StreamingVideoRefiner:
    def __init__(
        self,
        model: Any,
        processor: Any,
        torch_module: Any,
        model_id: str,
        device: str,
        mask_threshold: float,
    ) -> None:
        self.model = model
        self.processor = processor
        self.torch = torch_module
        self.model_id = model_id
        self.device = device
        self.mask_threshold = mask_threshold

    @classmethod
    def from_config(cls, config: IndexConfig) -> "Sam2StreamingVideoRefiner":
        try:
            import torch
            from transformers import Sam2VideoModel, Sam2VideoProcessor
        except ImportError as exc:
            raise Sam2RuntimeError(
                "SAM 2 video refinement requires the 'torch' and 'transformers>=5.5' packages."
            ) from exc

        device = Sam2FaceRefiner._resolve_device(torch, config.sam2_device)
        processor = Sam2VideoProcessor.from_pretrained(
            config.sam2_model_id,
            local_files_only=config.sam2_local_files_only,
        )
        model = Sam2VideoModel.from_pretrained(
            config.sam2_model_id,
            local_files_only=config.sam2_local_files_only,
        )
        model.to(device)
        model.eval()
        return cls(
            model=model,
            processor=processor,
            torch_module=torch,
            model_id=config.sam2_model_id,
            device=str(device),
            mask_threshold=config.sam2_mask_threshold,
        )

    def refine_face_sequence(
        self,
        face_name: str,
        face_rgbs_by_frame: list[Optional[np.ndarray]],
        prompts_by_frame: list[list[Sam2Prompt]],
        config: IndexConfig,
    ) -> tuple[list[list[RefinedMask]], list[int]]:
        valid_rgbs = [rgb for rgb in face_rgbs_by_frame if rgb is not None]
        if not valid_rgbs:
            return [[] for _ in face_rgbs_by_frame], [0 for _ in face_rgbs_by_frame]

        first_rgb = valid_rgbs[0]
        height, width = first_rgb.shape[:2]
        video = [Image.fromarray(rgb) for rgb in face_rgbs_by_frame if rgb is not None]
        valid_to_original = [index for index, rgb in enumerate(face_rgbs_by_frame) if rgb is not None]
        original_to_valid = {original: valid for valid, original in enumerate(valid_to_original)}
        session = self.processor.init_video_session(
            video=video,
            inference_device=self.device,
            inference_state_device=self.device,
            video_storage_device=self.device,
        )

        tracks: dict[int, Sam2VideoTrack] = {}
        next_obj_id = 1
        masks_by_frame: list[list[RefinedMask]] = [[] for _ in face_rgbs_by_frame]
        temporal_counts = [0 for _ in face_rgbs_by_frame]

        for original_frame_index in range(len(face_rgbs_by_frame)):
            valid_frame_index = original_to_valid.get(original_frame_index)
            if valid_frame_index is None:
                continue

            prompts = prompts_by_frame[original_frame_index]
            prompt_obj_ids: set[int] = set()
            reused_obj_ids: set[int] = set()
            if prompts:
                obj_ids: list[int] = []
                input_boxes: list[list[float]] = []
                input_points: list[list[list[float]]] = []
                input_labels: list[list[int]] = []
                matched_tracks: set[int] = set()
                for prompt in prompts:
                    track = self._match_track(
                        prompt=prompt,
                        tracks=tracks.values(),
                        frame_index=original_frame_index,
                        used_track_ids=matched_tracks,
                        config=config,
                    )
                    if track is None:
                        obj_id = next_obj_id
                        next_obj_id += 1
                        tracks[obj_id] = Sam2VideoTrack(
                            obj_id=obj_id,
                            face_name=prompt.face_name,
                            class_text=prompt.class_text,
                            source=prompt.source,
                            last_box_xyxy=prompt.box_xyxy,
                            last_frame_index=original_frame_index,
                            last_prompt_frame_index=original_frame_index,
                            prompt_id=prompt.prompt_id,
                            prompt_box_xyxy=prompt.box_xyxy,
                            prompt_score=float(prompt.score),
                        )
                    else:
                        obj_id = track.obj_id
                        matched_tracks.add(obj_id)
                        reused_obj_ids.add(obj_id)
                        track.last_prompt_frame_index = original_frame_index
                        track.prompt_id = prompt.prompt_id
                        track.prompt_box_xyxy = prompt.box_xyxy
                        track.prompt_score = float(prompt.score)

                    point = prompt.point_xy if prompt.point_xy is not None else _box_center(prompt.box_xyxy)
                    prompt_obj_ids.add(obj_id)
                    obj_ids.append(obj_id)
                    input_boxes.append(prompt.box_xyxy.tolist())
                    input_points.append([point.tolist()])
                    input_labels.append([1])

                self.processor.add_inputs_to_inference_session(
                    session,
                    frame_idx=valid_frame_index,
                    obj_ids=obj_ids,
                    input_boxes=[input_boxes],
                    input_points=[input_points],
                    input_labels=[input_labels],
                    clear_old_inputs=True,
                )

            if tracks:
                output = self.model(session, frame_idx=valid_frame_index)
                frame_masks = self._output_to_refined_masks(
                    output=output,
                    tracks=tracks,
                    face_name=face_name,
                    frame_index=original_frame_index,
                    prompt_obj_ids=prompt_obj_ids,
                    reused_obj_ids=reused_obj_ids,
                    width=width,
                    height=height,
                    config=config,
                )
                masks_by_frame[original_frame_index] = frame_masks
                temporal_counts[original_frame_index] = sum(1 for mask in frame_masks if mask.used_temporal_prior)

        return masks_by_frame, temporal_counts

    def _match_track(
        self,
        prompt: Sam2Prompt,
        tracks: Iterable[Sam2VideoTrack],
        frame_index: int,
        used_track_ids: set[int],
        config: IndexConfig,
    ) -> Optional[Sam2VideoTrack]:
        best_track = None
        best_iou = 0.0
        for track in tracks:
            if track.obj_id in used_track_ids:
                continue
            if track.face_name != prompt.face_name:
                continue
            if track.class_text != prompt.class_text or track.source != prompt.source:
                continue
            if frame_index - track.last_frame_index > config.sam2_temporal_max_gap:
                continue
            iou = _box_iou(prompt.box_xyxy, track.last_box_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is None:
            return None
        if prompt.class_text == "roof" and prompt.face_name == "down":
            return best_track
        if best_iou < config.sam2_temporal_iou_threshold:
            return None
        prior_area = max(1.0, _box_area(best_track.last_box_xyxy))
        ratio = _box_area(prompt.box_xyxy) / prior_area
        if config.sam2_temporal_area_ratio_min <= ratio <= config.sam2_temporal_area_ratio_max:
            return best_track
        return None

    def _output_to_refined_masks(
        self,
        output: Any,
        tracks: dict[int, Sam2VideoTrack],
        face_name: str,
        frame_index: int,
        prompt_obj_ids: set[int],
        reused_obj_ids: set[int],
        width: int,
        height: int,
        config: IndexConfig,
    ) -> list[RefinedMask]:
        object_ids = [int(obj_id) for obj_id in output.object_ids]
        if not object_ids:
            return []
        masks = self.processor.post_process_masks(
            output.pred_masks.detach().cpu(),
            [[height, width] for _ in object_ids],
            mask_threshold=self.mask_threshold,
            binarize=True,
        )
        mask_array = np.asarray(masks.detach().cpu().numpy()).reshape(len(object_ids), -1, height, width)[:, 0]
        logits = getattr(output, "object_score_logits", None)
        if logits is not None:
            scores = self.torch.sigmoid(logits.detach().cpu()).numpy().reshape(-1)
        else:
            scores = np.ones((len(object_ids),), dtype=np.float32)

        refined: list[RefinedMask] = []
        for index, obj_id in enumerate(object_ids):
            track = tracks.get(obj_id)
            if track is None:
                continue
            has_prompt = obj_id in prompt_obj_ids
            recently_tracked = frame_index - track.last_prompt_frame_index <= config.sam2_temporal_max_gap
            if not has_prompt and (not config.sam2_temporal_propagation or not recently_tracked):
                continue

            mask = np.where(mask_array[index] > 0, 255, 0).astype(np.uint8)
            bbox = _mask_bbox(mask)
            if bbox is None:
                continue
            track.last_box_xyxy = bbox.astype(np.float32)
            track.last_frame_index = frame_index

            used_temporal_prior = (not has_prompt) or obj_id in reused_obj_ids
            source = track.source if has_prompt else "temporal_prior"
            prompt_id = track.prompt_id if has_prompt else f"{face_name}:temporal:{track.class_text}:{obj_id}:{frame_index}"
            refined.append(
                RefinedMask(
                    prompt_id=prompt_id,
                    face_name=face_name,
                    class_text=track.class_text,
                    source=source,
                    mask=mask,
                    box_xyxy=bbox.astype(np.float32),
                    score=float(scores[index]) if index < len(scores) else track.prompt_score,
                    prompt_box_xyxy=track.prompt_box_xyxy.astype(np.float32),
                    prompt_score=float(track.prompt_score),
                    used_temporal_prior=used_temporal_prior,
                )
            )
        return refined


def _mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def _empty_face_payload() -> dict[str, np.ndarray]:
    return {
        "masks": np.zeros((0, 0, 0), dtype=np.uint8),
        "boxes_xyxy": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "class_text": np.asarray([], dtype="<U128"),
        "source": np.asarray([], dtype="<U64"),
        "prompt_ids": np.asarray([], dtype="<U256"),
        "prompt_boxes_xyxy": np.zeros((0, 4), dtype=np.float32),
        "prompt_scores": np.zeros((0,), dtype=np.float32),
        "used_temporal_prior": np.zeros((0,), dtype=np.uint8),
    }


def _write_face_masks(output_dir: Path, face_name: str, masks: list[RefinedMask], shape: tuple[int, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if masks:
        payload = {
            "masks": np.stack([mask.mask for mask in masks], axis=0).astype(np.uint8),
            "boxes_xyxy": np.asarray([mask.box_xyxy for mask in masks], dtype=np.float32),
            "scores": np.asarray([mask.score for mask in masks], dtype=np.float32),
            "class_text": np.asarray([mask.class_text for mask in masks], dtype="<U128"),
            "source": np.asarray([mask.source for mask in masks], dtype="<U64"),
            "prompt_ids": np.asarray([mask.prompt_id for mask in masks], dtype="<U256"),
            "prompt_boxes_xyxy": np.asarray([mask.prompt_box_xyxy for mask in masks], dtype=np.float32),
            "prompt_scores": np.asarray([mask.prompt_score for mask in masks], dtype=np.float32),
            "used_temporal_prior": np.asarray([mask.used_temporal_prior for mask in masks], dtype=np.uint8),
        }
    else:
        payload = _empty_face_payload()
        payload["masks"] = np.zeros((0, shape[0], shape[1]), dtype=np.uint8)
    np.savez_compressed(output_dir / f"{face_name}.npz", **payload)

    sidecar = {
        "face_name": face_name,
        "mask_count": len(masks),
        "masks": [
            {
                "prompt_id": mask.prompt_id,
                "class_text": mask.class_text,
                "source": mask.source,
                "box_xyxy": [float(value) for value in mask.box_xyxy.tolist()],
                "score": float(mask.score),
                "prompt_box_xyxy": [float(value) for value in mask.prompt_box_xyxy.tolist()],
                "prompt_score": float(mask.prompt_score),
                "used_temporal_prior": bool(mask.used_temporal_prior),
            }
            for mask in masks
        ],
    }
    (output_dir / f"{face_name}.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_sam2_metadata(output_dir: Path, refiner: Any, config: IndexConfig) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "model_id": getattr(refiner, "model_id", config.sam2_model_id),
        "face_order": list(FACE_ORDER),
        "mask_threshold": config.sam2_mask_threshold,
        "min_mask_area_px": config.sam2_min_mask_area_px,
        "refine_grounding": config.sam2_refine_grounding,
        "refine_semantic": config.sam2_refine_semantic,
        "refine_roof": config.sam2_refine_roof,
        "semantic_prompt_classes": list(config.sam2_semantic_prompt_classes),
        "roof_box_fraction": config.sam2_roof_box_fraction,
        "roof_prior_margin_fraction": config.sam2_roof_prior_margin_fraction,
        "roof_temporal_window": config.sam2_roof_temporal_window,
        "roof_temporal_disagreement_iou_threshold": config.sam2_roof_temporal_disagreement_iou_threshold,
        "temporal_propagation": config.sam2_temporal_propagation,
        "temporal_iou_threshold": config.sam2_temporal_iou_threshold,
        "temporal_area_ratio_min": config.sam2_temporal_area_ratio_min,
        "temporal_area_ratio_max": config.sam2_temporal_area_ratio_max,
        "temporal_max_gap": config.sam2_temporal_max_gap,
        "face_cache_format": "npz",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def _extract_roof_candidate(
    masks: list[RefinedMask],
    shape: tuple[int, int],
    config: IndexConfig,
) -> Optional[np.ndarray]:
    roof_masks = [
        mask.mask
        for mask in masks
        if mask.face_name == "down" and _normalize_class_text(mask.class_text) == "roof"
    ]
    if not roof_masks:
        return None

    height, width = shape
    combined = np.zeros((height, width), dtype=np.uint8)
    for mask in roof_masks:
        if mask.shape != combined.shape:
            continue
        combined = np.maximum(combined, np.where(mask > 0, 255, 0).astype(np.uint8))
    if int((combined > 0).sum()) < config.sam2_min_mask_area_px:
        return None
    return _regularize_roof_down_mask(combined, config)


def _temporal_median_roof_mask(
    candidates: list[Optional[np.ndarray]],
    frame_index: int,
    shape: tuple[int, int],
    config: IndexConfig,
) -> Optional[np.ndarray]:
    window = max(0, int(config.sam2_roof_temporal_window))
    if window == 0:
        return None

    neighbor_masks: list[np.ndarray] = []
    start = max(0, frame_index - window)
    stop = min(len(candidates), frame_index + window + 1)
    for neighbor_index in range(start, stop):
        if neighbor_index == frame_index:
            continue
        candidate = candidates[neighbor_index]
        if candidate is not None and candidate.shape == shape:
            neighbor_masks.append(candidate)
    if not neighbor_masks:
        return None

    votes = np.stack([mask > 0 for mask in neighbor_masks], axis=0).sum(axis=0)
    threshold = max(1, (len(neighbor_masks) + 1) // 2)
    return np.where(votes >= threshold, 255, 0).astype(np.uint8)


def _select_roof_mask_result(
    candidate: Optional[np.ndarray],
    temporal_mask: Optional[np.ndarray],
    shape: tuple[int, int],
    config: IndexConfig,
) -> RoofMaskResult:
    height, width = shape
    if candidate is None:
        if temporal_mask is not None:
            return RoofMaskResult(
                down_mask=_regularize_roof_down_mask(temporal_mask, config),
                source="temporal_median_fallback",
                temporal_disagreement=False,
            )
        return RoofMaskResult(
            down_mask=_roof_prior_mask(width, height, config, expanded=True),
            source="coarse_prior_fallback",
            temporal_disagreement=False,
        )

    candidate = _regularize_roof_down_mask(candidate, config)
    if temporal_mask is None:
        return RoofMaskResult(
            down_mask=candidate,
            source="sam2_current",
            temporal_disagreement=False,
        )

    temporal_mask = _regularize_roof_down_mask(temporal_mask, config)
    iou = _mask_iou(candidate, temporal_mask)
    threshold = float(config.sam2_roof_temporal_disagreement_iou_threshold)
    if iou < threshold:
        return RoofMaskResult(
            down_mask=candidate,
            source="sam2_current_temporal_disagreement",
            temporal_disagreement=True,
            error=f"roof temporal median IoU {iou:.3f} below threshold {threshold:.3f}; kept current evidence",
        )

    return RoofMaskResult(
        down_mask=_regularize_roof_down_mask(np.maximum(candidate, temporal_mask), config),
        source="sam2_current_temporal_regularized",
        temporal_disagreement=False,
    )


def _write_roof_mask_artifacts(
    frame: FrameRecord,
    result: RoofMaskResult,
    config: IndexConfig,
) -> int:
    if frame.erp_width is None or frame.erp_height is None:
        raise RuntimeError("Cannot write roof mask without ERP dimensions")
    if frame.cubemap_face_size is None or frame.cubemap_overlap_px is None:
        raise RuntimeError("Cannot write roof mask without cubemap geometry")

    erp_mask = cubemap_face_mask_to_erp(
        result.down_mask,
        face_name="down",
        erp_width=frame.erp_width,
        erp_height=frame.erp_height,
        face_size=frame.cubemap_face_size,
        overlap_px=frame.cubemap_overlap_px,
    )
    area_px = int((erp_mask > 0).sum())
    if config.dry_run:
        return area_px

    frame.roof_mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(erp_mask).save(frame.roof_mask_path)
    sidecar = {
        "frame_name": frame.frame_name,
        "source": result.source,
        "temporal_disagreement": result.temporal_disagreement,
        "error": result.error,
        "down_face_shape": [int(value) for value in result.down_mask.shape],
        "erp_shape": [int(frame.erp_height), int(frame.erp_width)],
        "area_px": area_px,
    }
    frame.roof_mask_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return area_px


def _finalize_roof_masks_for_sequence(
    rows: list[FrameRecord],
    masks_by_frame: list[list[RefinedMask]],
    face_shapes_by_frame: list[dict[str, tuple[int, int]]],
    config: IndexConfig,
) -> list[FrameRecord]:
    if not config.sam2_refine_roof:
        return [
            replace(row, roof_mask_status="skipped", roof_mask_source="", roof_mask_error="")
            for row in rows
        ]

    candidates: list[Optional[np.ndarray]] = []
    for frame_index, row in enumerate(rows):
        shape = face_shapes_by_frame[frame_index].get("down", None)
        if shape is None:
            candidates.append(None)
            continue
        candidates.append(_extract_roof_candidate(masks_by_frame[frame_index], shape, config))

    finalized_rows: list[FrameRecord] = []
    for frame_index, row in enumerate(rows):
        shape = face_shapes_by_frame[frame_index].get("down", None)
        if shape is None:
            finalized_rows.append(
                replace(
                    row,
                    roof_mask_status="failed",
                    roof_mask_source="",
                    roof_mask_area_px=None,
                    roof_mask_temporal_disagreement=False,
                    roof_mask_error="Missing down-face cubemap projection",
                )
            )
            continue

        temporal_mask = _temporal_median_roof_mask(candidates, frame_index, shape, config)
        result = _select_roof_mask_result(candidates[frame_index], temporal_mask, shape, config)
        try:
            area_px = _write_roof_mask_artifacts(row, result, config)
        except Exception as exc:
            finalized_rows.append(
                replace(
                    row,
                    roof_mask_status="failed",
                    roof_mask_source=result.source,
                    roof_mask_area_px=None,
                    roof_mask_temporal_disagreement=result.temporal_disagreement,
                    roof_mask_error=str(exc),
                )
            )
            continue

        status = "fallback" if result.source.endswith("_fallback") else "generated"
        finalized_rows.append(
            replace(
                row,
                roof_mask_status=status,
                roof_mask_source=result.source,
                roof_mask_area_px=area_px,
                roof_mask_temporal_disagreement=result.temporal_disagreement,
                roof_mask_error=result.error,
            )
        )
    return finalized_rows


def _refine_frame_masks_with_outputs(
    frame: FrameRecord,
    config: IndexConfig,
    refiner: Any,
    memory: TemporalMaskMemory,
    frame_index: int,
) -> FrameRefinementResult:
    try:
        projected_frame, _, face_rgbs = load_or_create_cubemap_face_rgbs(frame, config)
    except Exception as exc:
        return FrameRefinementResult(
            frame=replace(
                frame,
                sam2_model_id=getattr(refiner, "model_id", config.sam2_model_id),
                sam2_refine_status="failed",
                sam2_refine_error=str(exc),
            ),
            masks=[],
            face_shapes={},
        )

    output_dir = sam2_output_dir_for_frame(projected_frame)
    all_masks: list[RefinedMask] = []
    face_shapes = {face_name: face_rgbs[face_name].shape[:2] for face_name in FACE_ORDER}
    temporal_prior_count = 0
    roof_warning = ""
    try:
        for face_name in FACE_ORDER:
            rgb = face_rgbs[face_name]
            prompts = collect_face_prompts(projected_frame, face_name, rgb.shape[1], rgb.shape[0], config)
            prompts, face_prior_count = memory.attach_priors(prompts, frame_index, config, face_name=face_name)
            temporal_prior_count += face_prior_count

            if face_name == "down":
                roof_prompts = [
                    prompt for prompt in prompts
                    if _normalize_class_text(prompt.class_text) == "roof"
                ]
                other_prompts = [
                    prompt for prompt in prompts
                    if _normalize_class_text(prompt.class_text) != "roof"
                ]
                face_masks = refiner.refine_face(face_name, rgb, other_prompts) if other_prompts else []
                if roof_prompts:
                    try:
                        face_masks.extend(refiner.refine_face(face_name, rgb, roof_prompts))
                    except Exception as exc:
                        roof_warning = f"roof SAM 2 refinement failed; roof mask will use fallback: {exc}"
            else:
                face_masks = refiner.refine_face(face_name, rgb, prompts) if prompts else []

            face_masks = [
                mask for mask in face_masks if int((mask.mask > 0).sum()) >= config.sam2_min_mask_area_px
            ]
            all_masks.extend(face_masks)
            if not config.dry_run:
                _write_face_masks(output_dir, face_name, face_masks, rgb.shape[:2])
    except Exception as exc:
        return FrameRefinementResult(
            frame=replace(
                projected_frame,
                sam2_model_id=getattr(refiner, "model_id", config.sam2_model_id),
                sam2_output_dir=output_dir,
                sam2_refine_status="failed",
                sam2_refine_error=str(exc),
            ),
            masks=all_masks,
            face_shapes=face_shapes,
        )

    memory.update(all_masks, frame_index)
    if not config.dry_run:
        _write_sam2_metadata(output_dir, refiner, config)

    return FrameRefinementResult(
        frame=replace(
            projected_frame,
            sam2_model_id=getattr(refiner, "model_id", config.sam2_model_id),
            sam2_output_dir=output_dir,
            sam2_mask_count=len(all_masks),
            sam2_temporal_prior_count=temporal_prior_count,
            sam2_refine_status="refined",
            sam2_refine_error=roof_warning,
        ),
        masks=all_masks,
        face_shapes=face_shapes,
    )


def refine_frame_masks(
    frame: FrameRecord,
    config: IndexConfig,
    refiner: Any,
    memory: TemporalMaskMemory,
    frame_index: int,
) -> FrameRecord:
    return _refine_frame_masks_with_outputs(frame, config, refiner, memory, frame_index).frame


def refine_sequence_masks(sequence_manifest: SequenceManifest, config: IndexConfig, refiner: Any) -> SequenceManifest:
    if sequence_manifest.status != "ready":
        return sequence_manifest

    if hasattr(refiner, "refine_face_sequence"):
        return refine_sequence_masks_streaming(sequence_manifest, config, refiner)

    memory = TemporalMaskMemory.create()
    frame_results = [
        _refine_frame_masks_with_outputs(frame, config, refiner, memory, frame_index)
        for frame_index, frame in enumerate(sequence_manifest.rows)
    ]
    refined_rows = _finalize_roof_masks_for_sequence(
        rows=[result.frame for result in frame_results],
        masks_by_frame=[result.masks for result in frame_results],
        face_shapes_by_frame=[result.face_shapes for result in frame_results],
        config=config,
    )
    failed_rows = [row for row in refined_rows if row.sam2_refine_status == "failed"]
    status = sequence_manifest.status
    failure_reason = sequence_manifest.failure_reason
    if failed_rows:
        status = "failed_sam2_refinement"
        failure_reason = f"{len(failed_rows)} frame(s) failed SAM 2 refinement"

    refined_manifest = SequenceManifest(
        sequence_id=sequence_manifest.sequence_id,
        sequence_dir=sequence_manifest.sequence_dir,
        csv_path=sequence_manifest.csv_path,
        output_dir=sequence_manifest.output_dir,
        status=status,
        failure_reason=failure_reason,
        total_csv_rows=sequence_manifest.total_csv_rows,
        valid_frames=sequence_manifest.valid_frames,
        skipped_frames=sequence_manifest.skipped_frames,
        rows=refined_rows,
    )

    if not config.dry_run:
        write_sequence_manifest(refined_manifest)

    return refined_manifest


def refine_sequence_masks_streaming(
    sequence_manifest: SequenceManifest,
    config: IndexConfig,
    refiner: Any,
) -> SequenceManifest:
    projected_rows: list[FrameRecord] = []
    frame_face_rgbs: list[Optional[dict[str, np.ndarray]]] = []
    for frame in sequence_manifest.rows:
        try:
            projected_frame, _, face_rgbs = load_or_create_cubemap_face_rgbs(frame, config)
        except Exception as exc:
            projected_rows.append(
                replace(
                    frame,
                    sam2_model_id=getattr(refiner, "model_id", config.sam2_model_id),
                    sam2_refine_status="failed",
                    sam2_refine_error=str(exc),
                )
            )
            frame_face_rgbs.append(None)
            continue
        projected_rows.append(projected_frame)
        frame_face_rgbs.append(face_rgbs)

    masks_by_frame: list[list[RefinedMask]] = [[] for _ in sequence_manifest.rows]
    face_shapes_by_frame: list[dict[str, tuple[int, int]]] = [
        {} if face_rgbs is None else {face_name: rgb.shape[:2] for face_name, rgb in face_rgbs.items()}
        for face_rgbs in frame_face_rgbs
    ]
    streaming_error = ""
    roof_streaming_warning = ""
    for face_name in FACE_ORDER:
        face_rgbs_by_frame = [
            None if face_rgbs is None else face_rgbs[face_name]
            for face_rgbs in frame_face_rgbs
        ]
        prompts_by_frame: list[list[Sam2Prompt]] = []
        for frame_index, rgb in enumerate(face_rgbs_by_frame):
            if rgb is None:
                prompts_by_frame.append([])
                continue
            prompts_by_frame.append(
                collect_face_prompts(
                    projected_rows[frame_index],
                    face_name,
                    rgb.shape[1],
                    rgb.shape[0],
                    config,
                )
            )

        try:
            face_masks_by_frame, _ = refiner.refine_face_sequence(
                face_name,
                face_rgbs_by_frame,
                prompts_by_frame,
                config,
            )
        except Exception as exc:
            has_non_roof_prompt = any(
                _normalize_class_text(prompt.class_text) != "roof"
                for prompts in prompts_by_frame
                for prompt in prompts
            )
            if face_name == "down" and not has_non_roof_prompt and config.sam2_refine_roof:
                roof_streaming_warning = f"roof SAM 2 video refinement failed; roof mask will use fallback: {exc}"
                face_masks_by_frame = [[] for _ in face_rgbs_by_frame]
            else:
                streaming_error = str(exc)
                break

        for frame_index, face_masks in enumerate(face_masks_by_frame):
            masks_by_frame[frame_index].extend(face_masks)
            rgb = face_rgbs_by_frame[frame_index]
            if rgb is not None and not config.dry_run:
                _write_face_masks(
                    sam2_output_dir_for_frame(projected_rows[frame_index]),
                    face_name,
                    [
                        mask for mask in face_masks
                        if int((mask.mask > 0).sum()) >= config.sam2_min_mask_area_px
                    ],
                    rgb.shape[:2],
                )

    refined_rows: list[FrameRecord] = []
    for frame_index, frame in enumerate(projected_rows):
        if frame.sam2_refine_status == "failed":
            refined_rows.append(frame)
            continue
        output_dir = sam2_output_dir_for_frame(frame)
        frame_masks = [
            mask for mask in masks_by_frame[frame_index]
            if int((mask.mask > 0).sum()) >= config.sam2_min_mask_area_px
        ]
        frame_temporal_count = sum(1 for mask in frame_masks if mask.used_temporal_prior)
        if streaming_error:
            refined_rows.append(
                replace(
                    frame,
                    sam2_model_id=getattr(refiner, "model_id", config.sam2_model_id),
                    sam2_output_dir=output_dir,
                    sam2_refine_status="failed",
                    sam2_refine_error=streaming_error,
                )
            )
            continue
        if not config.dry_run:
            _write_sam2_metadata(output_dir, refiner, config)
        refined_rows.append(
            replace(
                frame,
                sam2_model_id=getattr(refiner, "model_id", config.sam2_model_id),
                sam2_output_dir=output_dir,
                sam2_mask_count=len(frame_masks),
                sam2_temporal_prior_count=frame_temporal_count,
                sam2_refine_status="refined",
                sam2_refine_error=roof_streaming_warning,
            )
        )

    refined_rows = _finalize_roof_masks_for_sequence(
        rows=refined_rows,
        masks_by_frame=masks_by_frame,
        face_shapes_by_frame=face_shapes_by_frame,
        config=config,
    )

    failed_rows = [row for row in refined_rows if row.sam2_refine_status == "failed"]
    status = sequence_manifest.status
    failure_reason = sequence_manifest.failure_reason
    if failed_rows:
        status = "failed_sam2_refinement"
        failure_reason = f"{len(failed_rows)} frame(s) failed SAM 2 refinement"

    refined_manifest = SequenceManifest(
        sequence_id=sequence_manifest.sequence_id,
        sequence_dir=sequence_manifest.sequence_dir,
        csv_path=sequence_manifest.csv_path,
        output_dir=sequence_manifest.output_dir,
        status=status,
        failure_reason=failure_reason,
        total_csv_rows=sequence_manifest.total_csv_rows,
        valid_frames=sequence_manifest.valid_frames,
        skipped_frames=sequence_manifest.skipped_frames,
        rows=refined_rows,
    )

    if not config.dry_run:
        write_sequence_manifest(refined_manifest)

    return refined_manifest


def run_sam2_refinement(
    config: IndexConfig,
    sequence_ids: Optional[Iterable[str]] = None,
    refiner: Any = None,
) -> list[SequenceManifest]:
    sam2_refiner = refiner if refiner is not None else Sam2StreamingVideoRefiner.from_config(config)
    indexed_manifests = run_indexing(config, sequence_ids=sequence_ids)
    return [refine_sequence_masks(manifest, config, sam2_refiner) for manifest in indexed_manifests]
