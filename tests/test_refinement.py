from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.cubemap import FACE_ORDER
from urpan_inpaint.refinement import RefinedMask, _regularize_sky_face_mask, run_sam2_refinement


class FakeSam2Refiner:
    model_id = "fake/sam2"

    def refine_face(self, face_name: str, rgb: np.ndarray, prompts):
        height, width = rgb.shape[:2]
        refined = []
        for prompt in prompts:
            x0, y0, x1, y1 = prompt.box_xyxy
            x0 = max(0, min(width - 1, int(np.floor(x0))))
            y0 = max(0, min(height - 1, int(np.floor(y0))))
            x1 = max(x0 + 1, min(width, int(np.ceil(x1))))
            y1 = max(y0 + 1, min(height, int(np.ceil(y1))))
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255
            box = np.asarray([x0, y0, x1, y1], dtype=np.float32)
            refined.append(
                RefinedMask(
                    prompt_id=prompt.prompt_id,
                    face_name=face_name,
                    class_text=prompt.class_text,
                    source=prompt.source,
                    mask=mask,
                    box_xyxy=box,
                    score=max(float(prompt.score), 0.5),
                    prompt_box_xyxy=prompt.box_xyxy.astype(np.float32),
                    prompt_score=float(prompt.score),
                    used_temporal_prior=prompt.used_temporal_prior,
                )
            )
        return refined


class FailingSam2Refiner:
    model_id = "fake/failing-sam2"

    def refine_face(self, face_name: str, rgb: np.ndarray, prompts):
        if prompts:
            raise RuntimeError(f"intentional SAM 2 failure on {face_name}")
        return []


class FakeStreamingSam2Refiner:
    model_id = "fake/streaming-sam2"

    def refine_face_sequence(self, face_name: str, face_rgbs_by_frame, prompts_by_frame, config):
        masks_by_frame = []
        temporal_counts = []
        face_refiner = FakeSam2Refiner()
        for frame_index, rgb in enumerate(face_rgbs_by_frame):
            if rgb is None:
                masks_by_frame.append([])
                temporal_counts.append(0)
                continue
            masks = face_refiner.refine_face(face_name, rgb, prompts_by_frame[frame_index])
            if frame_index > 0:
                masks = [
                    RefinedMask(
                        prompt_id=mask.prompt_id,
                        face_name=mask.face_name,
                        class_text=mask.class_text,
                        source=mask.source,
                        mask=mask.mask,
                        box_xyxy=mask.box_xyxy,
                        score=mask.score,
                        prompt_box_xyxy=mask.prompt_box_xyxy,
                        prompt_score=mask.prompt_score,
                        used_temporal_prior=mask.class_text == "roof",
                    )
                    for mask in masks
                ]
            masks_by_frame.append(masks)
            temporal_counts.append(sum(1 for mask in masks if mask.used_temporal_prior))
        return masks_by_frame, temporal_counts


class DisagreeingRoofRefiner(FakeSam2Refiner):
    model_id = "fake/disagreeing-roof-sam2"

    def refine_face(self, face_name: str, rgb: np.ndarray, prompts):
        height, width = rgb.shape[:2]
        refined = []
        for prompt in prompts:
            if face_name != "down" or prompt.class_text != "roof":
                refined.extend(super().refine_face(face_name, rgb, [prompt]))
                continue

            mask = np.zeros((height, width), dtype=np.uint8)
            if prompt.used_temporal_prior:
                mask[:, :] = 255
            else:
                cx = width // 2
                cy = height // 2
                mask[max(0, cy - 1) : min(height, cy + 1), max(0, cx - 1) : min(width, cx + 1)] = 255
            bbox = np.asarray([0, 0, width, height], dtype=np.float32)
            refined.append(
                RefinedMask(
                    prompt_id=prompt.prompt_id,
                    face_name=face_name,
                    class_text=prompt.class_text,
                    source=prompt.source,
                    mask=mask,
                    box_xyxy=bbox,
                    score=0.9,
                    prompt_box_xyxy=prompt.box_xyxy.astype(np.float32),
                    prompt_score=float(prompt.score),
                    used_temporal_prior=prompt.used_temporal_prior,
                )
            )
        return refined


class RefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_root = self.root / "dataset"
        self.output_root = self.root / "output"
        self.dataset_root.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_sequence(self, sequence_id: str, frame_count: int) -> None:
        sequence_dir = self.dataset_root / sequence_id
        fixed_dir = sequence_dir / "fixed"
        fixed_dir.mkdir(parents=True)
        rows = []
        for index in range(frame_count):
            frame_number = index + 1
            image_path = fixed_dir / f"{sequence_id}-frame-{frame_number:06d}.jpg"
            pixels = np.zeros((16, 32, 3), dtype=np.uint8)
            pixels[:, :, 0] = 30 + index
            pixels[:, :, 1] = 80
            pixels[:, :, 2] = 130
            Image.fromarray(pixels, mode="RGB").save(image_path, format="JPEG", quality=100, subsampling=0)
            rows.append(
                [
                    str(index),
                    str(image_path),
                    f"{index * 0.5:.1f}",
                    "39.0",
                    "-84.0",
                    "185.0",
                    "1.0",
                    "1.0",
                    "2020-09-24 21:05:33.914000+00:00",
                ]
            )

        with (sequence_dir / "gps-fixed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "path", "ts", "lat", "lon", "alt", "speed", "speed3d", "date"])
            writer.writerows(rows)

    def write_grounding_face(self, sequence_id: str, frame_number: int, face_name: str = "front") -> None:
        stem = f"{sequence_id}-frame-{frame_number:06d}"
        output_dir = self.output_root / sequence_id / "cubemap" / stem / "grounding_dino"
        output_dir.mkdir(parents=True)
        np.savez_compressed(
            output_dir / f"{face_name}.npz",
            boxes_xyxy=np.asarray([[1.0, 1.0, 5.0, 5.0]], dtype=np.float32),
            scores=np.asarray([0.91], dtype=np.float32),
            text_labels=np.asarray(["person"], dtype="<U128"),
            text_labels_normalized=np.asarray(["person"], dtype="<U128"),
        )

    def write_semantic_face(self, sequence_id: str, frame_number: int, face_name: str = "up") -> None:
        stem = f"{sequence_id}-frame-{frame_number:06d}"
        output_dir = self.output_root / sequence_id / "cubemap" / stem / "semantic_mask2former"
        output_dir.mkdir(parents=True)
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[:6, :] = 255
        np.savez_compressed(
            output_dir / f"{face_name}.npz",
            target_masks=np.expand_dims(mask, axis=0),
            target_class_ids=np.asarray([1], dtype=np.int32),
            target_class_names=np.asarray(["sky"], dtype="<U32"),
        )

    def test_run_sam2_refinement_writes_prompt_outputs_and_manifest(self) -> None:
        sequence_id = "GS999501"
        self.write_sequence(sequence_id, frame_count=1)
        self.write_grounding_face(sequence_id, frame_number=1)
        self.write_semantic_face(sequence_id, frame_number=1)

        manifests = run_sam2_refinement(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                cache_cubemap_faces=True,
            ),
            refiner=FakeSam2Refiner(),
        )

        self.assertEqual(len(manifests), 1)
        manifest = manifests[0]
        frame = manifest.rows[0]
        self.assertEqual(manifest.status, "ready")
        self.assertEqual(frame.sam2_refine_status, "refined")
        self.assertEqual(frame.sam2_model_id, FakeSam2Refiner.model_id)
        self.assertEqual(frame.sam2_mask_count, 3)
        self.assertEqual(frame.sam2_temporal_prior_count, 0)
        self.assertEqual(frame.roof_mask_status, "generated")
        self.assertEqual(frame.roof_mask_source, "sam2_current")
        self.assertFalse(frame.roof_mask_temporal_disagreement)
        self.assertTrue(frame.roof_mask_area_px and frame.roof_mask_area_px > 0)
        self.assertTrue(frame.roof_mask_path.is_file())
        self.assertTrue(frame.roof_mask_path.with_suffix(".json").is_file())
        self.assertEqual(frame.sky_mask_status, "generated")
        self.assertIn("semantic_sam2_boundary", frame.sky_mask_source)
        self.assertTrue(frame.sky_mask_area_px and frame.sky_mask_area_px > 0)
        self.assertTrue(frame.sky_mask_path.is_file())
        self.assertTrue(frame.sky_mask_path.with_suffix(".json").is_file())
        self.assertIsNotNone(frame.sam2_output_dir)

        output_dir = frame.sam2_output_dir
        self.assertTrue(output_dir and (output_dir / "metadata.json").is_file())
        with (output_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata["model_id"], FakeSam2Refiner.model_id)
        self.assertTrue(metadata["refine_roof"])

        for face_name in FACE_ORDER:
            face_npz = output_dir / f"{face_name}.npz"
            face_json = output_dir / f"{face_name}.json"
            self.assertTrue(face_npz.is_file())
            self.assertTrue(face_json.is_file())

        with np.load(output_dir / "front.npz") as payload:
            self.assertEqual(payload["masks"].shape, (1, 12, 12))
            self.assertEqual(payload["class_text"].tolist(), ["person"])
            self.assertEqual(payload["source"].tolist(), ["grounding_box"])
        with np.load(output_dir / "up.npz") as payload:
            self.assertEqual(payload["class_text"].tolist(), ["sky"])
            self.assertEqual(payload["source"].tolist(), ["semantic_region"])
        with np.load(output_dir / "down.npz") as payload:
            self.assertEqual(payload["class_text"].tolist(), ["roof"])
            self.assertEqual(payload["source"].tolist(), ["roof_down_prior"])

        frames_csv = self.output_root / sequence_id / "manifests" / "frames.csv"
        with frames_csv.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["sam2_model_id"], FakeSam2Refiner.model_id)
        self.assertEqual(rows[0]["sam2_refine_status"], "refined")
        self.assertEqual(rows[0]["sam2_mask_count"], "3")
        self.assertEqual(rows[0]["roof_mask_status"], "generated")
        self.assertEqual(rows[0]["roof_mask_source"], "sam2_current")
        self.assertEqual(rows[0]["sky_mask_status"], "generated")
        self.assertTrue(int(rows[0]["sky_mask_area_px"]) > 0)

    def test_sky_regularization_removes_lower_islands_and_semantic_blockers(self) -> None:
        candidate = np.zeros((12, 10), dtype=np.uint8)
        candidate[:4, :] = 255
        candidate[8:10, 4:8] = 255
        blocker = np.zeros_like(candidate)
        blocker[:4, 5] = 255

        result = _regularize_sky_face_mask(
            "front",
            candidate,
            blocker,
            IndexConfig(sky_mask_obstacle_dilation_px=0),
        )

        self.assertGreater(int((result[:4, :5] > 0).sum()), 0)
        self.assertEqual(int((result[8:10, :] > 0).sum()), 0)
        self.assertEqual(int((result[:4, 5] > 0).sum()), 0)

    def test_temporal_propagation_uses_stable_roof_prior(self) -> None:
        sequence_id = "GS999502"
        self.write_sequence(sequence_id, frame_count=2)

        manifests = run_sam2_refinement(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                cache_cubemap_faces=True,
            ),
            refiner=FakeSam2Refiner(),
        )

        first, second = manifests[0].rows
        self.assertEqual(first.sam2_refine_status, "refined")
        self.assertEqual(second.sam2_refine_status, "refined")
        self.assertEqual(first.sam2_mask_count, 1)
        self.assertEqual(second.sam2_mask_count, 1)
        self.assertEqual(first.sam2_temporal_prior_count, 0)
        self.assertEqual(second.sam2_temporal_prior_count, 1)

        self.assertIsNotNone(second.sam2_output_dir)
        with np.load(second.sam2_output_dir / "down.npz") as payload:
            self.assertEqual(payload["used_temporal_prior"].tolist(), [1])

    def test_streaming_refiner_path_writes_outputs_and_manifest(self) -> None:
        sequence_id = "GS999504"
        self.write_sequence(sequence_id, frame_count=2)

        manifests = run_sam2_refinement(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                cache_cubemap_faces=True,
            ),
            refiner=FakeStreamingSam2Refiner(),
        )

        self.assertEqual(manifests[0].status, "ready")
        first, second = manifests[0].rows
        self.assertEqual(first.sam2_model_id, FakeStreamingSam2Refiner.model_id)
        self.assertEqual(second.sam2_model_id, FakeStreamingSam2Refiner.model_id)
        self.assertEqual(first.sam2_mask_count, 1)
        self.assertEqual(second.sam2_mask_count, 1)
        self.assertEqual(second.sam2_temporal_prior_count, 1)
        self.assertTrue(second.sam2_output_dir and (second.sam2_output_dir / "metadata.json").is_file())

    def test_roof_temporal_disagreement_prefers_current_and_flags_frame(self) -> None:
        sequence_id = "GS999506"
        self.write_sequence(sequence_id, frame_count=2)

        manifests = run_sam2_refinement(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                cache_cubemap_faces=True,
                sam2_min_mask_area_px=1,
                sam2_roof_temporal_disagreement_iou_threshold=0.95,
            ),
            refiner=DisagreeingRoofRefiner(),
        )

        second = manifests[0].rows[1]
        self.assertEqual(second.roof_mask_status, "generated")
        self.assertEqual(second.roof_mask_source, "sam2_current_temporal_disagreement")
        self.assertTrue(second.roof_mask_temporal_disagreement)
        self.assertIn("kept current evidence", second.roof_mask_error)
        self.assertTrue(second.roof_mask_path.with_suffix(".json").is_file())

    def test_run_sam2_refinement_marks_frame_failures(self) -> None:
        sequence_id = "GS999503"
        self.write_sequence(sequence_id, frame_count=1)
        self.write_grounding_face(sequence_id, frame_number=1)

        manifests = run_sam2_refinement(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
            ),
            refiner=FailingSam2Refiner(),
        )

        self.assertEqual(manifests[0].status, "failed_sam2_refinement")
        self.assertEqual(manifests[0].rows[0].sam2_refine_status, "failed")
        self.assertIn("intentional SAM 2 failure", manifests[0].rows[0].sam2_refine_error)

    def test_roof_mask_falls_back_to_coarse_prior_when_roof_sam2_fails(self) -> None:
        sequence_id = "GS999505"
        self.write_sequence(sequence_id, frame_count=1)

        manifests = run_sam2_refinement(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
            ),
            refiner=FailingSam2Refiner(),
        )

        self.assertEqual(manifests[0].status, "ready")
        frame = manifests[0].rows[0]
        self.assertEqual(frame.sam2_refine_status, "refined")
        self.assertEqual(frame.sam2_mask_count, 0)
        self.assertIn("roof SAM 2 refinement failed", frame.sam2_refine_error)
        self.assertEqual(frame.roof_mask_status, "fallback")
        self.assertEqual(frame.roof_mask_source, "coarse_prior_fallback")
        self.assertTrue(frame.roof_mask_area_px and frame.roof_mask_area_px > 0)
        self.assertTrue(frame.roof_mask_path.is_file())
        roof_mask = np.asarray(Image.open(frame.roof_mask_path))
        self.assertEqual(roof_mask.shape, (16, 32))
        self.assertGreater(int((roof_mask > 0).sum()), 0)

    def test_sky_mask_uses_semantic_fallback_when_optional_sam2_fails(self) -> None:
        sequence_id = "GS999507"
        self.write_sequence(sequence_id, frame_count=1)
        self.write_semantic_face(sequence_id, frame_number=1, face_name="up")

        manifests = run_sam2_refinement(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
            ),
            refiner=FailingSam2Refiner(),
        )

        frame = manifests[0].rows[0]
        self.assertEqual(manifests[0].status, "ready")
        self.assertEqual(frame.sam2_refine_status, "refined")
        self.assertIn("sky SAM 2 refinement failed", frame.sam2_refine_error)
        self.assertEqual(frame.sky_mask_status, "generated")
        self.assertIn("semantic_mask2former", frame.sky_mask_source)
        self.assertTrue(frame.sky_mask_area_px and frame.sky_mask_area_px > 0)
        self.assertTrue(frame.sky_mask_path.is_file())


if __name__ == "__main__":
    unittest.main()
