from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from urpan_inpaint.config import DEFAULT_GROUNDING_DINO_PROMPTS, IndexConfig
from urpan_inpaint.cubemap import FACE_ORDER
from urpan_inpaint.detection import FaceDetection, class_aware_nms, run_grounding_detection


class FakeGroundingDetector:
    model_id = "fake/grounding-dino"
    prompts = DEFAULT_GROUNDING_DINO_PROMPTS
    box_threshold = 0.25
    text_threshold = 0.25
    nms_iou_threshold = 0.7

    def detect_face(self, face_name: str, rgb: np.ndarray):
        from urpan_inpaint.detection import FaceDetections

        detections = [
            FaceDetection(
                box_xyxy=np.asarray([1.0, 1.0, 6.0, 6.0], dtype=np.float32),
                score=0.93,
                text="person",
                text_normalized="person",
            ),
            FaceDetection(
                box_xyxy=np.asarray([1.2, 1.2, 6.1, 6.1], dtype=np.float32),
                score=0.65,
                text="person",
                text_normalized="person",
            ),
            FaceDetection(
                box_xyxy=np.asarray([2.0, 2.0, 3.0, 3.0], dtype=np.float32),
                score=0.99,
                text="bicycle",
                text_normalized="bicycle",
            ),
            FaceDetection(
                box_xyxy=np.asarray([7.0, 1.0, 11.0, 5.0], dtype=np.float32),
                score=0.74,
                text="car",
                text_normalized="car",
            ),
        ]
        filtered = class_aware_nms(detections, self.nms_iou_threshold)
        return FaceDetections(face_name=face_name, detections=filtered, raw_detection_count=len(detections))


class FailingGroundingDetector:
    model_id = "fake/failing-grounding"
    prompts = DEFAULT_GROUNDING_DINO_PROMPTS
    box_threshold = 0.25
    text_threshold = 0.25
    nms_iou_threshold = 0.7

    def detect_face(self, face_name: str, rgb: np.ndarray):
        raise RuntimeError(f"intentional detector failure on {face_name}")


class DetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_root = self.root / "dataset"
        self.output_root = self.root / "output"
        self.dataset_root.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_sequence(self, sequence_id: str) -> None:
        sequence_dir = self.dataset_root / sequence_id
        fixed_dir = sequence_dir / "fixed"
        fixed_dir.mkdir(parents=True)
        image_path = fixed_dir / f"{sequence_id}-frame-000001.jpg"
        pixels = np.zeros((12, 24, 3), dtype=np.uint8)
        pixels[:, :, 0] = 40
        pixels[:, :, 1] = 90
        pixels[:, :, 2] = 140
        Image.fromarray(pixels, mode="RGB").save(image_path, format="JPEG", quality=100, subsampling=0)

        with (sequence_dir / "gps-fixed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "path", "ts", "lat", "lon", "alt", "speed", "speed3d", "date"])
            writer.writerow(
                [
                    "0",
                    str(image_path),
                    "0.0",
                    "39.0",
                    "-84.0",
                    "185.0",
                    "1.0",
                    "1.0",
                    "2020-09-24 21:05:33.914000+00:00",
                ]
            )

    def test_class_aware_nms_suppresses_same_class_overlap_but_keeps_small_box(self) -> None:
        detections = [
            FaceDetection(
                box_xyxy=np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
                score=0.95,
                text="person",
                text_normalized="person",
            ),
            FaceDetection(
                box_xyxy=np.asarray([1.0, 1.0, 10.5, 10.5], dtype=np.float32),
                score=0.70,
                text="person",
                text_normalized="person",
            ),
            FaceDetection(
                box_xyxy=np.asarray([2.0, 2.0, 3.0, 3.0], dtype=np.float32),
                score=0.98,
                text="bicycle",
                text_normalized="bicycle",
            ),
        ]
        filtered = class_aware_nms(detections, iou_threshold=0.7)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].text_normalized, "bicycle")
        self.assertTrue(any(det.text_normalized == "person" for det in filtered))

    def test_run_grounding_detection_writes_face_outputs_and_manifest(self) -> None:
        sequence_id = "GS999401"
        self.write_sequence(sequence_id)

        manifests = run_grounding_detection(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                cache_cubemap_faces=True,
            ),
            detector=FakeGroundingDetector(),
        )

        self.assertEqual(len(manifests), 1)
        manifest = manifests[0]
        frame = manifest.rows[0]
        self.assertEqual(manifest.status, "ready")
        self.assertEqual(frame.grounding_detect_status, "detected")
        self.assertEqual(frame.grounding_model_id, FakeGroundingDetector.model_id)
        self.assertEqual(frame.grounding_box_count, 18)
        self.assertIsNotNone(frame.grounding_output_dir)
        self.assertTrue(frame.grounding_output_dir and frame.grounding_output_dir.is_dir())

        metadata_path = frame.grounding_output_dir / "metadata.json"
        self.assertTrue(metadata_path.is_file())
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata["model_id"], FakeGroundingDetector.model_id)
        self.assertEqual(metadata["prompts"], list(DEFAULT_GROUNDING_DINO_PROMPTS))

        for face_name in FACE_ORDER:
            face_npz = frame.grounding_output_dir / f"{face_name}.npz"
            face_json = frame.grounding_output_dir / f"{face_name}.json"
            self.assertTrue(face_npz.is_file())
            self.assertTrue(face_json.is_file())
            with np.load(face_npz) as payload:
                self.assertIn("boxes_xyxy", payload)
                self.assertIn("scores", payload)
                self.assertIn("text_labels", payload)
                self.assertIn("text_labels_normalized", payload)
                self.assertEqual(payload["boxes_xyxy"].shape, (3, 4))
            with face_json.open("r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
            self.assertEqual(sidecar["raw_detection_count"], 4)
            self.assertEqual(sidecar["filtered_detection_count"], 3)

        frames_csv = self.output_root / sequence_id / "manifests" / "frames.csv"
        with frames_csv.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["grounding_model_id"], FakeGroundingDetector.model_id)
        self.assertEqual(rows[0]["grounding_detect_status"], "detected")
        self.assertEqual(rows[0]["grounding_box_count"], "18")

    def test_run_grounding_detection_marks_failures(self) -> None:
        sequence_id = "GS999402"
        self.write_sequence(sequence_id)

        manifests = run_grounding_detection(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
            ),
            detector=FailingGroundingDetector(),
        )

        self.assertEqual(manifests[0].status, "failed_grounding_detection")
        self.assertEqual(manifests[0].rows[0].grounding_detect_status, "failed")
        self.assertIn("intentional detector failure", manifests[0].rows[0].grounding_detect_error)


if __name__ == "__main__":
    unittest.main()
