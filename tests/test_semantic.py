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
from urpan_inpaint.semantic import resolve_target_class_ids, run_semantic_parsing


class FakeSemanticParser:
    model_id = "fake/mask2former-cityscapes"
    id2label = {
        0: "road",
        1: "sky",
        2: "person",
        3: "car",
        4: "bus",
        5: "train",
    }
    target_class_ids = resolve_target_class_ids(id2label)

    def parse_face(self, face_name: str, rgb: np.ndarray):
        from urpan_inpaint.semantic import SemanticFacePrediction

        height, width = rgb.shape[:2]
        label_map = np.zeros((height, width), dtype=np.int32)
        label_map[: height // 3, :] = 1
        label_map[height // 3 : (2 * height) // 3, : width // 2] = 2
        label_map[height // 3 : (2 * height) // 3, width // 2 :] = 3
        label_map[(2 * height) // 3 :, : width // 2] = 4
        label_map[(2 * height) // 3 :, width // 2 :] = 5

        confidence_map = np.full((height, width), 0.95, dtype=np.float32)
        semantic_scores = np.stack(
            [np.full((height, width), fill_value=class_id, dtype=np.float16) for class_id in range(6)],
            axis=0,
        )
        panoptic_instance_map = (label_map + 100).astype(np.int32)
        panoptic_segments_info = [
            {"id": 101, "label_id": 1, "score": 0.99},
            {"id": 102, "label_id": 2, "score": 0.98},
        ]
        return SemanticFacePrediction(
            face_name=face_name,
            label_map=label_map,
            confidence_map=confidence_map,
            semantic_scores=semantic_scores,
            panoptic_instance_map=panoptic_instance_map,
            panoptic_segments_info=panoptic_segments_info,
        )


class FailingSemanticParser:
    model_id = "fake/failing"
    id2label = {0: "road", 1: "sky"}
    target_class_ids = resolve_target_class_ids(id2label)

    def parse_face(self, face_name: str, rgb: np.ndarray):
        raise RuntimeError(f"intentional failure on {face_name}")


class SemanticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_root = self.root / "dataset"
        self.output_root = self.root / "output"
        self.dataset_root.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_sequence(self, sequence_id: str) -> Path:
        sequence_dir = self.dataset_root / sequence_id
        fixed_dir = sequence_dir / "fixed"
        fixed_dir.mkdir(parents=True)
        image_path = fixed_dir / f"{sequence_id}-frame-000001.jpg"
        pixels = np.zeros((12, 24, 3), dtype=np.uint8)
        pixels[:, :, 0] = 50
        pixels[:, :, 1] = 100
        pixels[:, :, 2] = 150
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
        return sequence_dir

    def test_resolve_target_class_ids_cityscapes_like_mapping(self) -> None:
        id2label = {
            0: "road",
            1: "sky",
            2: "person",
            3: "rider",
            4: "car",
            5: "truck",
            6: "bus",
            7: "train",
            8: "motorcycle",
            9: "bicycle",
        }
        resolved = resolve_target_class_ids(id2label)
        self.assertEqual(resolved["sky"], 1)
        self.assertEqual(resolved["person"], 2)
        self.assertEqual(resolved["train"], 7)
        self.assertIsNone(resolved["trailer"])

    def test_run_semantic_parsing_writes_face_outputs_and_manifest(self) -> None:
        sequence_id = "GS999301"
        self.write_sequence(sequence_id)

        manifests = run_semantic_parsing(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                cache_cubemap_faces=True,
                semantic_save_logits=True,
            ),
            parser=FakeSemanticParser(),
        )

        self.assertEqual(len(manifests), 1)
        manifest = manifests[0]
        frame = manifest.rows[0]
        self.assertEqual(manifest.status, "ready")
        self.assertEqual(frame.semantic_parse_status, "parsed")
        self.assertTrue(frame.semantic_has_panoptic)
        self.assertTrue(frame.semantic_has_confidence)
        self.assertTrue(frame.semantic_has_logits)
        self.assertIsNotNone(frame.semantic_output_dir)
        self.assertTrue(frame.semantic_output_dir and frame.semantic_output_dir.is_dir())

        metadata_path = frame.semantic_output_dir / "metadata.json"
        self.assertTrue(metadata_path.is_file())
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata["model_id"], FakeSemanticParser.model_id)
        self.assertIn("rider", metadata["missing_target_classes"])

        for face_name in FACE_ORDER:
            face_npz = frame.semantic_output_dir / f"{face_name}.npz"
            self.assertTrue(face_npz.is_file())
            with np.load(face_npz) as payload:
                self.assertIn("label_map", payload)
                self.assertIn("target_masks", payload)
                self.assertIn("target_class_ids", payload)
                self.assertIn("target_class_names", payload)
                self.assertIn("confidence_map", payload)
                self.assertIn("semantic_scores", payload)
                self.assertIn("panoptic_instance_map", payload)
            self.assertTrue((frame.semantic_output_dir / f"{face_name}.panoptic.json").is_file())

        frames_csv = self.output_root / sequence_id / "manifests" / "frames.csv"
        with frames_csv.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["semantic_model_id"], FakeSemanticParser.model_id)
        self.assertEqual(rows[0]["semantic_parse_status"], "parsed")
        self.assertEqual(rows[0]["semantic_has_panoptic"], "1")
        self.assertEqual(rows[0]["semantic_has_confidence"], "1")
        self.assertEqual(rows[0]["semantic_has_logits"], "1")

    def test_run_semantic_parsing_marks_failures(self) -> None:
        sequence_id = "GS999302"
        self.write_sequence(sequence_id)

        manifests = run_semantic_parsing(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
            ),
            parser=FailingSemanticParser(),
        )

        self.assertEqual(manifests[0].status, "failed_semantic_parsing")
        self.assertEqual(manifests[0].rows[0].semantic_parse_status, "failed")
        self.assertIn("intentional failure", manifests[0].rows[0].semantic_parse_error)


if __name__ == "__main__":
    unittest.main()
