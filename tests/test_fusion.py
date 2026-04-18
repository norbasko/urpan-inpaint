from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.fusion import _morph_dynamic_or_roof, run_mask_fusion


class FusionTests(unittest.TestCase):
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
            pixels[:, :, 0] = 40 + index
            pixels[:, :, 1] = 90
            pixels[:, :, 2] = 140
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

    def write_semantic_dynamic_face(self, sequence_id: str, frame_number: int, face_name: str = "front") -> None:
        stem = f"{sequence_id}-frame-{frame_number:06d}"
        output_dir = self.output_root / sequence_id / "cubemap" / stem / "semantic_mask2former"
        output_dir.mkdir(parents=True)
        car = np.zeros((12, 12), dtype=np.uint8)
        car[4:8, 4:8] = 255
        sky = np.zeros((12, 12), dtype=np.uint8)
        np.savez_compressed(
            output_dir / f"{face_name}.npz",
            target_masks=np.stack([car, sky], axis=0),
            target_class_ids=np.asarray([13, 10], dtype=np.int32),
            target_class_names=np.asarray(["car", "sky"], dtype="<U32"),
        )

    def write_sam2_grounding_face(self, sequence_id: str, frame_number: int, face_name: str = "right") -> None:
        stem = f"{sequence_id}-frame-{frame_number:06d}"
        output_dir = self.output_root / sequence_id / "cubemap" / stem / "sam2_refined"
        output_dir.mkdir(parents=True)
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[3:7, 3:7] = 255
        np.savez_compressed(
            output_dir / f"{face_name}.npz",
            masks=np.expand_dims(mask, axis=0),
            class_text=np.asarray(["car"], dtype="<U128"),
            source=np.asarray(["grounding_box"], dtype="<U64"),
        )

    def write_refined_roof_and_sky(self, sequence_id: str, frame_number: int, sky_value: int) -> None:
        stem = f"{sequence_id}-frame-{frame_number:06d}"
        roof_dir = self.output_root / sequence_id / "masks" / "roof"
        sky_dir = self.output_root / sequence_id / "masks" / "sky"
        roof_dir.mkdir(parents=True)
        sky_dir.mkdir(parents=True)
        roof = np.zeros((16, 32), dtype=np.uint8)
        roof[10:14, 13:19] = 255
        sky = np.full((16, 32), sky_value, dtype=np.uint8)
        Image.fromarray(roof).save(roof_dir / f"{stem}.png")
        Image.fromarray(sky).save(sky_dir / f"{stem}.png")

    def test_morphology_fills_holes_removes_small_components_and_dilates(self) -> None:
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        mask[3:5, 3:5] = 0
        mask[8, 8] = 255

        result = _morph_dynamic_or_roof(
            mask,
            min_area_px=4,
            dilate_px=1,
            erode_after_dilate_px=0,
        )

        self.assertEqual(result[4, 4], 255)
        self.assertEqual(result[8, 8], 0)
        self.assertEqual(result[1, 3], 255)

    def test_run_mask_fusion_writes_final_masks_and_manifest(self) -> None:
        sequence_id = "GS999601"
        self.write_sequence(sequence_id, frame_count=1)
        self.write_semantic_dynamic_face(sequence_id, frame_number=1)
        self.write_sam2_grounding_face(sequence_id, frame_number=1)
        self.write_refined_roof_and_sky(sequence_id, frame_number=1, sky_value=0)

        manifests = run_mask_fusion(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                dyn_min_component_area_px=1,
                roof_min_component_area_px=1,
                dyn_dilate_px=1,
                roof_dilate_px=1,
                sky_mask_erp_smoothing_iterations=0,
            )
        )

        frame = manifests[0].rows[0]
        self.assertEqual(manifests[0].status, "ready")
        self.assertEqual(frame.mask_fusion_status, "fused")
        self.assertEqual(frame.dynamic_mask_status, "fused")
        self.assertEqual(frame.roof_mask_status, "fused")
        self.assertEqual(frame.sky_mask_status, "fused")
        self.assertEqual(frame.inpaint_mask_status, "fused")
        self.assertTrue(frame.dynamic_mask_area_px and frame.dynamic_mask_area_px > 0)
        self.assertTrue(frame.roof_mask_area_px and frame.roof_mask_area_px > 0)
        self.assertTrue(frame.inpaint_mask_area_px and frame.inpaint_mask_area_px > 0)
        self.assertTrue(frame.dynamic_mask_path.is_file())
        self.assertTrue(frame.roof_mask_path.is_file())
        self.assertTrue(frame.sky_mask_path.is_file())
        self.assertTrue(frame.inpaint_mask_path.is_file())
        self.assertTrue(frame.union_debug_path.is_file())

    def test_inpaint_excludes_sky(self) -> None:
        sequence_id = "GS999602"
        self.write_sequence(sequence_id, frame_count=1)
        self.write_semantic_dynamic_face(sequence_id, frame_number=1)
        self.write_refined_roof_and_sky(sequence_id, frame_number=1, sky_value=255)

        manifests = run_mask_fusion(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                dyn_min_component_area_px=1,
                roof_min_component_area_px=1,
                dyn_dilate_px=1,
                roof_dilate_px=1,
                sky_mask_erp_smoothing_iterations=0,
            )
        )

        frame = manifests[0].rows[0]
        dynamic = np.asarray(Image.open(frame.dynamic_mask_path))
        roof = np.asarray(Image.open(frame.roof_mask_path))
        inpaint = np.asarray(Image.open(frame.inpaint_mask_path))
        self.assertGreater(int((dynamic > 0).sum()), 0)
        self.assertGreater(int((roof > 0).sum()), 0)
        self.assertEqual(int((inpaint > 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
