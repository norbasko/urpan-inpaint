from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.qa import run_qa


class QaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_root = self.root / "dataset"
        self.output_root = self.root / "output"
        self.dataset_root.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_sequence_with_final_outputs(self, sequence_id: str, frame_count: int) -> None:
        sequence_dir = self.dataset_root / sequence_id
        fixed_dir = sequence_dir / "fixed"
        fixed_dir.mkdir(parents=True)
        output_dir = self.output_root / sequence_id
        rgb_dir = output_dir / "rgb"
        rgba_dir = output_dir / "rgba"
        mask_dirs = {
            name: output_dir / "masks" / name
            for name in ("dynamic", "roof", "sky", "inpaint")
        }
        rgb_dir.mkdir(parents=True)
        rgba_dir.mkdir(parents=True)
        for mask_dir in mask_dirs.values():
            mask_dir.mkdir(parents=True)

        rows = []
        for index in range(frame_count):
            frame_number = index + 1
            stem = f"{sequence_id}-frame-{frame_number:06d}"
            image_path = fixed_dir / f"{stem}.jpg"

            original = np.zeros((12, 20, 3), dtype=np.uint8)
            original[:, :, 0] = 30 + index
            original[:, :, 1] = 90
            original[:, :, 2] = 150
            Image.fromarray(original, mode="RGB").save(image_path, format="JPEG", quality=100, subsampling=0)

            final_rgb = np.full((12, 20, 3), fill_value=80 + index, dtype=np.uint8)
            sky = np.zeros((12, 20), dtype=np.uint8)
            dynamic = np.zeros((12, 20), dtype=np.uint8)
            roof = np.zeros((12, 20), dtype=np.uint8)
            sky[:3, :] = 255
            dynamic[5:7, 4:7] = 255
            roof[8:11, 12:16] = 255
            inpaint = np.where((dynamic > 0) | (roof > 0), 255, 0).astype(np.uint8)
            alpha = np.where(sky > 0, 0, 255).astype(np.uint8)
            final_rgba = np.dstack([final_rgb, alpha])

            Image.fromarray(final_rgb, mode="RGB").save(rgb_dir / f"{stem}.png", format="PNG")
            Image.fromarray(final_rgba, mode="RGBA").save(rgba_dir / f"{stem}.png", format="PNG")
            Image.fromarray(dynamic).save(mask_dirs["dynamic"] / f"{stem}.png", format="PNG")
            Image.fromarray(roof).save(mask_dirs["roof"] / f"{stem}.png", format="PNG")
            Image.fromarray(sky).save(mask_dirs["sky"] / f"{stem}.png", format="PNG")
            Image.fromarray(inpaint).save(mask_dirs["inpaint"] / f"{stem}.png", format="PNG")

            artifact_root = output_dir / "cubemap" / stem
            if index == 0:
                artifact_path = artifact_root / "propainter" / "front" / f"{stem}.png"
            else:
                artifact_path = artifact_root / "lama_fallback" / "front" / f"{stem}.png"
            artifact_path.parent.mkdir(parents=True)
            Image.fromarray(final_rgb, mode="RGB").save(artifact_path, format="PNG")

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

    def test_run_qa_writes_metrics_and_diagnostic_overlay(self) -> None:
        sequence_id = "GS999901"
        self.write_sequence_with_final_outputs(sequence_id, frame_count=2)

        manifests = run_qa(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                qa_sample_count=1,
                qa_diagnostic_panel_width_px=40,
            )
        )

        manifest = manifests[0]
        self.assertEqual(manifest.status, "ready")
        self.assertEqual([row.qa_status for row in manifest.rows], ["measured", "measured"])
        self.assertTrue(manifest.rows[0].qa_used_propainter)
        self.assertFalse(manifest.rows[0].qa_used_lama_fallback)
        self.assertFalse(manifest.rows[1].qa_used_propainter)
        self.assertTrue(manifest.rows[1].qa_used_lama_fallback)
        self.assertAlmostEqual(manifest.rows[0].qa_sky_ratio or 0.0, 0.25)
        self.assertAlmostEqual(manifest.rows[0].qa_dynamic_ratio or 0.0, 6 / 240)
        self.assertAlmostEqual(manifest.rows[0].qa_roof_ratio or 0.0, 12 / 240)
        self.assertAlmostEqual(manifest.rows[0].qa_inpaint_ratio or 0.0, 18 / 240)
        self.assertEqual(manifest.rows[0].qa_mask_components_dynamic, 1)
        self.assertEqual(manifest.rows[0].qa_mask_components_roof, 1)
        self.assertEqual(manifest.rows[0].qa_seam_delta_rgb, 0.0)
        self.assertEqual(manifest.rows[0].qa_seam_delta_alpha, 0.0)

        qa_metrics_dir = self.output_root / sequence_id / "qa" / "metrics"
        frame_metrics_path = qa_metrics_dir / "frame_metrics.csv"
        sequence_metrics_path = qa_metrics_dir / "sequence_metrics.json"
        self.assertTrue(frame_metrics_path.is_file())
        self.assertTrue(sequence_metrics_path.is_file())
        self.assertTrue(manifest.rows[0].overlay_output_path.is_file())

        with frame_metrics_path.open("r", newline="", encoding="utf-8") as handle:
            metric_rows = list(csv.DictReader(handle))
        self.assertEqual(len(metric_rows), 2)
        self.assertEqual(metric_rows[0]["qa_status"], "measured")
        self.assertEqual(metric_rows[0]["used_propainter"], "1")
        self.assertEqual(metric_rows[1]["used_lama_fallback"], "1")

        with sequence_metrics_path.open("r", encoding="utf-8") as handle:
            sequence_metrics = json.load(handle)
        self.assertEqual(sequence_metrics["frames_processed"], 2)
        self.assertEqual(sequence_metrics["frames_skipped"], 0)
        self.assertEqual(sequence_metrics["frames_with_fallback"], 1)
        self.assertEqual(sequence_metrics["failure_counts_by_stage"]["qa"], 0)
        self.assertAlmostEqual(sequence_metrics["median_mask_ratios"]["sky"], 0.25)


if __name__ == "__main__":
    unittest.main()
