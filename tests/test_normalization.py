from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.normalization import run_erp_normalization


class NormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_root = self.root / "dataset"
        self.output_root = self.root / "output"
        self.dataset_root.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_sequence_csv(self, sequence_id: str, frame_names: list[str]) -> None:
        sequence_dir = self.dataset_root / sequence_id
        fixed_dir = sequence_dir / "fixed"
        fixed_dir.mkdir(parents=True)

        with (sequence_dir / "gps-fixed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "path", "ts", "lat", "lon", "alt", "speed", "speed3d", "date"])
            for index, frame_name in enumerate(frame_names):
                writer.writerow(
                    [
                        str(index),
                        str(fixed_dir / frame_name),
                        f"{index * 0.5}",
                        "39.0",
                        "-84.0",
                        "185.0",
                        "1.0",
                        "1.0",
                        "2020-09-24 21:05:33.914000+00:00",
                    ]
                )

    def write_rgb_jpeg(self, path: Path, width: int, height: int) -> None:
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[:, :, 0] = 10
        pixels[:, :, 1] = 20
        pixels[:, :, 2] = 30
        Image.fromarray(pixels, mode="RGB").save(path, format="JPEG", quality=100, subsampling=0)

    def test_run_erp_normalization_records_dimensions_in_manifest(self) -> None:
        sequence_id = "GS999101"
        frame_names = ["GS999101-frame-000001.jpg", "GS999101-frame-000002.jpg"]
        self.write_sequence_csv(sequence_id, frame_names)
        fixed_dir = self.dataset_root / sequence_id / "fixed"
        self.write_rgb_jpeg(fixed_dir / frame_names[0], width=8, height=4)
        self.write_rgb_jpeg(fixed_dir / frame_names[1], width=8, height=4)

        manifests = run_erp_normalization(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
            )
        )

        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0].status, "ready")
        self.assertTrue(all(row.erp_normalization_status == "normalized" for row in manifests[0].rows))
        self.assertTrue(all(row.erp_width == 8 for row in manifests[0].rows))
        self.assertTrue(all(row.erp_height == 4 for row in manifests[0].rows))
        self.assertTrue(all(row.erp_horizontal_wrap_mode == "circular" for row in manifests[0].rows))

        frames_csv = self.output_root / sequence_id / "manifests" / "frames.csv"
        summary_json = self.output_root / sequence_id / "manifests" / "sequence_summary.json"
        self.assertTrue(frames_csv.is_file())
        self.assertTrue(summary_json.is_file())

        with frames_csv.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["erp_width"], "8")
        self.assertEqual(rows[0]["erp_height"], "4")
        self.assertEqual(rows[0]["erp_channels"], "3")
        self.assertEqual(rows[0]["erp_dtype"], "uint8")
        self.assertEqual(rows[0]["erp_horizontal_wrap_mode"], "circular")
        self.assertEqual(rows[0]["erp_normalization_status"], "normalized")
        self.assertTrue(rows[0]["erp_file_sha256"])

        with summary_json.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(summary["normalized_frames"], 2)
        self.assertEqual(summary["normalization_failed_frames"], [])

    def test_run_erp_normalization_marks_corrupt_frame_failed(self) -> None:
        sequence_id = "GS999102"
        frame_names = ["GS999102-frame-000001.jpg", "GS999102-frame-000002.jpg"]
        self.write_sequence_csv(sequence_id, frame_names)
        fixed_dir = self.dataset_root / sequence_id / "fixed"
        self.write_rgb_jpeg(fixed_dir / frame_names[0], width=6, height=3)
        (fixed_dir / frame_names[1]).write_bytes(b"not-a-jpeg")

        manifests = run_erp_normalization(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
            )
        )

        self.assertEqual(manifests[0].status, "failed_erp_normalization")
        self.assertEqual([row.erp_normalization_status for row in manifests[0].rows], ["normalized", "failed"])
        self.assertTrue(manifests[0].rows[1].erp_normalization_error)


if __name__ == "__main__":
    unittest.main()
