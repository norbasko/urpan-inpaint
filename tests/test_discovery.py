from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.discovery import index_sequence, run_indexing


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_root = self.root / "dataset"
        self.output_root = self.root / "output"
        self.dataset_root.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_sequence(self, sequence_id: str, rows: list[dict[str, str]], existing_frames: list[str]) -> Path:
        sequence_dir = self.dataset_root / sequence_id
        fixed_dir = sequence_dir / "fixed"
        fixed_dir.mkdir(parents=True)

        for frame_name in existing_frames:
            (fixed_dir / frame_name).write_bytes(b"frame")

        with (sequence_dir / "gps-fixed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "path", "ts", "lat", "lon", "alt", "speed", "speed3d", "date"])
            for row in rows:
                writer.writerow(
                    [
                        row["index"],
                        row["path"],
                        row["ts"],
                        row["lat"],
                        row["lon"],
                        row["alt"],
                        row["speed"],
                        row["speed3d"],
                        row["date"],
                    ]
                )
        return sequence_dir

    def base_row(self, sequence_id: str, frame_name: str, index: int) -> dict[str, str]:
        return {
            "index": str(index),
            "path": str(self.dataset_root / sequence_id / "fixed" / frame_name),
            "ts": f"{index * 0.5}",
            "lat": "39.0",
            "lon": "-84.0",
            "alt": "185.0",
            "speed": "1.0",
            "speed3d": "1.0",
            "date": "2020-09-24 21:05:33.914000+00:00",
        }

    def test_index_sequence_preserves_csv_order_and_skips_missing_frames(self) -> None:
        sequence_id = "GS999001"
        rows = [
            self.base_row(sequence_id, "GS999001-frame-000003.jpg", 0),
            self.base_row(sequence_id, "GS999001-frame-000001.jpg", 1),
            self.base_row(sequence_id, "GS999001-frame-000002.jpg", 2),
        ]
        sequence_dir = self.write_sequence(
            sequence_id=sequence_id,
            rows=rows,
            existing_frames=["GS999001-frame-000003.jpg", "GS999001-frame-000001.jpg"],
        )
        manifest = index_sequence(
            sequence_dir,
            IndexConfig(dataset_root=self.dataset_root, output_root=self.output_root, min_valid_frames=2),
        )

        self.assertEqual(manifest.status, "ready")
        self.assertEqual(manifest.valid_frames, 2)
        self.assertEqual([row.frame_name for row in manifest.rows], [Path(row["path"]).name for row in rows])
        self.assertEqual([row.valid_order_index for row in manifest.rows], [0, 1, None])
        self.assertEqual(manifest.rows[2].processing_status, "missing_source")

    def test_index_sequence_fails_when_csv_missing(self) -> None:
        sequence_dir = self.dataset_root / "GS999002"
        sequence_dir.mkdir()
        manifest = index_sequence(
            sequence_dir,
            IndexConfig(dataset_root=self.dataset_root, output_root=self.output_root),
        )
        self.assertEqual(manifest.status, "failed_missing_manifest")
        self.assertEqual(manifest.valid_frames, 0)

    def test_index_sequence_fails_when_too_few_valid_frames(self) -> None:
        sequence_id = "GS999003"
        rows = [
            self.base_row(sequence_id, "GS999003-frame-000001.jpg", 0),
            self.base_row(sequence_id, "GS999003-frame-000002.jpg", 1),
            self.base_row(sequence_id, "GS999003-frame-000003.jpg", 2),
        ]
        sequence_dir = self.write_sequence(
            sequence_id=sequence_id,
            rows=rows,
            existing_frames=["GS999003-frame-000001.jpg", "GS999003-frame-000002.jpg"],
        )
        manifest = index_sequence(
            sequence_dir,
            IndexConfig(dataset_root=self.dataset_root, output_root=self.output_root, min_valid_frames=3),
        )
        self.assertEqual(manifest.status, "failed_insufficient_valid_frames")
        self.assertIn("minimum is 3", manifest.failure_reason or "")

    def test_run_indexing_writes_manifest_files(self) -> None:
        sequence_id = "GS999004"
        rows = [
            self.base_row(sequence_id, "GS999004-frame-000001.jpg", 0),
            self.base_row(sequence_id, "GS999004-frame-000002.jpg", 1),
            self.base_row(sequence_id, "GS999004-frame-000003.jpg", 2),
        ]
        self.write_sequence(
            sequence_id=sequence_id,
            rows=rows,
            existing_frames=[Path(row["path"]).name for row in rows],
        )

        manifests = run_indexing(
            IndexConfig(dataset_root=self.dataset_root, output_root=self.output_root, min_valid_frames=3)
        )
        self.assertEqual(len(manifests), 1)

        frames_csv = self.output_root / sequence_id / "manifests" / "frames.csv"
        summary_json = self.output_root / sequence_id / "manifests" / "sequence_summary.json"
        self.assertTrue(frames_csv.is_file())
        self.assertTrue(summary_json.is_file())

        with summary_json.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["valid_frames"], 3)

    def test_index_sequence_normalizes_non_fixed_paths_to_fixed_directory(self) -> None:
        sequence_id = "GS999005"
        row = self.base_row(sequence_id, "GS999005-frame-000001.jpg", 0)
        row["path"] = str(self.dataset_root / sequence_id / "GS999005-frame-000001.jpg")
        sequence_dir = self.write_sequence(
            sequence_id=sequence_id,
            rows=[row],
            existing_frames=["GS999005-frame-000001.jpg"],
        )
        manifest = index_sequence(
            sequence_dir,
            IndexConfig(dataset_root=self.dataset_root, output_root=self.output_root, min_valid_frames=1),
        )
        self.assertEqual(manifest.status, "ready")
        self.assertEqual(
            manifest.rows[0].resolved_fixed_path,
            self.dataset_root / sequence_id / "fixed" / "GS999005-frame-000001.jpg",
        )


if __name__ == "__main__":
    unittest.main()
