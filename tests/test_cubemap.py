from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.cubemap import (
    FACE_ORDER,
    cubemap_to_erp,
    erp_to_cubemap,
    load_cubemap_projection,
    save_cubemap_projection,
)
from urpan_inpaint.erp import ERPFrame
from urpan_inpaint.projection import run_cubemap_projection


def build_seam_continuous_erp(width: int, height: int) -> ERPFrame:
    theta = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False, dtype=np.float32)
    phi = np.linspace(0.0, 1.0, height, endpoint=False, dtype=np.float32)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    red = np.rint((np.sin(theta_grid) * 0.5 + 0.5) * 255.0).astype(np.uint8)
    green = np.rint((np.cos(theta_grid) * 0.5 + 0.5) * 255.0).astype(np.uint8)
    blue = np.rint(phi_grid * 255.0).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    return ERPFrame(
        path=Path("synthetic.jpg"),
        width=width,
        height=height,
        channels=3,
        dtype="uint8",
        source_mode="RGB",
        file_sha256="",
        rgb=rgb,
    )


class CubemapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_erp_cubemap_roundtrip_is_deterministic(self) -> None:
        erp = build_seam_continuous_erp(width=128, height=64)
        projection = erp_to_cubemap(erp, face_size=32, overlap_px=4)
        roundtrip_a = cubemap_to_erp(projection)
        roundtrip_b = cubemap_to_erp(projection)
        mae = np.abs(roundtrip_a.astype(np.int16) - erp.rgb.astype(np.int16)).mean()

        self.assertTrue(np.array_equal(roundtrip_a, roundtrip_b))
        self.assertEqual(roundtrip_a.shape, erp.rgb.shape)
        self.assertLess(mae, 8.0)

    def test_save_and_load_projection_preserves_cache_and_down_face(self) -> None:
        erp = build_seam_continuous_erp(width=64, height=32)
        projection = erp_to_cubemap(erp, face_size=16, overlap_px=2)
        cache_dir = self.root / "cubemap-cache"
        metadata_path = save_cubemap_projection(projection, cache_dir)
        loaded = load_cubemap_projection(cache_dir)

        self.assertTrue(metadata_path.is_file())
        self.assertEqual(len(list(cache_dir.glob("*.npz"))), 6)
        for face_name in FACE_ORDER:
            self.assertTrue((cache_dir / f"{face_name}.npz").is_file())
            self.assertIn(face_name, loaded.faces)
        self.assertTrue((cache_dir / "down.npz").is_file())
        self.assertTrue(np.array_equal(cubemap_to_erp(projection), cubemap_to_erp(loaded)))

    def test_run_cubemap_projection_updates_manifest_and_cache(self) -> None:
        dataset_root = self.root / "dataset"
        output_root = self.root / "output"
        dataset_root.mkdir()
        sequence_id = "GS999201"
        sequence_dir = dataset_root / sequence_id
        fixed_dir = sequence_dir / "fixed"
        fixed_dir.mkdir(parents=True)

        image_path = fixed_dir / f"{sequence_id}-frame-000001.jpg"
        pixels = np.zeros((8, 16, 3), dtype=np.uint8)
        pixels[:, :, 0] = 64
        pixels[:, :, 1] = 128
        pixels[:, :, 2] = 192
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

        manifests = run_cubemap_projection(
            IndexConfig(
                dataset_root=dataset_root,
                output_root=output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                cache_cubemap_faces=True,
            )
        )

        self.assertEqual(len(manifests), 1)
        frame = manifests[0].rows[0]
        self.assertEqual(manifests[0].status, "ready")
        self.assertEqual(frame.cubemap_projection_status, "projected")
        self.assertEqual(frame.cubemap_face_size, 8)
        self.assertEqual(frame.cubemap_overlap_px, 2)
        self.assertEqual(frame.cubemap_total_face_size, 12)
        self.assertEqual(frame.erp_width, 16)
        self.assertEqual(frame.erp_height, 8)
        self.assertTrue(frame.cubemap_faces_cached)
        self.assertIsNotNone(frame.cubemap_metadata_path)
        self.assertTrue(frame.cubemap_metadata_path and frame.cubemap_metadata_path.is_file())
        self.assertTrue((frame.cubemap_cache_dir / "down.npz").is_file())
        self.assertTrue((frame.cubemap_cache_dir / "projection.json").is_file())


if __name__ == "__main__":
    unittest.main()
