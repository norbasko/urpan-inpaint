from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from urpan_inpaint.erp import circular_crop_erp, circular_pad_erp_horizontally, load_erp_rgb


class ERPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_load_erp_rgb_preserves_dimensions_and_checksum(self) -> None:
        image_path = self.root / "gray.jpg"
        pixels = np.array(
            [
                [0, 64, 128],
                [192, 224, 255],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(pixels, mode="L").save(image_path, format="JPEG", quality=100, subsampling=0)

        frame = load_erp_rgb(image_path)
        expected_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()

        self.assertEqual(frame.width, 3)
        self.assertEqual(frame.height, 2)
        self.assertEqual(frame.channels, 3)
        self.assertEqual(frame.dtype, "uint8")
        self.assertEqual(frame.source_mode, "L")
        self.assertEqual(frame.file_sha256, expected_sha)
        self.assertEqual(frame.rgb.shape, (2, 3, 3))
        self.assertTrue(np.array_equal(frame.rgb[:, :, 0], frame.rgb[:, :, 1]))
        self.assertTrue(np.array_equal(frame.rgb[:, :, 1], frame.rgb[:, :, 2]))

    def test_circular_padding_and_crop_wrap_erp_seam(self) -> None:
        rgb = np.array(
            [
                [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            ],
            dtype=np.uint8,
        )

        padded = circular_pad_erp_horizontally(rgb, left=1, right=2)
        cropped = circular_crop_erp(rgb, x_start=3, width=4)

        self.assertEqual(padded.shape, (1, 7, 3))
        self.assertTrue(np.array_equal(padded[:, 0, :], rgb[:, -1, :]))
        self.assertTrue(np.array_equal(padded[:, -2, :], rgb[:, 0, :]))
        self.assertTrue(np.array_equal(padded[:, -1, :], rgb[:, 1, :]))
        self.assertTrue(np.array_equal(cropped[0, :, 0], np.array([4, 1, 2, 3], dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
