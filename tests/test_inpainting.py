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
from urpan_inpaint.inpainting import _compose_masked_prediction, run_propainter_inpainting


class FakeProPainterBackend:
    model_id = "fake/propainter"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def inpaint_clip(
        self,
        face_name: str,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        config: IndexConfig,
    ) -> list[np.ndarray]:
        self.calls.append(
            {
                "face_name": face_name,
                "frame_count": len(frames),
                "mask_pixels": [int((mask > 0).sum()) for mask in masks],
            }
        )
        outputs: list[np.ndarray] = []
        for frame, mask in zip(frames, masks):
            output = frame.copy()
            output[mask > 0] = np.asarray([250, 20, 10], dtype=np.uint8)
            outputs.append(output)
        return outputs


class OomProPainterBackend:
    model_id = "fake/oom-propainter"

    def __init__(self) -> None:
        self.calls = 0

    def inpaint_clip(
        self,
        face_name: str,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        config: IndexConfig,
    ) -> list[np.ndarray]:
        self.calls += 1
        raise RuntimeError("CUDA out of memory")


class FakeLaMaBackend:
    model_id = "fake/lama"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def inpaint_image(
        self,
        face_name: str,
        image: np.ndarray,
        mask: np.ndarray,
        config: IndexConfig,
    ) -> np.ndarray:
        self.calls.append(
            {
                "face_name": face_name,
                "mask_pixels": int((mask > 0).sum()),
            }
        )
        output = image.copy()
        output[mask > 0] = np.asarray([10, 240, 30], dtype=np.uint8)
        return output


class InpaintingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_root = self.root / "dataset"
        self.output_root = self.root / "output"
        self.dataset_root.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_sequence_with_inpaint_masks(self, sequence_id: str, frame_count: int) -> None:
        sequence_dir = self.dataset_root / sequence_id
        fixed_dir = sequence_dir / "fixed"
        fixed_dir.mkdir(parents=True)
        mask_dir = self.output_root / sequence_id / "masks" / "inpaint"
        mask_dir.mkdir(parents=True)

        rows = []
        for index in range(frame_count):
            frame_number = index + 1
            stem = f"{sequence_id}-frame-{frame_number:06d}"
            image_path = fixed_dir / f"{stem}.jpg"
            pixels = np.zeros((16, 32, 3), dtype=np.uint8)
            pixels[:, :, 0] = 40 + index
            pixels[:, :, 1] = np.arange(32, dtype=np.uint8)[None, :]
            pixels[:, :, 2] = np.arange(16, dtype=np.uint8)[:, None]
            Image.fromarray(pixels, mode="RGB").save(image_path, format="JPEG", quality=100, subsampling=0)

            mask = np.zeros((16, 32), dtype=np.uint8)
            mask[6:11, 11:21] = 255
            Image.fromarray(mask).save(mask_dir / f"{stem}.png")

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

    def test_compose_masked_prediction_preserves_unmasked_pixels(self) -> None:
        original = np.zeros((2, 2, 3), dtype=np.uint8)
        original[:, :] = [1, 2, 3]
        prediction = np.zeros((2, 2, 3), dtype=np.uint8)
        prediction[:, :] = [9, 8, 7]
        mask = np.zeros((2, 2), dtype=np.uint8)
        mask[0, 1] = 255

        composed = _compose_masked_prediction(original, prediction, mask)

        self.assertTrue(np.array_equal(composed[0, 0], original[0, 0]))
        self.assertTrue(np.array_equal(composed[1, 1], original[1, 1]))
        self.assertTrue(np.array_equal(composed[0, 1], prediction[0, 1]))

    def test_run_propainter_inpainting_chunks_faces_and_writes_outputs(self) -> None:
        sequence_id = "GS999801"
        self.write_sequence_with_inpaint_masks(sequence_id, frame_count=4)
        backend = FakeProPainterBackend()

        manifests = run_propainter_inpainting(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                inpaint_window_size=3,
                inpaint_window_stride=1,
                propainter_chunk_size=2,
                propainter_face_feather_px=2,
                single_frame_min_mask_area_px=1,
            ),
            backend=backend,
        )

        manifest = manifests[0]
        self.assertEqual(manifest.status, "ready")
        self.assertEqual([row.propainter_status for row in manifest.rows], ["inpainted"] * 4)
        self.assertEqual([row.propainter_window_count for row in manifest.rows], [2] * 4)
        self.assertEqual([row.propainter_chunk_count for row in manifest.rows], [24] * 4)
        self.assertEqual(len(backend.calls), 24)
        self.assertTrue(all(call["frame_count"] <= 2 for call in backend.calls))

        call_faces = [str(call["face_name"]) for call in backend.calls]
        self.assertEqual([call_faces[index * 4] for index in range(len(FACE_ORDER))], list(FACE_ORDER))

        first_frame = manifest.rows[0]
        self.assertTrue(first_frame.rgb_output_path.is_file())
        self.assertTrue(first_frame.rgba_output_path.is_file())
        source_rgb = np.asarray(Image.open(first_frame.resolved_fixed_path).convert("RGB"), dtype=np.uint8)
        output_rgb = np.asarray(Image.open(first_frame.rgb_output_path).convert("RGB"), dtype=np.uint8)
        inpaint_mask = np.asarray(Image.open(first_frame.inpaint_mask_path).convert("L"), dtype=np.uint8)
        self.assertTrue(np.array_equal(output_rgb[inpaint_mask == 0], source_rgb[inpaint_mask == 0]))
        rgba = np.asarray(Image.open(first_frame.rgba_output_path).convert("RGBA"), dtype=np.uint8)
        self.assertTrue(np.all(rgba[..., 3] == 255))

        face_dir = first_frame.propainter_output_dir / "front"
        self.assertTrue((face_dir / f"{first_frame.frame_stem}.with_overlap.png").is_file())
        self.assertTrue((face_dir / f"{first_frame.frame_stem}.png").is_file())
        self.assertTrue((face_dir / f"{first_frame.frame_stem}.mask.png").is_file())

        metadata_path = self.output_root / sequence_id / "propainter" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["face_order"], list(FACE_ORDER))
        self.assertEqual(len(metadata["windows"]), 2)
        self.assertEqual(len(metadata["chunks"]), 24)

    def test_force_single_frame_fallback_runs_lama_per_face(self) -> None:
        sequence_id = "GS999802"
        self.write_sequence_with_inpaint_masks(sequence_id, frame_count=2)
        propainter = FakeProPainterBackend()
        lama = FakeLaMaBackend()

        manifests = run_propainter_inpainting(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                inpaint_window_size=3,
                inpaint_window_stride=1,
                force_single_frame_fallback=True,
                single_frame_min_mask_area_px=1,
            ),
            backend=propainter,
            fallback_backend=lama,
        )

        manifest = manifests[0]
        self.assertEqual(manifest.status, "ready")
        self.assertEqual(len(propainter.calls), 0)
        self.assertEqual(len(lama.calls), 12)
        self.assertEqual([row.propainter_status for row in manifest.rows], ["fallback_lama"] * 2)
        self.assertEqual([row.lama_status for row in manifest.rows], ["inpainted"] * 2)
        self.assertEqual([row.single_frame_fallback_reason for row in manifest.rows], ["force_single_frame_fallback"] * 2)
        metadata_path = self.output_root / sequence_id / "lama_fallback" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["fallback_reason"], "force_single_frame_fallback")

    def test_propainter_oom_after_retries_falls_back_to_lama(self) -> None:
        sequence_id = "GS999803"
        self.write_sequence_with_inpaint_masks(sequence_id, frame_count=3)
        propainter = OomProPainterBackend()
        lama = FakeLaMaBackend()

        manifests = run_propainter_inpainting(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                inpaint_window_size=3,
                inpaint_window_stride=1,
                propainter_chunk_size=4,
                single_frame_min_mask_area_px=1,
            ),
            backend=propainter,
            fallback_backend=lama,
        )

        manifest = manifests[0]
        self.assertEqual(manifest.status, "ready")
        self.assertGreaterEqual(propainter.calls, 3)
        self.assertEqual(len(lama.calls), 18)
        self.assertEqual({row.single_frame_fallback_reason for row in manifest.rows}, {"propainter_oom_all_retry_levels"})
        self.assertTrue(all("out-of-memory" in row.propainter_error for row in manifest.rows))

    def test_propainter_model_load_failure_falls_back_to_lama(self) -> None:
        sequence_id = "GS999804"
        self.write_sequence_with_inpaint_masks(sequence_id, frame_count=2)
        lama = FakeLaMaBackend()

        manifests = run_propainter_inpainting(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                inpaint_window_size=3,
                inpaint_window_stride=1,
                single_frame_min_mask_area_px=1,
            ),
            fallback_backend=lama,
        )

        manifest = manifests[0]
        self.assertEqual(manifest.status, "ready")
        self.assertEqual(len(lama.calls), 12)
        self.assertTrue(
            all(row.single_frame_fallback_reason.startswith("propainter_model_load_failure") for row in manifest.rows)
        )

    def test_short_window_uses_lama_fallback(self) -> None:
        sequence_id = "GS999805"
        self.write_sequence_with_inpaint_masks(sequence_id, frame_count=1)
        propainter = FakeProPainterBackend()
        lama = FakeLaMaBackend()

        manifests = run_propainter_inpainting(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                propainter_min_window_frames=2,
                single_frame_min_mask_area_px=1,
            ),
            backend=propainter,
            fallback_backend=lama,
        )

        manifest = manifests[0]
        self.assertEqual(manifest.status, "ready")
        self.assertEqual(len(propainter.calls), 0)
        self.assertEqual(len(lama.calls), 6)
        self.assertTrue(
            manifest.rows[0].single_frame_fallback_reason.startswith("window_length_below_minimum_viable_size")
        )

    def test_small_masks_use_lama_fallback(self) -> None:
        sequence_id = "GS999806"
        self.write_sequence_with_inpaint_masks(sequence_id, frame_count=2)
        propainter = FakeProPainterBackend()
        lama = FakeLaMaBackend()

        manifests = run_propainter_inpainting(
            IndexConfig(
                dataset_root=self.dataset_root,
                output_root=self.output_root,
                min_valid_frames=1,
                cube_face_size=8,
                cube_overlap_px=2,
                propainter_min_window_frames=2,
                single_frame_min_mask_area_px=10_000,
            ),
            backend=propainter,
            fallback_backend=lama,
        )

        manifest = manifests[0]
        self.assertEqual(manifest.status, "ready")
        self.assertEqual(len(propainter.calls), 0)
        self.assertEqual(len(lama.calls), 12)
        self.assertTrue(all(row.single_frame_fallback_reason.startswith("mask_area_too_small") for row in manifest.rows))


if __name__ == "__main__":
    unittest.main()
