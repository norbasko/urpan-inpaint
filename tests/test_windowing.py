from __future__ import annotations

import unittest
from pathlib import Path

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.models import FrameRecord, SequenceManifest
from urpan_inpaint.windowing import (
    build_inpaint_windows,
    build_reconciliation_plan,
    build_sequence_inpaint_windows,
    reconcile_window_predictions,
)


def make_frame(index: int, exists: bool = True) -> FrameRecord:
    base = Path("/tmp/out")
    return FrameRecord(
        sequence_id="GS999701",
        csv_row_number=index,
        source_index=str(index),
        csv_path=Path("/tmp/GS999701/gps-fixed.csv"),
        source_path=f"frame-{index:06d}.jpg",
        resolved_fixed_path=Path(f"/tmp/GS999701/fixed/frame-{index:06d}.jpg"),
        frame_name=f"frame-{index:06d}.jpg",
        frame_stem=f"frame-{index:06d}",
        frame_number=index,
        ts=str(index),
        lat="39.0",
        lon="-84.0",
        alt="185.0",
        speed="1.0",
        speed3d="1.0",
        date="2020-09-24 21:05:33.914000+00:00",
        file_exists=exists,
        processing_status="ready" if exists else "missing_source",
        skip_reason="" if exists else "missing",
        valid_order_index=index if exists else None,
        rgb_output_path=base / "rgb" / f"frame-{index:06d}.png",
        rgba_output_path=base / "rgba" / f"frame-{index:06d}.png",
        dynamic_mask_path=base / "masks" / "dynamic" / f"frame-{index:06d}.png",
        roof_mask_path=base / "masks" / "roof" / f"frame-{index:06d}.png",
        sky_mask_path=base / "masks" / "sky" / f"frame-{index:06d}.png",
        inpaint_mask_path=base / "masks" / "inpaint" / f"frame-{index:06d}.png",
        union_debug_path=base / "masks" / "union_debug" / f"frame-{index:06d}.png",
        overlay_output_path=base / "qa" / "overlays" / f"frame-{index:06d}.overlay.png",
        cubemap_cache_dir=base / "cubemap" / f"frame-{index:06d}",
    )


class WindowingTests(unittest.TestCase):
    def test_default_windows_overlap_and_cover_sequence(self) -> None:
        windows = build_inpaint_windows(n_frames=60, window_size=24, window_stride=12)

        self.assertEqual([(window.start, window.end) for window in windows], [(0, 24), (12, 36), (24, 48), (36, 60)])
        self.assertTrue(all(left.end > right.start for left, right in zip(windows, windows[1:])))
        covered = sorted({frame for window in windows for frame in window.frame_indices})
        self.assertEqual(covered, list(range(60)))

    def test_short_sequence_uses_single_short_window(self) -> None:
        windows = build_inpaint_windows(n_frames=5, window_size=24, window_stride=12)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start, 0)
        self.assertEqual(windows[0].end, 5)
        self.assertEqual(windows[0].frame_indices, (0, 1, 2, 3, 4))

    def test_tail_window_is_full_size_when_possible(self) -> None:
        windows = build_inpaint_windows(n_frames=25, window_size=24, window_stride=12)

        self.assertEqual([(window.start, window.end) for window in windows], [(0, 24), (1, 25)])
        self.assertEqual([window.size for window in windows], [24, 24])

    def test_invalid_stride_rejected_because_windows_would_not_overlap(self) -> None:
        with self.assertRaises(ValueError):
            build_inpaint_windows(n_frames=48, window_size=24, window_stride=24)

    def test_reconciliation_prefers_center_then_earlier_window_on_tie(self) -> None:
        windows = build_inpaint_windows(n_frames=5, window_size=4, window_stride=2)
        plan = build_reconciliation_plan(n_frames=5, windows=windows)

        self.assertEqual([(window.start, window.end) for window in windows], [(0, 4), (1, 5)])
        self.assertEqual(plan[2].window_index, 0)
        self.assertEqual(plan[3].window_index, 1)

    def test_reconcile_window_predictions_uses_actual_frame_indices(self) -> None:
        windows = build_inpaint_windows(
            n_frames=5,
            window_size=4,
            window_stride=2,
            frame_indices=[10, 11, 12, 13, 14],
        )
        predictions = {
            0: {10: "w0f10", 11: "w0f11", 12: "w0f12", 13: "w0f13"},
            1: {11: "w1f11", 12: "w1f12", 13: "w1f13", 14: "w1f14"},
        }

        reconciled = reconcile_window_predictions(windows, predictions)

        self.assertEqual(reconciled, ["w0f10", "w0f11", "w0f12", "w1f13", "w1f14"])

    def test_sequence_windowing_ignores_missing_frames_but_keeps_row_indices(self) -> None:
        rows = [make_frame(0), make_frame(1, exists=False), make_frame(2), make_frame(3), make_frame(4)]
        manifest = SequenceManifest(
            sequence_id="GS999701",
            sequence_dir=Path("/tmp/GS999701"),
            csv_path=Path("/tmp/GS999701/gps-fixed.csv"),
            output_dir=Path("/tmp/out/GS999701"),
            status="ready",
            failure_reason=None,
            total_csv_rows=len(rows),
            valid_frames=4,
            skipped_frames=1,
            rows=rows,
        )

        windows = build_sequence_inpaint_windows(
            manifest,
            IndexConfig(inpaint_window_size=3, inpaint_window_stride=1),
        )

        self.assertEqual([window.frame_indices for window in windows], [(0, 2, 3), (2, 3, 4)])


if __name__ == "__main__":
    unittest.main()
