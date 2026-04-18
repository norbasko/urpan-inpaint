from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TypeVar

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.models import FrameRecord, SequenceManifest


T = TypeVar("T")


@dataclass(frozen=True)
class InpaintWindow:
    window_index: int
    start: int
    end: int
    frame_indices: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def center(self) -> float:
        return (self.start + self.end - 1) * 0.5

    def contains_position(self, temporal_position: int) -> bool:
        return self.start <= temporal_position < self.end

    def frame_index_for_position(self, temporal_position: int) -> int:
        if not self.contains_position(temporal_position):
            raise ValueError(
                f"Temporal position {temporal_position} is outside window "
                f"{self.window_index} [{self.start}, {self.end})"
            )
        return self.frame_indices[temporal_position - self.start]

    def to_dict(self) -> dict[str, object]:
        return {
            "window_index": self.window_index,
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "center": self.center,
            "frame_indices": list(self.frame_indices),
        }


@dataclass(frozen=True)
class WindowFrameAssignment:
    temporal_position: int
    frame_index: int
    window_index: int
    distance_to_center: float

    def to_dict(self) -> dict[str, object]:
        return {
            "temporal_position": self.temporal_position,
            "frame_index": self.frame_index,
            "window_index": self.window_index,
            "distance_to_center": self.distance_to_center,
        }


def validate_window_config(window_size: int, window_stride: int) -> None:
    if window_size < 2:
        raise ValueError("window_size must be at least 2 so overlapping windows are possible")
    if window_stride < 1:
        raise ValueError("window_stride must be at least 1")
    if window_stride >= window_size:
        raise ValueError("window_stride must be smaller than window_size so windows overlap")


def build_inpaint_windows(
    n_frames: int,
    window_size: int = 24,
    window_stride: int = 12,
    frame_indices: Sequence[int] | None = None,
) -> list[InpaintWindow]:
    validate_window_config(window_size, window_stride)
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if frame_indices is None:
        frame_indices = tuple(range(n_frames))
    else:
        frame_indices = tuple(int(index) for index in frame_indices)
    if len(frame_indices) != n_frames:
        raise ValueError("frame_indices length must match n_frames")
    if n_frames == 0:
        return []

    starts = [0]
    if n_frames > window_size:
        while starts[-1] + window_size < n_frames:
            next_start = starts[-1] + window_stride
            if next_start + window_size > n_frames:
                next_start = n_frames - window_size
            if next_start <= starts[-1]:
                break
            starts.append(next_start)

    windows: list[InpaintWindow] = []
    for window_index, start in enumerate(starts):
        end = min(start + window_size, n_frames)
        windows.append(
            InpaintWindow(
                window_index=window_index,
                start=start,
                end=end,
                frame_indices=tuple(frame_indices[start:end]),
            )
        )
    return windows


def build_sequence_inpaint_windows(
    sequence_manifest: SequenceManifest,
    config: IndexConfig,
) -> list[InpaintWindow]:
    frame_indices = [index for index, frame in enumerate(sequence_manifest.rows) if frame.file_exists]
    return build_inpaint_windows(
        n_frames=len(frame_indices),
        window_size=config.inpaint_window_size,
        window_stride=config.inpaint_window_stride,
        frame_indices=frame_indices,
    )


def build_frame_inpaint_windows(
    frames: Sequence[FrameRecord],
    config: IndexConfig,
) -> list[InpaintWindow]:
    frame_indices = [index for index, frame in enumerate(frames) if frame.file_exists]
    return build_inpaint_windows(
        n_frames=len(frame_indices),
        window_size=config.inpaint_window_size,
        window_stride=config.inpaint_window_stride,
        frame_indices=frame_indices,
    )


def select_reconciliation_window(temporal_position: int, windows: Sequence[InpaintWindow]) -> WindowFrameAssignment:
    candidates = [window for window in windows if window.contains_position(temporal_position)]
    if not candidates:
        raise ValueError(f"No inpaint window covers temporal position {temporal_position}")

    selected = min(
        candidates,
        key=lambda window: (
            abs(float(temporal_position) - window.center),
            window.start,
            window.window_index,
        ),
    )
    return WindowFrameAssignment(
        temporal_position=temporal_position,
        frame_index=selected.frame_index_for_position(temporal_position),
        window_index=selected.window_index,
        distance_to_center=abs(float(temporal_position) - selected.center),
    )


def build_reconciliation_plan(
    n_frames: int,
    windows: Sequence[InpaintWindow],
) -> list[WindowFrameAssignment]:
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if n_frames == 0:
        return []
    return [select_reconciliation_window(position, windows) for position in range(n_frames)]


def reconcile_window_predictions(
    windows: Sequence[InpaintWindow],
    predictions_by_window: Mapping[int, Mapping[int, T]],
    n_frames: int | None = None,
) -> list[T]:
    if n_frames is None:
        n_frames = max((window.end for window in windows), default=0)
    plan = build_reconciliation_plan(n_frames, windows)
    reconciled: list[T] = []
    for assignment in plan:
        try:
            window_predictions = predictions_by_window[assignment.window_index]
        except KeyError as exc:
            raise KeyError(f"Missing predictions for window {assignment.window_index}") from exc
        try:
            reconciled.append(window_predictions[assignment.frame_index])
        except KeyError as exc:
            raise KeyError(
                f"Missing prediction for frame {assignment.frame_index} "
                f"in window {assignment.window_index}"
            ) from exc
    return reconciled
