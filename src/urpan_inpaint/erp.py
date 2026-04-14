from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ERPFrame:
    path: Path
    width: int
    height: int
    channels: int
    dtype: str
    source_mode: str
    file_sha256: str
    rgb: np.ndarray


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_erp_rgb(path: Path, compute_checksum: bool = True) -> ERPFrame:
    file_sha256 = compute_file_sha256(path) if compute_checksum else ""
    with Image.open(path) as image:
        width, height = image.size
        source_mode = image.mode
        rgb_image = image.convert("RGB")
        rgb = np.asarray(rgb_image, dtype=np.uint8)

    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 ERP image, got {rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 ERP image, got shape {rgb.shape}")
    if rgb.shape[1] != width or rgb.shape[0] != height:
        raise ValueError(
            f"ERP dimensions changed during normalization: original {(width, height)} vs array {rgb.shape[1::-1]}"
        )

    return ERPFrame(
        path=path,
        width=width,
        height=height,
        channels=rgb.shape[2],
        dtype=str(rgb.dtype),
        source_mode=source_mode,
        file_sha256=file_sha256,
        rgb=rgb,
    )


def circular_pad_erp_horizontally(rgb: np.ndarray, left: int = 0, right: int = 0) -> np.ndarray:
    if rgb.ndim != 3:
        raise ValueError(f"Expected HxWxC array, got {rgb.shape}")
    if left < 0 or right < 0:
        raise ValueError("Circular ERP padding must be non-negative")
    if left == 0 and right == 0:
        return rgb.copy()
    return np.pad(rgb, ((0, 0), (left, right), (0, 0)), mode="wrap")


def circular_crop_erp(rgb: np.ndarray, x_start: int, width: int) -> np.ndarray:
    if rgb.ndim != 3:
        raise ValueError(f"Expected HxWxC array, got {rgb.shape}")
    if width < 1:
        raise ValueError("Circular crop width must be positive")
    erp_width = rgb.shape[1]
    if erp_width == 0:
        raise ValueError("ERP width must be positive")

    x_indices = (np.arange(width, dtype=np.int64) + x_start) % erp_width
    return rgb[:, x_indices, :]
