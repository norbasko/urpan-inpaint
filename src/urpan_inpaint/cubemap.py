from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from urpan_inpaint.erp import ERPFrame


FACE_ORDER = ("front", "right", "back", "left", "up", "down")


@dataclass(frozen=True)
class CubemapFace:
    name: str
    rgb: np.ndarray
    erp_x: np.ndarray
    erp_y: np.ndarray
    inner_mask: np.ndarray


@dataclass(frozen=True)
class CubemapProjection:
    erp_width: int
    erp_height: int
    face_size: int
    overlap_px: int
    total_face_size: int
    faces: dict[str, CubemapFace]


def _face_coordinate_grid(face_size: int, overlap_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_face_size = face_size + (2 * overlap_px)
    coords = (((np.arange(total_face_size, dtype=np.float32) + 0.5) - overlap_px) / face_size) * 2.0 - 1.0
    u, v = np.meshgrid(coords, coords)
    inner_axis = (np.arange(total_face_size) >= overlap_px) & (np.arange(total_face_size) < overlap_px + face_size)
    inner_mask = np.outer(inner_axis, inner_axis)
    return u, v, inner_mask


def _face_directions(face_name: str, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if face_name == "front":
        x, y, z = u, -v, np.ones_like(u)
    elif face_name == "right":
        x, y, z = np.ones_like(u), -v, -u
    elif face_name == "back":
        x, y, z = -u, -v, -np.ones_like(u)
    elif face_name == "left":
        x, y, z = -np.ones_like(u), -v, u
    elif face_name == "up":
        x, y, z = u, np.ones_like(u), v
    elif face_name == "down":
        x, y, z = u, -np.ones_like(u), -v
    else:
        raise ValueError(f"Unsupported cubemap face: {face_name}")

    norm = np.sqrt((x * x) + (y * y) + (z * z))
    return x / norm, y / norm, z / norm


def _direction_to_erp_coords(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    erp_width: int,
    erp_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y, -1.0, 1.0))
    erp_x = ((lon / (2.0 * np.pi)) + 0.5) * erp_width - 0.5
    erp_y = (0.5 - (lat / np.pi)) * erp_height - 0.5
    return erp_x.astype(np.float32), erp_y.astype(np.float32)


def _sample_bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray, wrap_x: bool) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if wrap_x:
        x = np.mod(x, width)
    else:
        x = np.clip(x, 0.0, width - 1.0)
    y = np.clip(y, 0.0, height - 1.0)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = (x0 + 1) % width if wrap_x else np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    wx = wx[..., None]
    wy = wy[..., None]

    top_left = image[y0, x0].astype(np.float32)
    top_right = image[y0, x1].astype(np.float32)
    bottom_left = image[y1, x0].astype(np.float32)
    bottom_right = image[y1, x1].astype(np.float32)

    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.clip(np.rint(sampled), 0, 255).astype(np.uint8)


def erp_to_cubemap(erp: ERPFrame, face_size: int, overlap_px: int) -> CubemapProjection:
    u, v, inner_mask = _face_coordinate_grid(face_size, overlap_px)
    total_face_size = face_size + (2 * overlap_px)
    faces: dict[str, CubemapFace] = {}

    for face_name in FACE_ORDER:
        dir_x, dir_y, dir_z = _face_directions(face_name, u, v)
        erp_x, erp_y = _direction_to_erp_coords(dir_x, dir_y, dir_z, erp.width, erp.height)
        rgb = _sample_bilinear(erp.rgb, erp_x, erp_y, wrap_x=True)
        faces[face_name] = CubemapFace(
            name=face_name,
            rgb=rgb,
            erp_x=erp_x,
            erp_y=erp_y,
            inner_mask=inner_mask.copy(),
        )

    return CubemapProjection(
        erp_width=erp.width,
        erp_height=erp.height,
        face_size=face_size,
        overlap_px=overlap_px,
        total_face_size=total_face_size,
        faces=faces,
    )


def _erp_pixel_directions(erp_width: int, erp_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = (((np.arange(erp_width, dtype=np.float32) + 0.5) / erp_width) - 0.5) * (2.0 * np.pi)
    lat = (0.5 - ((np.arange(erp_height, dtype=np.float32) + 0.5) / erp_height)) * np.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat_grid)
    x = cos_lat * np.sin(lon_grid)
    y = np.sin(lat_grid)
    z = cos_lat * np.cos(lon_grid)
    return x, y, z


def cubemap_to_erp(
    projection: CubemapProjection,
    erp_width: int | None = None,
    erp_height: int | None = None,
) -> np.ndarray:
    erp_width = projection.erp_width if erp_width is None else erp_width
    erp_height = projection.erp_height if erp_height is None else erp_height
    x, y, z = _erp_pixel_directions(erp_width, erp_height)

    abs_x = np.abs(x)
    abs_y = np.abs(y)
    abs_z = np.abs(z)

    result = np.empty((erp_height, erp_width, 3), dtype=np.uint8)

    face_masks = {
        "right": (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0),
        "left": (abs_x >= abs_y) & (abs_x >= abs_z) & (x <= 0),
        "up": (abs_y > abs_x) & (abs_y >= abs_z) & (y > 0),
        "down": (abs_y > abs_x) & (abs_y >= abs_z) & (y <= 0),
        "front": (abs_z > abs_x) & (abs_z > abs_y) & (z > 0),
        "back": (abs_z > abs_x) & (abs_z > abs_y) & (z <= 0),
    }

    for face_name, mask in face_masks.items():
        if not np.any(mask):
            continue

        if face_name == "front":
            u = x[mask] / abs_z[mask]
            v = -y[mask] / abs_z[mask]
        elif face_name == "right":
            u = -z[mask] / abs_x[mask]
            v = -y[mask] / abs_x[mask]
        elif face_name == "back":
            u = -x[mask] / abs_z[mask]
            v = -y[mask] / abs_z[mask]
        elif face_name == "left":
            u = z[mask] / abs_x[mask]
            v = -y[mask] / abs_x[mask]
        elif face_name == "up":
            u = x[mask] / abs_y[mask]
            v = z[mask] / abs_y[mask]
        elif face_name == "down":
            u = x[mask] / abs_y[mask]
            v = -z[mask] / abs_y[mask]
        else:
            raise ValueError(f"Unsupported cubemap face: {face_name}")

        face_x = ((u + 1.0) * projection.face_size * 0.5) + projection.overlap_px - 0.5
        face_y = ((v + 1.0) * projection.face_size * 0.5) + projection.overlap_px - 0.5
        sampled = _sample_bilinear(projection.faces[face_name].rgb, face_x, face_y, wrap_x=False)
        result[mask] = sampled

    return result


def save_cubemap_projection(projection: CubemapProjection, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = cache_dir / "projection.json"
    metadata = {
        "erp_width": projection.erp_width,
        "erp_height": projection.erp_height,
        "face_size": projection.face_size,
        "overlap_px": projection.overlap_px,
        "total_face_size": projection.total_face_size,
        "face_order": list(FACE_ORDER),
        "face_cache_format": "npz",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for face_name, face in projection.faces.items():
        np.savez_compressed(
            cache_dir / f"{face_name}.npz",
            rgb=face.rgb,
            erp_x=face.erp_x,
            erp_y=face.erp_y,
            inner_mask=face.inner_mask.astype(np.uint8),
        )

    return metadata_path


def load_cubemap_projection(cache_dir: Path) -> CubemapProjection:
    metadata = json.loads((cache_dir / "projection.json").read_text(encoding="utf-8"))
    faces: dict[str, CubemapFace] = {}
    for face_name in metadata["face_order"]:
        with np.load(cache_dir / f"{face_name}.npz") as payload:
            faces[face_name] = CubemapFace(
                name=face_name,
                rgb=payload["rgb"],
                erp_x=payload["erp_x"],
                erp_y=payload["erp_y"],
                inner_mask=payload["inner_mask"].astype(bool),
            )

    return CubemapProjection(
        erp_width=int(metadata["erp_width"]),
        erp_height=int(metadata["erp_height"]),
        face_size=int(metadata["face_size"]),
        overlap_px=int(metadata["overlap_px"]),
        total_face_size=int(metadata["total_face_size"]),
        faces=faces,
    )
