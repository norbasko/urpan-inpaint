#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt sczhou/ProPainter inference outputs to urpan-inpaint's command contract."
    )
    parser.add_argument("--repo", type=Path, required=True, help="Path to the ProPainter checkout.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing 000000.png input frames.")
    parser.add_argument("--masks-dir", type=Path, required=True, help="Directory containing 000000.png binary masks.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where urpan-inpaint expects outputs.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to run ProPainter.")
    parser.add_argument("--resize-ratio", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=-1)
    parser.add_argument("--width", type=int, default=-1)
    parser.add_argument("--mask-dilation", type=int, default=4)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--subvideo-length", type=int, default=80)
    parser.add_argument("--raft-iter", type=int, default=20)
    parser.add_argument("--fp16", action="store_true", help="Run ProPainter with --fp16.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    script = repo / "inference_propainter.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing ProPainter entrypoint: {script}")

    frame_files = sorted(args.frames_dir.glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No input frames found in {args.frames_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        str(script),
        "--video",
        str(args.frames_dir),
        "--mask",
        str(args.masks_dir),
        "--output",
        str(args.output_dir),
        "--save_frames",
        "--mode",
        "video_inpainting",
        "--resize_ratio",
        str(args.resize_ratio),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--mask_dilation",
        str(args.mask_dilation),
        "--ref_stride",
        str(args.ref_stride),
        "--neighbor_length",
        str(args.neighbor_length),
        "--subvideo_length",
        str(args.subvideo_length),
        "--raft_iter",
        str(args.raft_iter),
    ]
    if args.fp16:
        command.append("--fp16")

    subprocess.run(command, cwd=repo, check=True)

    source_frames = args.output_dir / args.frames_dir.name / "frames"
    if not source_frames.is_dir():
        raise FileNotFoundError(f"ProPainter did not write frame outputs under {source_frames}")

    for index, _frame_path in enumerate(frame_files):
        source = source_frames / f"{index:04d}.png"
        destination = args.output_dir / f"{index:06d}.png"
        if not source.is_file():
            raise FileNotFoundError(f"Missing ProPainter frame output: {source}")
        shutil.copyfile(source, destination)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
