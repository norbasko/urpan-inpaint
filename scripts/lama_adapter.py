#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt advimman/lama prediction outputs to urpan-inpaint's command contract."
    )
    parser.add_argument("--repo", type=Path, required=True, help="Path to the LaMa checkout.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the prepared LaMa checkpoint directory.")
    parser.add_argument("--image-path", type=Path, required=True, help="Input RGB image path from urpan-inpaint.")
    parser.add_argument("--mask-path", type=Path, required=True, help="Input binary mask path from urpan-inpaint.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output image path expected by urpan-inpaint.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to run LaMa.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    script = repo / "bin" / "predict.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing LaMa entrypoint: {script}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Missing LaMa model directory: {args.model_path}")

    work_dir = args.output_path.parent
    input_dir = work_dir / "lama_adapter_input"
    predict_dir = work_dir / "lama_adapter_predict"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    if predict_dir.exists():
        shutil.rmtree(predict_dir)
    input_dir.mkdir(parents=True)
    predict_dir.mkdir(parents=True)

    case_image = input_dir / "case.png"
    case_mask = input_dir / "case_mask.png"
    shutil.copyfile(args.image_path, case_image)
    shutil.copyfile(args.mask_path, case_mask)

    command = [
        args.python,
        str(script),
        f"model.path={args.model_path}",
        f"indir={input_dir}",
        f"outdir={predict_dir}",
    ]
    subprocess.run(command, cwd=repo, check=True)

    prediction = predict_dir / "case_mask.png"
    if not prediction.is_file():
        raise FileNotFoundError(f"LaMa did not write expected prediction: {prediction}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prediction, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
