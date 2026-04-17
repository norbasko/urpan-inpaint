from __future__ import annotations

import argparse
import json
from pathlib import Path

from urpan_inpaint.config import IndexConfig
from urpan_inpaint.discovery import run_indexing
from urpan_inpaint.normalization import run_erp_normalization
from urpan_inpaint.projection import run_cubemap_projection
from urpan_inpaint.semantic import run_semantic_parsing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urpan-inpaint",
        description="Production pipeline scaffold for 360 panorama inpainting.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index",
        help="Discover sequences and write per-sequence frame manifests.",
    )
    index_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust"),
        help="Dataset root containing GS* sequence directories.",
    )
    index_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust/inpaint"),
        help="Output root for manifests and later pipeline artifacts.",
    )
    index_parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence ID to process. May be passed multiple times.",
    )
    index_parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=3,
        help="Minimum valid frames required for a sequence to remain eligible.",
    )
    index_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and validate sequences without writing output files.",
    )

    normalize_parser = subparsers.add_parser(
        "normalize-erp",
        help="Load canonical ERP JPEGs, convert to RGB uint8 in memory, and record geometry/checksum metadata.",
    )
    normalize_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust"),
        help="Dataset root containing GS* sequence directories.",
    )
    normalize_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust/inpaint"),
        help="Output root for manifests and later pipeline artifacts.",
    )
    normalize_parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence ID to process. May be passed multiple times.",
    )
    normalize_parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=3,
        help="Minimum valid frames required for a sequence to remain eligible.",
    )
    normalize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate ERP frames without writing output files.",
    )
    normalize_parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA-256 computation on the original JPEG bytes.",
    )

    cubemap_parser = subparsers.add_parser(
        "project-cubemap",
        help="Project ERP panoramas into six cubemap faces with guard-band overlap and optional on-disk caching.",
    )
    cubemap_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust"),
        help="Dataset root containing GS* sequence directories.",
    )
    cubemap_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust/inpaint"),
        help="Output root for manifests and later pipeline artifacts.",
    )
    cubemap_parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence ID to process. May be passed multiple times.",
    )
    cubemap_parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=3,
        help="Minimum valid frames required for a sequence to remain eligible.",
    )
    cubemap_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Project cubemap faces without writing output files.",
    )
    cubemap_parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA-256 computation on the original JPEG bytes.",
    )
    cubemap_parser.add_argument(
        "--cube-face-size",
        type=int,
        default=1536,
        help="Inner cubemap face size in pixels before overlap padding.",
    )
    cubemap_parser.add_argument(
        "--cube-overlap-px",
        type=int,
        default=64,
        help="Guard-band overlap on all cubemap faces in pixels.",
    )
    cubemap_parser.add_argument(
        "--skip-face-cache",
        action="store_true",
        help="Keep projected face tensors in memory only and skip on-disk cubemap cache writes.",
    )

    semantic_parser = subparsers.add_parser(
        "parse-semantic",
        help="Run Mask2Former on cubemap faces and write coarse semantic parsing outputs per face.",
    )
    semantic_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust"),
        help="Dataset root containing GS* sequence directories.",
    )
    semantic_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust/inpaint"),
        help="Output root for manifests and later pipeline artifacts.",
    )
    semantic_parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence ID to process. May be passed multiple times.",
    )
    semantic_parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=3,
        help="Minimum valid frames required for a sequence to remain eligible.",
    )
    semantic_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run semantic parsing without writing output files.",
    )
    semantic_parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA-256 computation on the original JPEG bytes.",
    )
    semantic_parser.add_argument(
        "--cube-face-size",
        type=int,
        default=1536,
        help="Inner cubemap face size in pixels before overlap padding when projection cache is missing.",
    )
    semantic_parser.add_argument(
        "--cube-overlap-px",
        type=int,
        default=64,
        help="Guard-band overlap on all cubemap faces when projection cache is missing.",
    )
    semantic_parser.add_argument(
        "--skip-face-cache",
        action="store_true",
        help="Keep projected face tensors in memory only if cubemap cache is missing.",
    )
    semantic_parser.add_argument(
        "--semantic-model-id",
        default="facebook/mask2former-swin-tiny-cityscapes-semantic",
        help="Mask2Former checkpoint identifier or local path.",
    )
    semantic_parser.add_argument(
        "--semantic-device",
        default="auto",
        help="Torch device for inference. Use auto, cpu, cuda, or mps.",
    )
    semantic_parser.add_argument(
        "--semantic-local-files-only",
        action="store_true",
        help="Restrict model loading to locally cached files.",
    )
    semantic_parser.add_argument(
        "--save-semantic-logits",
        action="store_true",
        help="Write per-class semantic score tensors into each face output.",
    )
    semantic_parser.add_argument(
        "--skip-confidence",
        action="store_true",
        help="Skip confidence-map output even when it can be derived from semantic scores.",
    )
    semantic_parser.add_argument(
        "--skip-panoptic",
        action="store_true",
        help="Skip panoptic post-processing even if the checkpoint supports it.",
    )

    return parser


def handle_index(args: argparse.Namespace) -> int:
    config = IndexConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        min_valid_frames=args.min_valid_frames,
        dry_run=args.dry_run,
    )
    manifests = run_indexing(config, sequence_ids=args.sequence or None)
    summary = {
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "dry_run": config.dry_run,
        "sequence_count": len(manifests),
        "ready_sequences": sum(1 for item in manifests if item.status == "ready"),
        "failed_sequences": sum(1 for item in manifests if item.status != "ready"),
        "sequences": [item.to_summary_dict() for item in manifests],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_sequences"] == 0 else 1


def handle_normalize_erp(args: argparse.Namespace) -> int:
    config = IndexConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        min_valid_frames=args.min_valid_frames,
        dry_run=args.dry_run,
        compute_checksums=not args.skip_checksum,
    )
    manifests = run_erp_normalization(config, sequence_ids=args.sequence or None)
    summary = {
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "dry_run": config.dry_run,
        "compute_checksums": config.compute_checksums,
        "sequence_count": len(manifests),
        "normalized_sequences": sum(1 for item in manifests if item.status == "ready"),
        "failed_sequences": sum(1 for item in manifests if item.status != "ready"),
        "normalized_frames": sum(
            sum(1 for row in item.rows if row.erp_normalization_status == "normalized")
            for item in manifests
        ),
        "failed_frames": sum(
            sum(1 for row in item.rows if row.erp_normalization_status == "failed")
            for item in manifests
        ),
        "sequences": [item.to_summary_dict() for item in manifests],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_sequences"] == 0 else 1


def handle_project_cubemap(args: argparse.Namespace) -> int:
    config = IndexConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        min_valid_frames=args.min_valid_frames,
        dry_run=args.dry_run,
        compute_checksums=not args.skip_checksum,
        cube_face_size=args.cube_face_size,
        cube_overlap_px=args.cube_overlap_px,
        cache_cubemap_faces=not args.skip_face_cache,
    )
    manifests = run_cubemap_projection(config, sequence_ids=args.sequence or None)
    summary = {
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "dry_run": config.dry_run,
        "compute_checksums": config.compute_checksums,
        "cube_face_size": config.cube_face_size,
        "cube_overlap_px": config.cube_overlap_px,
        "cache_cubemap_faces": config.cache_cubemap_faces,
        "sequence_count": len(manifests),
        "projected_sequences": sum(1 for item in manifests if item.status == "ready"),
        "failed_sequences": sum(1 for item in manifests if item.status != "ready"),
        "projected_frames": sum(
            sum(1 for row in item.rows if row.cubemap_projection_status == "projected")
            for item in manifests
        ),
        "failed_frames": sum(
            sum(1 for row in item.rows if row.cubemap_projection_status == "failed")
            for item in manifests
        ),
        "sequences": [item.to_summary_dict() for item in manifests],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_sequences"] == 0 else 1


def handle_parse_semantic(args: argparse.Namespace) -> int:
    config = IndexConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        min_valid_frames=args.min_valid_frames,
        dry_run=args.dry_run,
        compute_checksums=not args.skip_checksum,
        cube_face_size=args.cube_face_size,
        cube_overlap_px=args.cube_overlap_px,
        cache_cubemap_faces=not args.skip_face_cache,
        semantic_model_id=args.semantic_model_id,
        semantic_device=args.semantic_device,
        semantic_local_files_only=args.semantic_local_files_only,
        semantic_save_logits=args.save_semantic_logits,
        semantic_save_confidence=not args.skip_confidence,
        semantic_attempt_panoptic=not args.skip_panoptic,
    )
    manifests = run_semantic_parsing(config, sequence_ids=args.sequence or None)
    summary = {
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "dry_run": config.dry_run,
        "compute_checksums": config.compute_checksums,
        "cube_face_size": config.cube_face_size,
        "cube_overlap_px": config.cube_overlap_px,
        "cache_cubemap_faces": config.cache_cubemap_faces,
        "semantic_model_id": config.semantic_model_id,
        "semantic_device": config.semantic_device,
        "semantic_local_files_only": config.semantic_local_files_only,
        "semantic_save_logits": config.semantic_save_logits,
        "semantic_save_confidence": config.semantic_save_confidence,
        "semantic_attempt_panoptic": config.semantic_attempt_panoptic,
        "sequence_count": len(manifests),
        "parsed_sequences": sum(1 for item in manifests if item.status == "ready"),
        "failed_sequences": sum(1 for item in manifests if item.status != "ready"),
        "parsed_frames": sum(
            sum(1 for row in item.rows if row.semantic_parse_status == "parsed")
            for item in manifests
        ),
        "failed_frames": sum(
            sum(1 for row in item.rows if row.semantic_parse_status == "failed")
            for item in manifests
        ),
        "sequences": [item.to_summary_dict() for item in manifests],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_sequences"] == 0 else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "index":
        return handle_index(args)
    if args.command == "normalize-erp":
        return handle_normalize_erp(args)
    if args.command == "project-cubemap":
        return handle_project_cubemap(args)
    if args.command == "parse-semantic":
        return handle_parse_semantic(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
