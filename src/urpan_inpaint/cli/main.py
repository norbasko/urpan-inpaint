from __future__ import annotations

import argparse
import json
from pathlib import Path

from urpan_inpaint.config import DEFAULT_GROUNDING_DINO_PROMPTS, DEFAULT_SAM2_SEMANTIC_PROMPT_CLASSES
from urpan_inpaint.config import IndexConfig
from urpan_inpaint.detection import run_grounding_detection
from urpan_inpaint.discovery import run_indexing
from urpan_inpaint.fusion import run_mask_fusion
from urpan_inpaint.inpainting import run_propainter_inpainting
from urpan_inpaint.normalization import run_erp_normalization
from urpan_inpaint.projection import run_cubemap_projection
from urpan_inpaint.refinement import run_sam2_refinement
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

    grounding_parser = subparsers.add_parser(
        "detect-dynamic",
        help="Run Grounding DINO on cubemap faces to recover dynamic-object detections missed by semantic parsing.",
    )
    grounding_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust"),
        help="Dataset root containing GS* sequence directories.",
    )
    grounding_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust/inpaint"),
        help="Output root for manifests and later pipeline artifacts.",
    )
    grounding_parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence ID to process. May be passed multiple times.",
    )
    grounding_parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=3,
        help="Minimum valid frames required for a sequence to remain eligible.",
    )
    grounding_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run Grounding DINO without writing output files.",
    )
    grounding_parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA-256 computation on the original JPEG bytes.",
    )
    grounding_parser.add_argument(
        "--cube-face-size",
        type=int,
        default=1536,
        help="Inner cubemap face size in pixels before overlap padding when projection cache is missing.",
    )
    grounding_parser.add_argument(
        "--cube-overlap-px",
        type=int,
        default=64,
        help="Guard-band overlap on all cubemap faces when projection cache is missing.",
    )
    grounding_parser.add_argument(
        "--skip-face-cache",
        action="store_true",
        help="Keep projected face tensors in memory only if cubemap cache is missing.",
    )
    grounding_parser.add_argument(
        "--grounding-model-id",
        default="IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO checkpoint identifier or local path.",
    )
    grounding_parser.add_argument(
        "--grounding-device",
        default="auto",
        help="Torch device for inference. Use auto, cpu, cuda, or mps.",
    )
    grounding_parser.add_argument(
        "--grounding-local-files-only",
        action="store_true",
        help="Restrict model loading to locally cached files.",
    )
    grounding_parser.add_argument(
        "--grounding-prompt",
        action="append",
        default=[],
        help="Candidate prompt term. May be passed multiple times. Defaults to the built-in road-user prompt set.",
    )
    grounding_parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Drop detections below this Grounding DINO box confidence threshold.",
    )
    grounding_parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Phrase-token confidence threshold used by Grounding DINO label extraction.",
    )
    grounding_parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.7,
        help="IoU threshold for class-aware NMS after Grounding DINO post-processing.",
    )

    refine_parser = subparsers.add_parser(
        "refine-masks",
        help="Use SAM 2 to refine coarse semantic regions, Grounding DINO boxes, down-face roof prompts, and stable temporal priors.",
    )
    refine_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust"),
        help="Dataset root containing GS* sequence directories.",
    )
    refine_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust/inpaint"),
        help="Output root for manifests and later pipeline artifacts.",
    )
    refine_parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence ID to process. May be passed multiple times.",
    )
    refine_parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=3,
        help="Minimum valid frames required for a sequence to remain eligible.",
    )
    refine_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run SAM 2 refinement without writing output files.",
    )
    refine_parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA-256 computation on the original JPEG bytes.",
    )
    refine_parser.add_argument(
        "--cube-face-size",
        type=int,
        default=1536,
        help="Inner cubemap face size in pixels before overlap padding when projection cache is missing.",
    )
    refine_parser.add_argument(
        "--cube-overlap-px",
        type=int,
        default=64,
        help="Guard-band overlap on all cubemap faces when projection cache is missing.",
    )
    refine_parser.add_argument(
        "--skip-face-cache",
        action="store_true",
        help="Keep projected face tensors in memory only if cubemap cache is missing.",
    )
    refine_parser.add_argument(
        "--sam2-model-id",
        default="facebook/sam2.1-hiera-tiny",
        help="SAM 2 checkpoint identifier or local path.",
    )
    refine_parser.add_argument(
        "--sam2-device",
        default="auto",
        help="Torch device for inference. Use auto, cpu, cuda, or mps.",
    )
    refine_parser.add_argument(
        "--sam2-local-files-only",
        action="store_true",
        help="Restrict SAM 2 model loading to locally cached files.",
    )
    refine_parser.add_argument(
        "--sam2-mask-threshold",
        type=float,
        default=0.0,
        help="Mask binarization threshold passed to SAM 2 post-processing.",
    )
    refine_parser.add_argument(
        "--sam2-min-mask-area-px",
        type=int,
        default=16,
        help="Drop refined masks with fewer foreground pixels than this.",
    )
    refine_parser.add_argument(
        "--skip-grounding-prompts",
        action="store_true",
        help="Do not use Grounding DINO boxes as SAM 2 prompts.",
    )
    refine_parser.add_argument(
        "--skip-semantic-prompts",
        action="store_true",
        help="Do not use semantic region boxes/points as SAM 2 prompts.",
    )
    refine_parser.add_argument(
        "--skip-roof-prompt",
        action="store_true",
        help="Do not add the down-face nadir roof prompt.",
    )
    refine_parser.add_argument(
        "--sam2-semantic-class",
        action="append",
        default=[],
        help="Semantic class to refine with SAM 2. May be passed multiple times.",
    )
    refine_parser.add_argument(
        "--sam2-roof-box-fraction",
        type=float,
        default=0.55,
        help="Base fraction of the down-face width/height covered by the nadir roof prior.",
    )
    refine_parser.add_argument(
        "--sam2-roof-prior-margin-fraction",
        type=float,
        default=0.15,
        help="Additional radial fraction used to expand the roof prompt/support region.",
    )
    refine_parser.add_argument(
        "--sam2-roof-temporal-window",
        type=int,
        default=1,
        help="Neighboring-frame radius used for temporal roof-mask median regularization and fallback.",
    )
    refine_parser.add_argument(
        "--sam2-roof-temporal-disagreement-iou-threshold",
        type=float,
        default=0.4,
        help="Flag roof masks below this IoU with the neighboring-frame median and keep current evidence.",
    )
    refine_parser.add_argument(
        "--sky-mask-top-seed-fraction",
        type=float,
        default=0.12,
        help="Top face fraction used to seed conservative side-face sky connectivity.",
    )
    refine_parser.add_argument(
        "--sky-mask-sam2-boundary-margin-px",
        type=int,
        default=3,
        help="Pixel radius around Mask2Former sky boundaries where SAM 2 may adjust skyline edges.",
    )
    refine_parser.add_argument(
        "--sky-mask-obstacle-dilation-px",
        type=int,
        default=1,
        help="Dilate semantic non-sky blockers before suppressing sky transparency.",
    )
    refine_parser.add_argument(
        "--sky-mask-erp-smoothing-iterations",
        type=int,
        default=1,
        help="Conservative seam-aware ERP sky-mask smoothing iterations.",
    )
    refine_parser.add_argument(
        "--disable-temporal-propagation",
        action="store_true",
        help="Disable adjacent-frame prior-mask prompting.",
    )
    refine_parser.add_argument(
        "--sam2-temporal-iou-threshold",
        type=float,
        default=0.45,
        help="Minimum box IoU required to use adjacent-frame prior masks for non-roof prompts.",
    )
    refine_parser.add_argument(
        "--sam2-temporal-area-ratio-min",
        type=float,
        default=0.5,
        help="Minimum current/prior box area ratio for temporal prior stability.",
    )
    refine_parser.add_argument(
        "--sam2-temporal-area-ratio-max",
        type=float,
        default=2.0,
        help="Maximum current/prior box area ratio for temporal prior stability.",
    )
    refine_parser.add_argument(
        "--sam2-temporal-max-gap",
        type=int,
        default=1,
        help="Maximum adjacent-frame gap allowed for temporal prior masks.",
    )

    fusion_parser = subparsers.add_parser(
        "fuse-masks",
        help="Fuse parser, Grounding DINO/SAM 2, roof, and sky masks into final dynamic/roof/sky/inpaint masks.",
    )
    fusion_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust"),
        help="Dataset root containing GS* sequence directories.",
    )
    fusion_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust/inpaint"),
        help="Output root for manifests and pipeline artifacts.",
    )
    fusion_parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence ID to process. May be passed multiple times.",
    )
    fusion_parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=3,
        help="Minimum valid frames required for a sequence to remain eligible.",
    )
    fusion_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fuse masks without writing output files.",
    )
    fusion_parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA-256 computation if cubemap projection cache is missing.",
    )
    fusion_parser.add_argument(
        "--cube-face-size",
        type=int,
        default=1536,
        help="Inner cubemap face size in pixels when projection cache is missing.",
    )
    fusion_parser.add_argument(
        "--cube-overlap-px",
        type=int,
        default=64,
        help="Guard-band overlap on all cubemap faces when projection cache is missing.",
    )
    fusion_parser.add_argument(
        "--skip-face-cache",
        action="store_true",
        help="Keep projected face tensors in memory only if cubemap cache is missing.",
    )
    fusion_parser.add_argument(
        "--dyn-min-component-area-px",
        type=int,
        default=64,
        help="Drop dynamic components smaller than this after hole filling.",
    )
    fusion_parser.add_argument(
        "--roof-min-component-area-px",
        type=int,
        default=256,
        help="Drop roof components smaller than this after hole filling.",
    )
    fusion_parser.add_argument(
        "--dyn-dilate-px",
        type=int,
        default=3,
        help="Dynamic-mask edge dilation radius in pixels.",
    )
    fusion_parser.add_argument(
        "--roof-dilate-px",
        type=int,
        default=5,
        help="Roof-mask edge dilation radius in pixels.",
    )
    fusion_parser.add_argument(
        "--dyn-erode-after-dilate-px",
        type=int,
        default=0,
        help="Optional dynamic-mask erosion after dilation when expansion is excessive.",
    )
    fusion_parser.add_argument(
        "--roof-erode-after-dilate-px",
        type=int,
        default=0,
        help="Optional roof-mask erosion after dilation when expansion is excessive.",
    )
    fusion_parser.add_argument(
        "--sky-mask-erp-smoothing-iterations",
        type=int,
        default=1,
        help="Seam-aware smoothing iterations for the refined sky mask.",
    )

    inpaint_parser = subparsers.add_parser(
        "inpaint-sequence",
        help="Run ProPainter on each cubemap face video stream using fused INPAINT masks.",
    )
    inpaint_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust"),
        help="Dataset root containing GS* sequence directories.",
    )
    inpaint_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/vision/data/kaust/inpaint"),
        help="Output root for manifests and pipeline artifacts.",
    )
    inpaint_parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence ID to process. May be passed multiple times.",
    )
    inpaint_parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=3,
        help="Minimum valid frames required for a sequence to remain eligible.",
    )
    inpaint_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan ProPainter inpainting without writing output files.",
    )
    inpaint_parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA-256 computation if cubemap projection cache is missing.",
    )
    inpaint_parser.add_argument(
        "--cube-face-size",
        type=int,
        default=1536,
        help="Inner cubemap face size in pixels when projection cache is missing.",
    )
    inpaint_parser.add_argument(
        "--cube-overlap-px",
        type=int,
        default=64,
        help="Guard-band overlap on all cubemap faces when projection cache is missing.",
    )
    inpaint_parser.add_argument(
        "--skip-face-cache",
        action="store_true",
        help="Keep projected face tensors in memory only if cubemap cache is missing.",
    )
    inpaint_parser.add_argument(
        "--inpaint-window-size",
        type=int,
        default=24,
        help="Temporal ProPainter window size.",
    )
    inpaint_parser.add_argument(
        "--inpaint-window-stride",
        type=int,
        default=12,
        help="Temporal ProPainter window stride. Must be smaller than the window size.",
    )
    inpaint_parser.add_argument(
        "--propainter-model-id",
        default="ProPainter",
        help="ProPainter implementation identifier recorded in manifests.",
    )
    inpaint_parser.add_argument(
        "--propainter-command",
        default="",
        help=(
            "External ProPainter command template. It may use {frames_dir}, {masks_dir}, "
            "{output_dir}, {face_name}, and {device}; it must write 000000.png, 000001.png, ..."
        ),
    )
    inpaint_parser.add_argument(
        "--propainter-device",
        default="auto",
        help="Device token passed into the ProPainter command template.",
    )
    inpaint_parser.add_argument(
        "--propainter-chunk-size",
        type=int,
        default=8,
        help="Maximum frames sent to one ProPainter inference call.",
    )
    inpaint_parser.add_argument(
        "--propainter-face-feather-px",
        type=int,
        default=64,
        help="Feather width used when reprojecting independently inpainted faces back to ERP.",
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


def handle_detect_dynamic(args: argparse.Namespace) -> int:
    prompts = tuple(args.grounding_prompt) if args.grounding_prompt else DEFAULT_GROUNDING_DINO_PROMPTS
    config = IndexConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        min_valid_frames=args.min_valid_frames,
        dry_run=args.dry_run,
        compute_checksums=not args.skip_checksum,
        cube_face_size=args.cube_face_size,
        cube_overlap_px=args.cube_overlap_px,
        cache_cubemap_faces=not args.skip_face_cache,
        grounding_model_id=args.grounding_model_id,
        grounding_device=args.grounding_device,
        grounding_local_files_only=args.grounding_local_files_only,
        grounding_prompts=prompts,
        grounding_box_threshold=args.box_threshold,
        grounding_text_threshold=args.text_threshold,
        grounding_nms_iou_threshold=args.nms_iou_threshold,
    )
    manifests = run_grounding_detection(config, sequence_ids=args.sequence or None)
    summary = {
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "dry_run": config.dry_run,
        "compute_checksums": config.compute_checksums,
        "cube_face_size": config.cube_face_size,
        "cube_overlap_px": config.cube_overlap_px,
        "cache_cubemap_faces": config.cache_cubemap_faces,
        "grounding_model_id": config.grounding_model_id,
        "grounding_device": config.grounding_device,
        "grounding_local_files_only": config.grounding_local_files_only,
        "grounding_prompts": list(config.grounding_prompts),
        "grounding_box_threshold": config.grounding_box_threshold,
        "grounding_text_threshold": config.grounding_text_threshold,
        "grounding_nms_iou_threshold": config.grounding_nms_iou_threshold,
        "sequence_count": len(manifests),
        "detected_sequences": sum(1 for item in manifests if item.status == "ready"),
        "failed_sequences": sum(1 for item in manifests if item.status != "ready"),
        "detected_frames": sum(
            sum(1 for row in item.rows if row.grounding_detect_status == "detected")
            for item in manifests
        ),
        "failed_frames": sum(
            sum(1 for row in item.rows if row.grounding_detect_status == "failed")
            for item in manifests
        ),
        "total_boxes": sum(
            sum((row.grounding_box_count or 0) for row in item.rows if row.grounding_detect_status == "detected")
            for item in manifests
        ),
        "sequences": [item.to_summary_dict() for item in manifests],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_sequences"] == 0 else 1


def handle_refine_masks(args: argparse.Namespace) -> int:
    semantic_classes = (
        tuple(args.sam2_semantic_class)
        if args.sam2_semantic_class
        else DEFAULT_SAM2_SEMANTIC_PROMPT_CLASSES
    )
    config = IndexConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        min_valid_frames=args.min_valid_frames,
        dry_run=args.dry_run,
        compute_checksums=not args.skip_checksum,
        cube_face_size=args.cube_face_size,
        cube_overlap_px=args.cube_overlap_px,
        cache_cubemap_faces=not args.skip_face_cache,
        sam2_model_id=args.sam2_model_id,
        sam2_device=args.sam2_device,
        sam2_local_files_only=args.sam2_local_files_only,
        sam2_mask_threshold=args.sam2_mask_threshold,
        sam2_min_mask_area_px=args.sam2_min_mask_area_px,
        sam2_refine_grounding=not args.skip_grounding_prompts,
        sam2_refine_semantic=not args.skip_semantic_prompts,
        sam2_refine_roof=not args.skip_roof_prompt,
        sam2_semantic_prompt_classes=semantic_classes,
        sam2_roof_box_fraction=args.sam2_roof_box_fraction,
        sam2_roof_prior_margin_fraction=args.sam2_roof_prior_margin_fraction,
        sam2_roof_temporal_window=args.sam2_roof_temporal_window,
        sam2_roof_temporal_disagreement_iou_threshold=args.sam2_roof_temporal_disagreement_iou_threshold,
        sky_mask_top_seed_fraction=args.sky_mask_top_seed_fraction,
        sky_mask_sam2_boundary_margin_px=args.sky_mask_sam2_boundary_margin_px,
        sky_mask_obstacle_dilation_px=args.sky_mask_obstacle_dilation_px,
        sky_mask_erp_smoothing_iterations=args.sky_mask_erp_smoothing_iterations,
        sam2_temporal_propagation=not args.disable_temporal_propagation,
        sam2_temporal_iou_threshold=args.sam2_temporal_iou_threshold,
        sam2_temporal_area_ratio_min=args.sam2_temporal_area_ratio_min,
        sam2_temporal_area_ratio_max=args.sam2_temporal_area_ratio_max,
        sam2_temporal_max_gap=args.sam2_temporal_max_gap,
    )
    manifests = run_sam2_refinement(config, sequence_ids=args.sequence or None)
    summary = {
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "dry_run": config.dry_run,
        "compute_checksums": config.compute_checksums,
        "cube_face_size": config.cube_face_size,
        "cube_overlap_px": config.cube_overlap_px,
        "cache_cubemap_faces": config.cache_cubemap_faces,
        "sam2_model_id": config.sam2_model_id,
        "sam2_device": config.sam2_device,
        "sam2_local_files_only": config.sam2_local_files_only,
        "sam2_mask_threshold": config.sam2_mask_threshold,
        "sam2_min_mask_area_px": config.sam2_min_mask_area_px,
        "sam2_refine_grounding": config.sam2_refine_grounding,
        "sam2_refine_semantic": config.sam2_refine_semantic,
        "sam2_refine_roof": config.sam2_refine_roof,
        "sam2_semantic_prompt_classes": list(config.sam2_semantic_prompt_classes),
        "sam2_roof_box_fraction": config.sam2_roof_box_fraction,
        "sam2_roof_prior_margin_fraction": config.sam2_roof_prior_margin_fraction,
        "sam2_roof_temporal_window": config.sam2_roof_temporal_window,
        "sam2_roof_temporal_disagreement_iou_threshold": (
            config.sam2_roof_temporal_disagreement_iou_threshold
        ),
        "sky_mask_top_seed_fraction": config.sky_mask_top_seed_fraction,
        "sky_mask_sam2_boundary_margin_px": config.sky_mask_sam2_boundary_margin_px,
        "sky_mask_obstacle_dilation_px": config.sky_mask_obstacle_dilation_px,
        "sky_mask_erp_smoothing_iterations": config.sky_mask_erp_smoothing_iterations,
        "sam2_temporal_propagation": config.sam2_temporal_propagation,
        "sam2_temporal_iou_threshold": config.sam2_temporal_iou_threshold,
        "sam2_temporal_area_ratio_min": config.sam2_temporal_area_ratio_min,
        "sam2_temporal_area_ratio_max": config.sam2_temporal_area_ratio_max,
        "sam2_temporal_max_gap": config.sam2_temporal_max_gap,
        "sequence_count": len(manifests),
        "refined_sequences": sum(1 for item in manifests if item.status == "ready"),
        "failed_sequences": sum(1 for item in manifests if item.status != "ready"),
        "refined_frames": sum(
            sum(1 for row in item.rows if row.sam2_refine_status == "refined")
            for item in manifests
        ),
        "failed_frames": sum(
            sum(1 for row in item.rows if row.sam2_refine_status == "failed")
            for item in manifests
        ),
        "total_masks": sum(
            sum((row.sam2_mask_count or 0) for row in item.rows if row.sam2_refine_status == "refined")
            for item in manifests
        ),
        "total_temporal_priors": sum(
            sum((row.sam2_temporal_prior_count or 0) for row in item.rows if row.sam2_refine_status == "refined")
            for item in manifests
        ),
        "roof_masks": sum(
            sum(1 for row in item.rows if row.roof_mask_status in {"generated", "fallback"})
            for item in manifests
        ),
        "roof_fallbacks": sum(
            sum(1 for row in item.rows if row.roof_mask_status == "fallback")
            for item in manifests
        ),
        "roof_temporal_disagreements": sum(
            sum(1 for row in item.rows if row.roof_mask_temporal_disagreement)
            for item in manifests
        ),
        "sky_masks": sum(
            sum(1 for row in item.rows if row.sky_mask_status == "generated")
            for item in manifests
        ),
        "sky_empty_frames": sum(
            sum(1 for row in item.rows if row.sky_mask_status == "empty")
            for item in manifests
        ),
        "sequences": [item.to_summary_dict() for item in manifests],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_sequences"] == 0 else 1


def handle_fuse_masks(args: argparse.Namespace) -> int:
    config = IndexConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        min_valid_frames=args.min_valid_frames,
        dry_run=args.dry_run,
        compute_checksums=not args.skip_checksum,
        cube_face_size=args.cube_face_size,
        cube_overlap_px=args.cube_overlap_px,
        cache_cubemap_faces=not args.skip_face_cache,
        dyn_min_component_area_px=args.dyn_min_component_area_px,
        roof_min_component_area_px=args.roof_min_component_area_px,
        dyn_dilate_px=args.dyn_dilate_px,
        roof_dilate_px=args.roof_dilate_px,
        dyn_erode_after_dilate_px=args.dyn_erode_after_dilate_px,
        roof_erode_after_dilate_px=args.roof_erode_after_dilate_px,
        sky_mask_erp_smoothing_iterations=args.sky_mask_erp_smoothing_iterations,
    )
    manifests = run_mask_fusion(config, sequence_ids=args.sequence or None)
    summary = {
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "dry_run": config.dry_run,
        "compute_checksums": config.compute_checksums,
        "cube_face_size": config.cube_face_size,
        "cube_overlap_px": config.cube_overlap_px,
        "cache_cubemap_faces": config.cache_cubemap_faces,
        "dyn_min_component_area_px": config.dyn_min_component_area_px,
        "roof_min_component_area_px": config.roof_min_component_area_px,
        "dyn_dilate_px": config.dyn_dilate_px,
        "roof_dilate_px": config.roof_dilate_px,
        "dyn_erode_after_dilate_px": config.dyn_erode_after_dilate_px,
        "roof_erode_after_dilate_px": config.roof_erode_after_dilate_px,
        "sky_mask_erp_smoothing_iterations": config.sky_mask_erp_smoothing_iterations,
        "sequence_count": len(manifests),
        "fused_sequences": sum(1 for item in manifests if item.status == "ready"),
        "failed_sequences": sum(1 for item in manifests if item.status != "ready"),
        "fused_frames": sum(
            sum(1 for row in item.rows if row.mask_fusion_status == "fused")
            for item in manifests
        ),
        "failed_frames": sum(
            sum(1 for row in item.rows if row.mask_fusion_status == "failed")
            for item in manifests
        ),
        "dynamic_mask_area_px": sum(
            sum((row.dynamic_mask_area_px or 0) for row in item.rows)
            for item in manifests
        ),
        "roof_mask_area_px": sum(
            sum((row.roof_mask_area_px or 0) for row in item.rows)
            for item in manifests
        ),
        "sky_mask_area_px": sum(
            sum((row.sky_mask_area_px or 0) for row in item.rows)
            for item in manifests
        ),
        "inpaint_mask_area_px": sum(
            sum((row.inpaint_mask_area_px or 0) for row in item.rows)
            for item in manifests
        ),
        "sequences": [item.to_summary_dict() for item in manifests],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_sequences"] == 0 else 1


def handle_inpaint_sequence(args: argparse.Namespace) -> int:
    config = IndexConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        min_valid_frames=args.min_valid_frames,
        dry_run=args.dry_run,
        compute_checksums=not args.skip_checksum,
        cube_face_size=args.cube_face_size,
        cube_overlap_px=args.cube_overlap_px,
        cache_cubemap_faces=not args.skip_face_cache,
        inpaint_window_size=args.inpaint_window_size,
        inpaint_window_stride=args.inpaint_window_stride,
        propainter_model_id=args.propainter_model_id,
        propainter_command=args.propainter_command,
        propainter_device=args.propainter_device,
        propainter_chunk_size=args.propainter_chunk_size,
        propainter_face_feather_px=args.propainter_face_feather_px,
    )
    if not config.dry_run and not config.propainter_command:
        print(
            json.dumps(
                {
                    "error": "inpaint-sequence requires --propainter-command unless --dry-run is used",
                    "propainter_command_placeholders": [
                        "{frames_dir}",
                        "{masks_dir}",
                        "{output_dir}",
                        "{face_name}",
                        "{device}",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    manifests = run_propainter_inpainting(config, sequence_ids=args.sequence or None)
    summary = {
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "dry_run": config.dry_run,
        "compute_checksums": config.compute_checksums,
        "cube_face_size": config.cube_face_size,
        "cube_overlap_px": config.cube_overlap_px,
        "cache_cubemap_faces": config.cache_cubemap_faces,
        "inpaint_window_size": config.inpaint_window_size,
        "inpaint_window_stride": config.inpaint_window_stride,
        "propainter_model_id": config.propainter_model_id,
        "propainter_device": config.propainter_device,
        "propainter_chunk_size": config.propainter_chunk_size,
        "propainter_face_feather_px": config.propainter_face_feather_px,
        "sequence_count": len(manifests),
        "inpainted_sequences": sum(1 for item in manifests if item.status == "ready"),
        "failed_sequences": sum(1 for item in manifests if item.status != "ready"),
        "inpainted_frames": sum(
            sum(1 for row in item.rows if row.propainter_status == "inpainted")
            for item in manifests
        ),
        "failed_frames": sum(
            sum(1 for row in item.rows if row.propainter_status == "failed")
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
    if args.command == "detect-dynamic":
        return handle_detect_dynamic(args)
    if args.command == "refine-masks":
        return handle_refine_masks(args)
    if args.command == "fuse-masks":
        return handle_fuse_masks(args)
    if args.command == "inpaint-sequence":
        return handle_inpaint_sequence(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
