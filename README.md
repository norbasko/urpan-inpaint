# urpan-inpaint

`urpan-inpaint` is the production pipeline scaffold for 360-degree street-scene panorama cleanup on the KAUST GoPro MAX dataset rooted at `/mnt/vision/data/kaust/`.

This implementation currently covers:

- Sequence discovery and indexing:
  - discover `GS*` sequence directories,
  - read `gps-fixed.csv` as the authoritative per-sequence frame manifest,
  - verify the canonical `fixed/` frame paths referenced by the CSV,
  - skip missing frames while preserving their metadata in the output manifest,
  - fail sequences with a missing CSV or with fewer than the configured minimum number of valid frames,
  - write normalized `manifests/frames.csv` and `manifests/sequence_summary.json` files,
  - precompute downstream output paths for RGB, RGBA, masks, cubemap cache, and QA artifacts.
- ERP normalization metadata:
  - load canonical ERP JPEGs from `fixed/`,
  - preserve original width and height,
  - convert frames to in-memory RGB `uint8`,
  - record original dimensions, source mode, SHA-256 checksum, and seam wrap mode in the manifest,
  - avoid any lossy intermediate re-encoding or global crop.
- ERP to cubemap projection:
  - convert each canonical ERP frame into `front`, `right`, `back`, `left`, `up`, and `down` faces,
  - add guard-band overlap on all faces,
  - retain exact inverse-reprojection metadata alongside each cached face tensor,
  - optionally cache face tensors as `.npz` files under each frame's cubemap cache directory.
- Semantic / panoptic parsing:
  - run Mask2Former on cubemap faces,
  - write per-face semantic label maps and required-class coarse masks,
  - write confidence maps when available,
  - write panoptic instance maps and segment sidecars when available,
  - use a Cityscapes Mask2Former checkpoint by default and continue even if only semantic outputs are available.
- Open-vocabulary dynamic detection:
  - run Grounding DINO on cubemap faces,
  - use the default prompt vocabulary `person`, `pedestrian`, `rider`, `bicyclist`, `bicycle`, `motorcycle`, `scooter`, `car`, `van`, `pickup truck`, `truck`, `bus`, `trailer`, `caravan`,
  - drop detections below `box_threshold`,
  - merge overlapping detections with class-aware NMS,
  - keep small high-confidence detections because vulnerable road users matter.
- Promptable mask refinement and temporal propagation:
  - run SAM 2 video inference on cubemap-face frame sequences with streaming memory,
  - refine Grounding DINO boxes, semantic region boxes/points, and the explicit down-face roof prompt,
  - refine sky and Cityscapes-style skyline classes when their semantic masks are available,
  - associate stable prompts across neighboring frames and propagate refined masks through SAM 2 memory.
- Roof mask generation:
  - build an expanded elliptical nadir prior on the cubemap down face,
  - refine the current down-face roof candidate with SAM 2 when available,
  - regularize roof masks with neighboring-frame medians,
  - emit ERP-space `masks/roof/*.png` masks plus JSON sidecars,
  - fall back to the temporal median or coarse nadir prior when roof SAM 2 refinement fails.
- Sky mask generation:
  - start from Mask2Former `sky` target masks,
  - optionally use SAM 2 only in a narrow boundary band around semantic skyline edges,
  - suppress semantic building, tree, vegetation, vehicle, and person masks before compositing,
  - enforce conservative top-connected sky topology on side faces,
  - reproject to ERP and apply seam-aware smoothing without expanding sky into foreground.
- Mask fusion:
  - build `DYN_COARSE` from parser dynamic classes,
  - build `DYN_OVD` from Grounding DINO prompts refined by SAM 2,
  - apply hole filling, small-component suppression, edge dilation, optional erosion, and temporal consistency to dynamic and roof masks,
  - seam-smooth the refined sky mask,
  - write final `dynamic`, `roof`, `sky`, and `inpaint` masks where `INPAINT = (DYN OR ROOF) AND NOT SKY`.
- Temporal windowing for inpainting:
  - split eligible frames into overlapping sequence windows,
  - default to `window_size=24` and `window_stride=12`,
  - support short sequences with a single shorter window,
  - reconcile overlapping predictions by choosing the window where each frame is closest to the temporal center,
  - break reconciliation ties deterministically in favor of the earlier window.
- Sequence-level inpainting:
  - run ProPainter independently on `front`, `right`, `back`, `left`, `up`, and `down` face streams,
  - use fused ERP `INPAINT` masks projected into each face as mask guidance,
  - preserve unmasked face pixels after ProPainter returns predictions,
  - support chunked inference inside temporal windows for memory safety,
  - crop guard bands for per-face artifacts and feather face seams during ERP reprojection.
- Single-frame fallback inpainting:
  - run LaMa per cubemap face when ProPainter is unavailable, exhausts OOM retries, or is not worth invoking,
  - trigger fallback for too-short windows, very small masks, or `--force-single-frame-fallback`,
  - preserve the same non-sky `INPAINT` mask composition, guard-band cropping, and feathered ERP seam rules,
  - record the fallback reason and LaMa status in the frame manifest.

## Install

```bash
python3 -m pip install -e .
```

For Mask2Former, Grounding DINO, and SAM 2, install the optional ML runtime in the `urpan-inpaint` environment:

```bash
python3 -m pip install -e ".[ml]"
```

## Usage

Index every sequence using the production defaults:

```bash
urpan-inpaint index
```

Index a subset and write results to a sandbox-safe output root:

```bash
urpan-inpaint index \
  --sequence GS010001 \
  --sequence GS010002 \
  --output-root /tmp/urpan-inpaint-output
```

Dry-run discovery without writing manifests:

```bash
urpan-inpaint index --dry-run
```

Normalize ERP frames and enrich the manifest with image metadata:

```bash
urpan-inpaint normalize-erp \
  --sequence GS010001 \
  --output-root /tmp/urpan-inpaint-output
```

Skip checksum computation when you only want geometry/type validation:

```bash
urpan-inpaint normalize-erp --skip-checksum
```

Project ERP panoramas into cubemap faces and cache them on disk:

```bash
urpan-inpaint project-cubemap \
  --sequence GS030002 \
  --output-root /tmp/urpan-inpaint-output
```

Use smaller face sizes for quick smoke tests:

```bash
urpan-inpaint project-cubemap \
  --sequence GS030002 \
  --cube-face-size 128 \
  --cube-overlap-px 8 \
  --skip-checksum
```

Run Mask2Former semantic parsing on cube faces:

```bash
urpan-inpaint parse-semantic \
  --sequence GS030002 \
  --output-root /tmp/urpan-inpaint-output \
  --semantic-model-id facebook/mask2former-swin-tiny-cityscapes-semantic
```

Restrict semantic parsing to locally cached checkpoint files:

```bash
urpan-inpaint parse-semantic \
  --semantic-local-files-only \
  --skip-panoptic
```

Run Grounding DINO dynamic detection on cubemap faces:

```bash
urpan-inpaint detect-dynamic \
  --sequence GS030002 \
  --output-root /tmp/urpan-inpaint-output \
  --grounding-model-id IDEA-Research/grounding-dino-tiny
```

Override the prompt set and tighten detection filtering:

```bash
urpan-inpaint detect-dynamic \
  --grounding-prompt person \
  --grounding-prompt bicycle \
  --grounding-prompt car \
  --box-threshold 0.3 \
  --nms-iou-threshold 0.6
```

Run SAM 2 refinement from the coarse masks and boxes:

```bash
urpan-inpaint refine-masks \
  --sequence GS030002 \
  --output-root /tmp/urpan-inpaint-output \
  --sam2-model-id facebook/sam2.1-hiera-tiny
```

Tune the roof prior and temporal regularization:

```bash
urpan-inpaint refine-masks \
  --sam2-roof-box-fraction 0.55 \
  --sam2-roof-prior-margin-fraction 0.15 \
  --sam2-roof-temporal-window 1
```

Tune conservative sky compositing masks:

```bash
urpan-inpaint refine-masks \
  --sky-mask-top-seed-fraction 0.12 \
  --sky-mask-sam2-boundary-margin-px 3 \
  --sky-mask-obstacle-dilation-px 1
```

Restrict SAM 2 to local checkpoint files and disable temporal prior prompting:

```bash
urpan-inpaint refine-masks \
  --sam2-local-files-only \
  --disable-temporal-propagation
```

Fuse final per-frame masks:

```bash
urpan-inpaint fuse-masks \
  --sequence GS030002 \
  --output-root /tmp/urpan-inpaint-output \
  --dyn-min-component-area-px 64 \
  --roof-min-component-area-px 256 \
  --dyn-dilate-px 3 \
  --roof-dilate-px 5
```

Run sequence-level ProPainter inpainting from the fused masks:

```bash
urpan-inpaint inpaint-sequence \
  --sequence GS030002 \
  --output-root /tmp/urpan-inpaint-output \
  --propainter-command "python run_propainter.py --frames {frames_dir} --masks {masks_dir} --output {output_dir} --device {device}" \
  --inpaint-window-size 24 \
  --inpaint-window-stride 12 \
  --propainter-chunk-size 8
```

Enable LaMa fallback for short clips, small masks, ProPainter failures, or explicit single-frame operation:

```bash
urpan-inpaint inpaint-sequence \
  --sequence GS030002 \
  --output-root /tmp/urpan-inpaint-output \
  --propainter-command "python run_propainter.py --frames {frames_dir} --masks {masks_dir} --output {output_dir} --device {device}" \
  --lama-command "python run_lama.py --image {image_path} --mask {mask_path} --output {output_path} --device {device}" \
  --propainter-min-window-frames 2 \
  --single-frame-min-mask-area-px 64
```

Force single-frame fallback without trying ProPainter:

```bash
urpan-inpaint inpaint-sequence \
  --sequence GS030002 \
  --output-root /tmp/urpan-inpaint-output \
  --force-single-frame-fallback \
  --lama-command "python run_lama.py --image {image_path} --mask {mask_path} --output {output_path}"
```

## Output layout

For each sequence `GSxxxxxx`, the pipeline writes:

```text
<output-root>/GSxxxxxx/
  rgb/
  rgba/
  masks/
    dynamic/
    roof/
    sky/
    inpaint/
    union_debug/
  cubemap/
  qa/
    overlays/
    contact_sheets/
  manifests/
    frames.csv
    sequence_summary.json
```

No normalized intermediate image files are written during ERP normalization. The canonical JPEGs remain untouched until a later final-write path emits RGB or RGBA outputs.

Semantic face outputs are written under each frame's cubemap cache directory:

```text
<output-root>/GSxxxxxx/cubemap/GSxxxxxx-frame-YYYYYY/
  projection.json
  front.npz
  right.npz
  back.npz
  left.npz
  up.npz
  down.npz
  semantic_mask2former/
    metadata.json
    front.npz
    front.panoptic.json
    ...
```

Grounding DINO outputs are written alongside the cubemap cache as:

```text
<output-root>/GSxxxxxx/cubemap/GSxxxxxx-frame-YYYYYY/
  grounding_dino/
    metadata.json
    front.npz
    front.json
    right.npz
    right.json
    ...
```

SAM 2 refined masks are written alongside the cubemap cache as:

```text
<output-root>/GSxxxxxx/cubemap/GSxxxxxx-frame-YYYYYY/
  sam2_refined/
    metadata.json
    front.npz
    front.json
    right.npz
    right.json
    ...
```

Each SAM 2 face `.npz` contains binary `masks`, refined `boxes_xyxy`, `scores`, `class_text`, prompt provenance, and a `used_temporal_prior` flag. The down face is always eligible for the roof prompt unless `--skip-roof-prompt` is passed.

The finalized per-frame roof mask is written to `masks/roof/<frame>.png` in ERP coordinates. Its adjacent JSON sidecar records whether the mask came from current SAM 2 evidence, temporal regularization, temporal fallback, coarse-prior fallback, or a current-vs-temporal disagreement where current evidence was kept.

The finalized per-frame sky mask is written to `masks/sky/<frame>.png` in ERP coordinates when usable Mask2Former sky predictions exist. Its JSON sidecar records source faces, whether SAM 2 boundary refinement contributed, and the conservative topology/smoothing policy used for alpha compositing.

The fusion stage overwrites final ERP-space masks under `masks/dynamic`, `masks/roof`, `masks/sky`, and `masks/inpaint`, and writes RGB union debug masks under `masks/union_debug`. Fusion JSON sidecars record source terms, morphology settings, warnings for missing inputs, and final mask areas.

Temporal inpainting windows are planned by `urpan_inpaint.windowing`. The helper keeps windows overlapping by requiring `inpaint_window_stride < inpaint_window_size`, covers all valid frames, and exposes a reconciliation plan that downstream inpainting runners can use to stitch overlapping window predictions deterministically.

Sequence-level inpainting writes final ERP `rgb/<frame>.png` and `rgba/<frame>.png` outputs. Final products are PNG only and must match the input ERP resolution exactly. RGBA alpha is derived from the final ERP sky mask: `alpha=0` where `SKY == 255`, and `alpha=255` elsewhere. ProPainter face artifacts are written under each frame's cubemap directory as `propainter/<face>/<frame>.with_overlap.png`, the cropped `propainter/<face>/<frame>.png`, and the projected face mask `propainter/<face>/<frame>.mask.png`. Sequence-level `propainter/metadata.json` records face order, windows, chunks, and seam-feather settings.

LaMa fallback writes the same final ERP `rgb` and `rgba` outputs. Face artifacts are written under each frame's cubemap directory as `lama_fallback/<face>/...`, and sequence-level `lama_fallback/metadata.json` records the fallback reason, processed frame indices, face order, and seam-feather settings. The frame manifest records `single_frame_fallback_reason`, `lama_status`, `lama_model_id`, and `lama_output_dir`.
