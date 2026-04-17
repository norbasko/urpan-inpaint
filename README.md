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

Restrict SAM 2 to local checkpoint files and disable temporal prior prompting:

```bash
urpan-inpaint refine-masks \
  --sam2-local-files-only \
  --disable-temporal-propagation
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
