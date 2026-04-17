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

## Install

```bash
python3 -m pip install -e .
```

For Mask2Former, install the optional ML runtime in the `urpan-inpaint` environment:

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
