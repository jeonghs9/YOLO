# Traffic Video Analytics

Traffic-video pipeline for vehicle and plate detection, lane instance segmentation, tracking, solid-lane crossing detection, and optional plate OCR.

## Current release

The verified deployment entry point is:

```text
scripts/visualize_solid_lane_crossing_v5.3.1.py
```

This file was copied byte-for-byte from the legacy project location. The legacy project remains in place until relocation verification is complete.

## Runtime dependencies

- YOLO runtime: `/home/hsjeong/workspace/Yolo26/ultralytics` (`ultralytics 8.4.6` editable install)
- Integrated runtime environment: `paddleocr_gpu_cu126`
- Detection and lane models: configured in `configs/runtime.yaml`
- PaddleOCR is optional when `--no-ocr` is supplied

`vision-det/YOLO26-MASTER` is not a dependency of this application.

## Layout

```text
src/       reusable modules for future extraction from the standalone release
scripts/   executable entry points
configs/   runtime paths and application defaults
docs/      design, experiment, and release records
tests/     unit, integration, and small test fixtures
deploy/    deployment environment and service files
models/    local weights and engines; excluded from Git
outputs/   generated videos, CSV files, images, and logs; excluded from Git
```

The v5.3.1 implementation remains standalone during the first migration stage. Splitting it into `src/` modules is deferred until behavior-preserving integration tests are in place.

