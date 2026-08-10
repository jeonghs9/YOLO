# Migration Verification — 2026-08-10

## Scope

This verification covers the first-stage relocation of the standalone deployment script from the legacy project:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/PROJECT/OCR-CAR-CATEGORY/UTIL/visualize_solid_lane_crossing_v5.3.1.py
```

to:

```text
/home/hsjeong/workspace/applications/traffic-video-analytics/scripts/visualize_solid_lane_crossing_v5.3.1.py
```

The legacy project, YOLO runtime, models, datasets, and result files were not moved or deleted.

## Runtime under test

```text
Conda environment: paddleocr_gpu_cu126
Ultralytics source: /home/hsjeong/workspace/Yolo26/ultralytics/ultralytics
Ultralytics version: 8.4.6
Detection model: /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt
Lane model: /home/hsjeong/workspace/Yolo26/ultralytics/PROJECT/OCR-CAR-CATEGORY/260727_Y26S_V_LANE_seg/weights/best.pt
Input: /home/hsjeong/workspace/dataset/TEST_VIDEO/19700101_001037_T.mp4
GPU: 1
```

`vision-det/YOLO26-MASTER` was not used by this application or this verification.

## Results

### Source integrity

The legacy and relocated scripts had the same SHA-256 value:

```text
313b445429d0760d663304e5742e7773c55e01c1bfdf35bb1c0b909bf291b8d0
```

Both files passed Python bytecode compilation. Their `--help` output was byte-identical.

### Ten-frame regression without OCR

Both scripts processed the same first 10 frames using the same models, input, GPU, and arguments. The generated files were byte-identical.

```text
video SHA-256: b954d08fa375574b4b7a1a9841124f481d7d4dee7921100d142bb2b93ba5444a
CSV SHA-256:   d4095b18cba634f27619f98c15a21f5d0baf8219a41d2ace9e0e8a7e6592476d
CSV rows:      255
```

Both runs reported 26 vehicles, 12 solid-lane decisions, no bridge, and no fired event on the first processed frame. The final processed frame also matched.

### OCR smoke test

The relocated script successfully initialized the cached PaddleOCR models on GPU 1 and processed one frame with OCR enabled.

The OCR cache still resides at the legacy path:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/UTIL/LPR/License-Plate-Recognition-System
```

This path must remain available until the cache location is deliberately migrated and revalidated.

## Decision

The relocated standalone script is behavior-equivalent for the tested 10-frame no-OCR regression and successfully runs the OCR-enabled startup path. This is sufficient to start using the new script location, but not sufficient to delete the entire legacy project.

Before deleting legacy files, preserve or relocate the training, dataset-preparation, SegFormer, and historical investigation assets listed in `UTIL_FILE_DISPOSITION.md`.

