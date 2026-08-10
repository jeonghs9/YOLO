# DOTA v1.5 Dataset Cleaning

## 진행 사항
- 원본 위치: `/home/hsjeong/workspace/SOD-PAPER/dataset/DOTA/original/`
- 정리 위치: `/home/hsjeong/workspace/SOD-PAPER/dataset/DOTA/cleaning/`
- 라벨 버전: DOTA v1.5 HBB (`DOTA-v1.5_train_hbb`, `DOTA-v1.5_val_hbb`)
- 원본 test 셋에는 실제 파일이 없어, 원본 `val` 이미지를 cleaning의 `test` 셋으로 사용했습니다.
- DOTA v1.5 HBB 라벨의 네 꼭짓점 좌표에서 `xmin, ymin, xmax, ymax`를 계산해 YOLO detect 형식 `class x_center y_center width height`로 저장했습니다.
- 원본 `train` 1411장 중 282장을 `cleaning/val`로 분리했습니다.
- `data.yaml`은 `train: train/images`, `val: val/images`, `test: test/images`로 작성했습니다.

## 클래스 매핑
- 0: `plane`
- 1: `baseball-diamond`
- 2: `bridge`
- 3: `ground-track-field`
- 4: `small-vehicle`
- 5: `large-vehicle`
- 6: `ship`
- 7: `tennis-court`
- 8: `basketball-court`
- 9: `storage-tank`
- 10: `soccer-ball-field`
- 11: `roundabout`
- 12: `harbor`
- 13: `swimming-pool`
- 14: `helicopter`
- 15: `container-crane`

## 원본 파일 수
| 구분 | 원본 split | cleaning split | images | labels | objects | 비고 |
|---|---:|---:|---:|---:|---:|---|
| train | train | - | 1411 | 1411 | - |  |
| val | val | - | 458 | 458 | - | cleaning/test로 사용 |
| test | test | - | 0 | 0 | - | 원본 test 파일 없음 |

## Cleaning 파일 수
| 구분 | 원본 split | cleaning split | images | labels | objects | 비고 |
|---|---:|---:|---:|---:|---:|---|
| train | train | train | 1129 | 1129 | 162032 | 원본 train에서 val 분리 후 남은 데이터 |
| val | train | val | 282 | 282 | 48513 | 원본 train의 약 20% |
| test | val | test | 458 | 458 | 69565 | original/val 기반 |

## 변환 상세
- train missing_labels: 0
- train missing_images: 0
- train skipped_unknown: 0
- train skipped_invalid: 86
- train malformed: 2822
- test missing_labels: 0
- test missing_images: 0
- test skipped_unknown: 0
- test skipped_invalid: 0
- test malformed: 916
- train_to_val requested: 282
- train_to_val moved: 282
- train_to_val missing_labels: 0


## HBB Tiling 전처리 적용

- 입력 폴더: `/home/hsjeong/workspace/SOD-PAPER/dataset/DOTA/cleaning`
- 출력 폴더: `/home/hsjeong/workspace/SOD-PAPER/dataset/DOTA/cleaning_tiled`
- 기준: YOLO detect HBB 5열 라벨 `class x_center y_center width height`
- crop_size: `1024`
- gap: `200`
- rates: `[1.0]`
- crop sizes: `[1024]`
- gaps: `[200]`
- object keep threshold: IoF >= `0.7`
- train, val, test 모두 라벨 포함 crop/clipping을 적용했습니다.
- 기존 full-image `cleaning/`은 유지하고, tiled 버전은 `cleaning_tiled/`에 별도로 저장했습니다.

| split | source images | source labels | tiled images | tiled labels | tiled objects | empty tiles |
|---|---:|---:|---:|---:|---:|---:|
| train | 1129 | 1129 | 8210 | 8210 | 293529 | 0 |
| val | 282 | 282 | 2095 | 2095 | 82816 | 0 |
| test | 458 | 458 | 3322 | 3322 | 121387 | 0 |

`cleaning_tiled/data.yaml`은 `train: train/images`, `val: val/images`, `test: test/images`를 가리킵니다.

## 최종 요약

최종 생성 경로:
- full-image YOLO 데이터셋: `/home/hsjeong/workspace/SOD-PAPER/dataset/DOTA/cleaning`
- tiled YOLO 데이터셋: `/home/hsjeong/workspace/SOD-PAPER/dataset/DOTA/cleaning_tiled`
- full-image YAML: `/home/hsjeong/workspace/SOD-PAPER/dataset/DOTA/cleaning/data.yaml`
- tiled YAML: `/home/hsjeong/workspace/SOD-PAPER/dataset/DOTA/cleaning_tiled/data.yaml`
- 변환 스크립트: `/home/hsjeong/workspace/SOD-PAPER/dataset/DOTA/convert_dota_to_yolo.py`
- 타일링 스크립트: `/home/hsjeong/workspace/SOD-PAPER/dataset/DOTA/tile_yolo_hbb.py`

용량:
- `cleaning`: 약 `14G`
- `cleaning_tiled`: 약 `4.4G`

Full-image 최종 결과:

| split | images | labels | objects |
|---|---:|---:|---:|
| train | 1129 | 1129 | 162032 |
| val | 282 | 282 | 48513 |
| test | 458 | 458 | 69565 |

Tiled 최종 결과:

| split | images | labels | objects |
|---|---:|---:|---:|
| train | 8210 | 8210 | 293529 |
| val | 2095 | 2095 | 82816 |
| test | 3322 | 3322 | 121387 |

최종 클래스 수는 DOTA v1.5 기준 `16`개이며, 추가 클래스 `container-crane`은 class id `15`로 매핑했습니다.

---

## Baseline 학습 결과 (완료)

데이터: `dataset/DOTA/cleaning_tiled/` (tiled 640, 16클래스).
평가: ultralytics 내장 val, val split (2095장), conf=0.25.

### 학습 설정
| 모델 | Optimizer | Params | GFLOPs | Best epoch | weights |
|---|---|---|---|---|---|
| YOLO26s | muSGD | 9.47M | 20.5 | 219 | `runs/BASELINE/260611_Y26S_BASELINE_DOTA/weights/best.pt` |
| YOLO26n | muSGD | 2.38M | 5.2 | 300 | `runs/BASELINE/260611_Y26N_BASELINE_DOTA/weights/best.pt` |
| YOLO11s | SGD | 9.42M | 21.3 | 300 | `runs/BASELINE/260616_Y11S_BASELINE_DOTA/weights/best.pt` |
| YOLO11n | SGD | 2.59M | 6.3 | 219 | `runs/BASELINE/260616_Y11N_BASELINE_DOTA/weights/best.pt` |

### val mAP (ultralytics 내장, val split)

| 모델 | Optimizer | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLO26s | muSGD | 0.691 | 0.509 | 0.604 | 0.391 |
| YOLO26n | muSGD | 0.650 | 0.489 | 0.581 | 0.371 |
| YOLO11s | SGD | 0.676 | 0.532 | 0.616 | 0.405 |
| YOLO11n | SGD | 0.637 | 0.525 | 0.593 | 0.375 |

### YOLO26s 클래스별 mAP50 (val)
| class | mAP50 | | class | mAP50 |
|---|---|---|---|---|
| plane | 0.904 | | storage-tank | 0.780 |
| baseball-diamond | 0.718 | | soccer-ball-field | 0.483 |
| bridge | 0.580 | | roundabout | 0.701 |
| ground-track-field | 0.578 | | harbor | 0.728 |
| small-vehicle | 0.612 | | swimming-pool | 0.724 |
| large-vehicle | 0.557 | | helicopter | 0.568 |
| ship | 0.848 | | container-crane | 0.000 |
| tennis-court | 0.434 | | basketball-court | 0.450 |

### YOLO26n 클래스별 mAP50 (val)
| class | mAP50 | | class | mAP50 |
|---|---|---|---|---|
| plane | 0.894 | | storage-tank | 0.728 |
| baseball-diamond | 0.682 | | soccer-ball-field | 0.479 |
| bridge | 0.512 | | roundabout | 0.658 |
| ground-track-field | 0.523 | | harbor | 0.724 |
| small-vehicle | 0.577 | | swimming-pool | 0.744 |
| large-vehicle | 0.539 | | helicopter | 0.487 |
| ship | 0.834 | | container-crane | 0.000 |
| tennis-court | 0.460 | | basketball-court | 0.450 |

### YOLO11s 클래스별 mAP50 (val)
| class | mAP50 | | class | mAP50 |
|---|---|---|---|---|
| plane | 0.913 | | storage-tank | 0.779 |
| baseball-diamond | 0.762 | | soccer-ball-field | 0.485 |
| bridge | 0.555 | | roundabout | 0.719 |
| ground-track-field | 0.648 | | harbor | 0.761 |
| small-vehicle | 0.605 | | swimming-pool | 0.718 |
| large-vehicle | 0.556 | | helicopter | 0.585 |
| ship | 0.852 | | container-crane | 0.000 |
| tennis-court | 0.438 | | basketball-court | 0.486 |

### YOLO11n 클래스별 mAP50 (val)
| class | mAP50 | | class | mAP50 |
|---|---|---|---|---|
| plane | 0.900 | | storage-tank | 0.752 |
| baseball-diamond | 0.692 | | soccer-ball-field | 0.464 |
| bridge | 0.446 | | roundabout | 0.645 |
| ground-track-field | 0.562 | | harbor | 0.734 |
| small-vehicle | 0.579 | | swimming-pool | 0.734 |
| large-vehicle | 0.582 | | helicopter | 0.539 |
| ship | 0.840 | | container-crane | 0.000 |
| tennis-court | 0.494 | | basketball-court | 0.529 |

### 관찰
- YOLO11s > YOLO26s (mAP50: 0.616 vs 0.604, +0.012) — DOTA에서 YOLO11이 약간 앞섬.
- container-crane: 전 모델 AP=0.000 (학습 데이터 절대 부족).
- small-vehicle, bridge 등 작고 세로로 긴 클래스에서 전 모델 성능 저조.

## TODO
- pycocotools 크기별 AP (test split) 측정 — 요청 시 진행.
