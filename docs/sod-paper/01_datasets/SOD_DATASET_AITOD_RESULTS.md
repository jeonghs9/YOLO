# AI-TOD 실험 기록

데이터: `dataset/AI-TOD/cleaning/` (YOLO detect, 8클래스, train/val/test).

클래스: 0 airplane, 1 bridge, 2 storage-tank, 3 ship, 4 swimming-pool, 5 vehicle, 6 person, 7 wind-mill.

## Baseline 학습 (완료)

### YOLO26 (muSGD)
- weights: `ultralytics/runs/BASELINE/260612_Y26{S,N}_BASELINE_AI-TOD/weights/best.pt`
- YOLO26s: Params 9.47M, GFLOPs 20.6, Best epoch 300.
- YOLO26n: Params 2.38M, GFLOPs 5.2, Best epoch 300.

### YOLO11 (SGD, 비교 참고용)
- RUN_260617_01 (yolo11s): 진행 중
- RUN_260617_02 (yolo11n): 시작 전

## Baseline val mAP (ultralytics 내장, val split, conf=0.25) — Notion 기록

| 모델 | Optimizer | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLO26s | muSGD | 0.717 | 0.407 | 0.570 | 0.299 |
| YOLO26n | muSGD | 0.677 | 0.298 | 0.493 | 0.238 |
| YOLO11s | SGD | — | — | — | — |
| YOLO11n | SGD | — | — | — | — |

### YOLO26s 클래스별 mAP50 (val)
| class | mAP50 | | class | mAP50 |
|---|---|---|---|---|
| airplane | 0.699 | | swimming-pool | 0.470 |
| bridge | 0.452 | | vehicle | 0.689 |
| storage-tank | 0.784 | | person | 0.487 |
| ship | 0.779 | | wind-mill | 0.196 |

### YOLO26n 클래스별 mAP50 (val)
| class | mAP50 | | class | mAP50 |
|---|---|---|---|---|
| airplane | 0.605 | | swimming-pool | 0.408 |
| bridge | 0.319 | | vehicle | 0.634 |
| storage-tank | 0.733 | | person | 0.419 |
| ship | 0.685 | | wind-mill | 0.137 |

### 관찰
- wind-mill: 전 모델 mAP50 < 0.20 (작고 희귀).
- bridge: s 0.452, n 0.319 — 작고 가로로 긴 구조물, 성능 낮음.
- swimming-pool: Recall 극히 낮음 (s 0.099, n 0.014) — 유사 배경 혼동.
- → SOD 개선 타깃 클래스: wind-mill, bridge, swimming-pool.

## TODO
- YOLO11 AI-TOD 학습 완료 후 val mAP 업데이트.
- pycocotools 크기별 AP (test split) 측정 — 요청 시 진행.
- 평가 파이프라인: `../05_eval/SOD_EVAL_PIPELINE.md`.
