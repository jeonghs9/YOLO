# VisDrone 실험 기록

데이터: `dataset/VisDrone/cleaning/` (YOLO detect, 10클래스, train/val/test).
test = 1610장 / 75,102객체.

클래스: 0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck, 6 tricycle, 7 awning-tricycle, 8 bus, 9 motor.

## Baseline 학습 (완료)

### YOLO26 (muSGD)
- weights: `ultralytics/runs/BASELINE/260610_Y26{S,N}_BASELINE_VISDRONE/weights/best.pt`
- YOLO26s: Params 9.47M, GFLOPs 20.5, Best epoch 203.
- YOLO26n: Params 2.38M, GFLOPs 5.2, Best epoch 300.

### YOLO11 (SGD, 비교 참고용)
- weights: `ultralytics/runs/BASELINE/260616_Y11{S,N}_BASELINE_VISDRONE/weights/best.pt`
- YOLO11s: Params 9.42M, GFLOPs 21.3, Best epoch 249.
- YOLO11n: Params 2.58M, GFLOPs 6.3, Best epoch 229.

## Baseline val mAP (ultralytics 내장, val split, conf=0.25) — Notion 기록

| 모델 | Optimizer | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLO26s | muSGD | 0.517 | 0.283 | 0.396 | 0.246 |
| YOLO26n | muSGD | 0.545 | 0.213 | 0.377 | 0.232 |
| YOLO11s | SGD | 0.502 | 0.309 | 0.405 | 0.253 |
| YOLO11n | SGD | 0.487 | 0.245 | 0.365 | 0.224 |

> 주의: 위 수치는 ultralytics 내장 val (val split) 기준. pycocotools test AP와 혼용 금지.

### YOLO11s val 클래스별 mAP50 (참고)
| class | mAP50 | | class | mAP50 |
|---|---|---|---|---|
| pedestrian | 0.431 | | tricycle | 0.234 |
| people | 0.344 | | awning-tricycle | 0.255 |
| bicycle | 0.171 | | bus | 0.631 |
| car | 0.755 | | motor | 0.386 |
| van | 0.406 | | truck | 0.437 |

### YOLO11n val 클래스별 mAP50 (참고)
| class | mAP50 | | class | mAP50 |
|---|---|---|---|---|
| pedestrian | 0.373 | | tricycle | 0.174 |
| people | 0.328 | | awning-tricycle | 0.240 |
| bicycle | 0.156 | | bus | 0.581 |
| car | 0.712 | | motor | 0.341 |
| van | 0.366 | | truck | 0.383 |

### 참고: split별 mAP 차이
**test split은 더 어려움** → 아래 크기별 AP 평가(pycocotools) 기준.

## 크기별 AP (pycocotools, test split, conf=0.001, max_det=500) — 2026-06-12

### YOLO26s (test)
| 지표 | 값 |
|---|---|
| AP@[.5:.95] | 0.1646 |
| AP50 | 0.2918 |
| AP75 | 0.1647 |
| AR | 0.3168 |

크기 구간별 (√area px):
| band | AP | AR |
|---|---|---|
| vt(2-8) | 0.0205 | 0.0583 |
| **tiny/APT(8-16)** | **0.0564** | 0.1682 |
| small(16-32) | 0.1191 | 0.2925 |
| medium(32+) | 0.2558 | 0.4415 |
| cocoS(<32) | 0.0845 | 0.2274 |
| cocoM(32-96) | 0.2441 | 0.4302 |
| cocoL(>96) | 0.3342 | 0.5113 |

클래스별 AP@[.5:.95]:
| class | AP |  | class | AP |
|---|---|---|---|---|
| pedestrian | 0.1147 | | tricycle | 0.0766 |
| people | 0.0556 | | awning-tricycle | 0.0765 |
| bicycle | 0.0337 | | bus | 0.3394 |
| car | 0.4379 | | motor | 0.1158 |
| van | 0.2035 | | truck | 0.1926 |

### YOLO26n (test)
| 지표 | 값 |
|---|---|
| AP@[.5:.95] | 0.1394 |
| AP50 | 0.2533 |
| AP75 | 0.1374 |
| AR | 0.2875 |

크기 구간별: vt 0.0077 / **tiny(APT) 0.0376** / small 0.0939 / medium 0.2228 / cocoS 0.0656 / cocoM 0.2094 / cocoL 0.3132.

클래스별 AP@[.5:.95]: pedestrian 0.0871, people 0.0431, bicycle 0.0255, car 0.3963, van 0.1788, truck 0.1530, tricycle 0.0619, awning-tricycle 0.0729, bus 0.2850, motor 0.0903.

### s vs n (test)
| | AP | AP50 | APT(tiny) | small | medium | cocoL |
|---|---|---|---|---|---|---|
| YOLO26s | 0.1646 | 0.2918 | 0.0564 | 0.1191 | 0.2558 | 0.3342 |
| YOLO26n | 0.1394 | 0.2533 | 0.0376 | 0.0939 | 0.2228 | 0.3132 |

s가 전 구간 우세, 격차는 작은 객체일수록 큼(APT ~1.5배 vs large ~1.07배).

### 관찰 (SOD 가설 뒷받침)
- 크기 단조 증가: vt 0.020 → tiny 0.056 → small 0.119 → medium 0.256.
- 가늘고 작은 클래스(bicycle 0.034, people 0.056)가 큰 객체(car 0.438, bus 0.339)보다 현저히 낮음.
- → "작을수록 특징·학습신호 모두 불리" 가설을 데이터가 지지. APT가 핵심 개선 타깃.

## TODO
- YOLO11 pycocotools 크기별 AP (test) 측정 — 요청 시 진행.
- 평가 파이프라인 상세: `../05_eval/SOD_EVAL_PIPELINE.md`.
