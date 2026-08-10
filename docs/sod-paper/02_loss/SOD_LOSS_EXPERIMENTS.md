# Loss 실험 기록

box regression loss 변형 실험. 모든 실험은 YOLO26s + muSGD + VisDrone val split 기준 (conf=0.25).

Baseline: `BboxLoss.forward()` → `CIoU=True` (ultralytics 기본).

---

## 실험 목록

| 실험 ID | Loss | ratio | 모델 | 데이터셋 | mAP50 | mAP50-95 | P | R | 상태 |
|---|---|---|---|---|---|---|---|---|---|
| BASELINE | CIoU | — | YOLO26s | VisDrone | 0.396 | 0.246 | 0.517 | 0.283 | 완료 |
| RUN_260617_03 | InnerCIoU | 1.0 | YOLO26s | VisDrone | 0.403 | 0.251 | 0.525 | 0.285 | 완료 |
| RUN_260618_01 | InnerSIoU | 1.0 | YOLO26s | VisDrone | **0.406** | **0.253** | 0.528 | 0.285 | 완료 |
| RUN_260619_01 | NWD | C=14.5 | YOLO26s | VisDrone | 0.410† | 0.248 | 0.561 | 0.264 | 완료 |
| RUN_260622_01 | InnerSIoU | 0.7 | YOLO26s | VisDrone | 0.406 | 0.251 | 0.530 | 0.286 | 완료 |
| RUN_260622_02 | InnerCIoU | 0.7 | YOLO26s | VisDrone | 0.400 | 0.248 | 0.524 | 0.283 | 완료 |
| RUN_262623_01 | WIoU v3 | α=1.9,δ=3 | YOLO26s | VisDrone | 0.410 | 0.253 | 0.533 | 0.290 | 완료 |
| RUN_260624_01 | SA-WIoU v1 (stride bin) | 3-bin | YOLO26s | VisDrone | 0.406 | 0.251 | 0.526 | 0.290 | 완료 |
| RUN_260624_02 | SA-WIoU v2 (픽셀 bin) | 4-bin(8/16/32) | YOLO26s | VisDrone | 0.403 | 0.253 | 0.524 | 0.286 | 완료 (역효과) |

† NWD val mAP50은 최고(0.410)이나 pycocotools test 기준 전 구간 Baseline 이하 — val/test 평가 조건 차이(conf=0.25 vs 0.001)로 인한 착시. pycocotools 기준 최고는 InnerSIoU r=1.0.

---

## InnerCIoU (ratio=1.0) — RUN_260617_03

- 코드: `bbox_iou(..., InnerCIoU=True, ratio=1.0)`
- ratio=1.0이므로 inner box = original box → 수치상 CIoU와 동일해야 하나, 실제 차이 발생 (훈련 variance 가능성 있음)

### val 결과 (ultralytics 내장, val split, conf=0.25)

| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| BASELINE (CIoU) | 0.517 | 0.283 | 0.396 | 0.246 |
| InnerCIoU (ratio=1.0) | 0.525 | 0.285 | **0.403** | **0.251** |
| Δ | +0.008 | +0.002 | **+0.007** | **+0.005** |

### 클래스별 mAP50 (val)

| class | Baseline | InnerCIoU |
|---|---|---|
| pedestrian | — | 0.460 |
| people | — | 0.352 |
| bicycle | — | 0.203 |
| car | — | 0.748 |
| van | — | 0.394 |
| truck | — | 0.415 |
| tricycle | — | 0.212 |
| awning-tricycle | — | 0.254 |
| bus | — | 0.600 |
| motor | — | 0.388 |
| **all** | 0.396 | **0.403** |

> 주의: baseline 클래스별 수치가 Notion에 없어 `—` 처리. 추가 시 업데이트.

### 관찰
- mAP50 +0.007, mAP50-95 +0.005 소폭 향상.
- ratio=1.0 기준이므로 inner box 축소 효과 없음 — ratio < 1.0 실험으로 추가 개선 여지 있음.
- InnerSIoU로 angle/distance/shape cost 도입 시 SOD 성능 추가 향상 기대.

---

## InnerSIoU (ratio=1.0) — RUN_260618_01

- 코드: `bbox_iou(..., InnerSIoU=True, ratio=1.0)`
- SIoU의 angle + distance + shape cost를 inner box IoU에 결합.
- Best epoch: 131

### val 결과 (ultralytics 내장, val split, conf=0.25)

| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| BASELINE (CIoU) | 0.517 | 0.283 | 0.396 | 0.246 |
| InnerCIoU (ratio=1.0) | 0.525 | 0.285 | 0.403 | 0.251 |
| InnerSIoU (ratio=1.0) | 0.528 | 0.285 | **0.406** | **0.253** |
| Δ vs Baseline | +0.011 | +0.002 | **+0.010** | **+0.007** |
| Δ vs InnerCIoU | +0.003 | ±0 | **+0.003** | **+0.002** |

### 클래스별 mAP50 (val)

| class | InnerCIoU | InnerSIoU | Δ |
|---|---|---|---|
| pedestrian | 0.460 | 0.449 | -0.011 |
| people | 0.352 | 0.349 | -0.003 |
| bicycle | 0.203 | 0.210 | +0.007 |
| car | 0.748 | 0.747 | -0.001 |
| van | 0.394 | 0.389 | -0.005 |
| truck | 0.415 | 0.412 | -0.003 |
| tricycle | 0.212 | 0.227 | **+0.015** |
| awning-tricycle | 0.254 | 0.279 | **+0.025** |
| bus | 0.600 | 0.604 | +0.004 |
| motor | 0.388 | 0.393 | +0.005 |
| **all** | 0.403 | **0.406** | +0.003 |

### 관찰
- mAP50 0.396 → 0.406 (+0.010, baseline 대비). InnerCIoU 대비 +0.003 추가 향상.
- angle/distance/shape cost 도입 효과: tricycle(+0.015), awning-tricycle(+0.025) 등 작고 가늘어 방향성 있는 클래스 개선.
- 큰 클래스(car, bus)는 거의 변화 없음 — SIoU cost가 SOD 타깃 클래스에 집중 효과.
- ratio=1.0이므로 inner box 축소 효과 없음. ratio < 1.0 실험은 NWD 이후 우선순위 재논의.

---

## NWD (C=14.5) — RUN_260619_01

- 코드: `wasserstein_nwd(pred, target, xywh=False, C=14.5)`
- 박스를 2D Gaussian으로 모델링, W2² closed-form + exp(-√W2²/C)
- C=14.5: VisDrone train 343,204 객체 기반 산정 (sqrt((mean_w/2)²+(mean_h/2)²), mean_w=16.1px, mean_h=24.1px @640)
- Best epoch: 300 (CIoU/InnerSIoU 대비 늦은 수렴)
- 브랜치: `1.1-loss_nwd` (main 기반)

### val 결과 (ultralytics 내장, val split, conf=0.25)

| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| BASELINE (CIoU) | 0.517 | 0.283 | 0.396 | 0.246 |
| InnerCIoU (ratio=1.0) | 0.525 | 0.285 | 0.403 | 0.251 |
| InnerSIoU (ratio=1.0) | 0.528 | 0.285 | 0.406 | 0.253 |
| **NWD (C=14.5)** | **0.561** | **0.264** | **0.410** | 0.248 |
| Δ NWD vs Baseline | +0.044 | -0.019 | **+0.014** | +0.002 |
| Δ NWD vs InnerSIoU | +0.033 | -0.021 | **+0.004** | -0.005 |

### 클래스별 mAP50 (val)

| class | InnerSIoU | NWD | Δ |
|---|---|---|---|
| pedestrian | 0.449 | 0.432 | -0.017 |
| people | 0.349 | 0.358 | +0.009 |
| bicycle | 0.210 | **0.240** | **+0.030** |
| car | 0.747 | 0.740 | -0.007 |
| van | 0.389 | 0.408 | +0.019 |
| truck | 0.412 | 0.425 | +0.013 |
| tricycle | 0.227 | 0.250 | **+0.023** |
| awning-tricycle | 0.279 | 0.254 | -0.025 |
| bus | 0.604 | 0.599 | -0.005 |
| motor | 0.393 | 0.392 | -0.001 |
| **all** | 0.406 | **0.410** | **+0.004** |

### 관찰
- mAP50 0.410 — 현재까지 4개 실험 중 최고. Baseline 대비 +0.014.
- **Precision 0.561 (최고) vs Recall 0.264 (최저)**: NWD가 high-confidence 예측에 집중, 저신뢰 예측을 억제하는 경향.
- mAP50-95 0.248 — InnerSIoU(0.253)보다 낮음: IoU-based localization 정확도는 Inner계열이 우세.
- bicycle(+0.030), tricycle(+0.023) 개선 — Gaussian 모델링이 작고 비겹침 많은 클래스에 효과적.
- awning-tricycle(-0.025): InnerSIoU의 shape cost 이점이 사라짐 — NWD는 형태 정보를 직접 활용 안 함.
- Best epoch 300 (조기종료 없음): Gaussian loss가 수렴이 느림 → size-adaptive 결합에서 초기엔 CIoU, 후기엔 NWD 비중 조절 가능성.
- **pycocotools 크기별 AP 측정 필수**: val mAP만으로 APtiny/APmedium 가설 검증 불가.

---

## 종합 비교 (val split, mAP50 기준)

| Loss | mAP50 | mAP50-95 | P | R | Best epoch |
|---|---|---|---|---|---|
| CIoU (Baseline) | 0.396 | 0.246 | 0.517 | 0.283 | 203 |
| InnerCIoU (r=1.0) | 0.403 | 0.251 | 0.525 | 0.285 | — |
| InnerSIoU (r=1.0) | 0.406 | 0.253 | 0.528 | 0.285 | 131 |
| **NWD (C=14.5)** | **0.410** | 0.248 | 0.561 | 0.264 | 300 |

**val mAP50**: NWD†(0.410) > InnerSIoU r=1.0(0.406) = InnerSIoU r=0.7(0.406) > InnerCIoU r=1.0(0.403) > InnerCIoU r=0.7(0.400) > Baseline(0.396)  
**mAP50-95**: InnerSIoU r=1.0(0.253) > InnerCIoU r=1.0(0.251) = InnerSIoU r=0.7(0.251) > NWD(0.248) ≈ InnerCIoU r=0.7(0.248) ≈ Baseline(0.246)  
**pycocotools AP@.5:.95**: InnerSIoU r=1.0(0.1703) > InnerCIoU r=1.0(0.1697) > Baseline(0.1646) > NWD(0.1570)

---

## InnerSIoU ratio=0.7 — RUN_260622_01

- 코드: `bbox_iou(..., InnerSIoU=True, ratio=0.7)`
- Best epoch: 166
- 브랜치: `1-loss` (현재 loss.py 상태)

### val 결과

| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| InnerSIoU (r=1.0) | 0.528 | 0.285 | **0.406** | **0.253** |
| InnerSIoU (r=0.7) | 0.530 | 0.286 | 0.406 | 0.251 |
| Δ | +0.002 | +0.001 | ±0 | -0.002 |

### 클래스별 mAP50 (val)

| class | r=1.0 | r=0.7 | Δ |
|---|---|---|---|
| pedestrian | 0.449 | 0.449 | ±0 |
| people | 0.349 | 0.352 | +0.003 |
| bicycle | 0.210 | 0.229 | **+0.019** |
| car | 0.747 | 0.747 | ±0 |
| van | 0.389 | 0.391 | +0.002 |
| truck | 0.412 | 0.396 | -0.016 |
| tricycle | 0.227 | 0.246 | +0.019 |
| awning-tricycle | 0.279 | 0.241 | **-0.038** |
| bus | 0.604 | 0.611 | +0.007 |
| motor | 0.393 | 0.396 | +0.003 |
| **all** | **0.406** | 0.406 | ±0 |

### 관찰
- mAP50는 r=1.0과 동일, mAP50-95는 오히려 -0.002 저하.
- bicycle, tricycle은 소폭 개선되나 awning-tricycle -0.038로 상쇄.
- ratio 축소가 VisDrone 극소 객체에 추가 이득 없음 — inner box가 이미 너무 작아 신뢰성 있는 IoU 계산 어려움.

---

## InnerCIoU ratio=0.7 — RUN_260622_02

- 코드: `bbox_iou(..., InnerCIoU=True, ratio=0.7)`
- Best epoch: 121
- 브랜치: `1-loss`

### val 결과

| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| InnerCIoU (r=1.0) | 0.525 | 0.285 | 0.403 | 0.251 |
| InnerCIoU (r=0.7) | 0.524 | 0.283 | 0.400 | 0.248 |
| Δ | -0.001 | -0.002 | **-0.003** | **-0.003** |

### 클래스별 mAP50 (val)

| class | r=1.0 | r=0.7 | Δ |
|---|---|---|---|
| pedestrian | 0.460 | 0.440 | -0.020 |
| people | 0.352 | 0.342 | -0.010 |
| bicycle | 0.203 | 0.223 | +0.020 |
| car | 0.748 | 0.746 | -0.002 |
| van | 0.394 | 0.397 | +0.003 |
| truck | 0.415 | 0.399 | -0.016 |
| tricycle | 0.212 | 0.211 | -0.001 |
| awning-tricycle | 0.254 | 0.262 | +0.008 |
| bus | 0.600 | 0.601 | +0.001 |
| motor | 0.388 | 0.384 | -0.004 |
| **all** | 0.403 | 0.400 | **-0.003** |

### 관찰
- r=1.0보다 전반 저하. InnerCIoU는 ratio 축소가 오히려 역효과.
- bicycle만 +0.020 개선. pedestrian -0.020, truck -0.016 등 주요 클래스 저하.
- **ratio 튜닝 방향으로 추가 개선 여지 없음** — InnerCIoU는 r=1.0이 최선.

---

## WIoU v3 (Wise-IoU) — RUN_262623_01 (브랜치 `1.2-loss_WIoU`)

### 결과 요약 (결론: APtiny 개선 실패, 전반 지표는 최고)

**val (ultralytics 내장, conf=0.25)**

| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| InnerSIoU r=1.0 | 0.528 | 0.285 | 0.406 | 0.253 |
| **WIoU v3** | 0.533 | **0.290** | **0.410** | 0.253 |

- val상으로는 가장 균형적 — mAP50 0.410(NWD와 동률 최고), mAP50-95 0.253(InnerSIoU와 동률 최고), **Recall 0.290 단독 최고**.

**pycocotools (test, conf=0.001, max_det=500)**

| size band | Baseline | InnerSIoU r=1.0 | **WIoU v3** |
|---|---|---|---|
| vt(2-8) | 0.0205 | 0.0149 | 0.0142 |
| **tiny/APT(8-16)** | 0.0564 | 0.0568 | **0.0557** |
| small(16-32) | 0.1191 | 0.1221 | 0.1222 |
| medium(32+) | 0.2558 | 0.2660 | **0.2668** |
| **AP@[.5:.95]** | 0.1646 | 0.1703 | **0.1708** |
| AP50 | 0.2918 | 0.3039 | **0.3089** |
| AR | 0.3168 | 0.3232 | **0.3250** |

클래스별 AP@.5:.95: pedestrian 0.1167, people 0.0590, bicycle 0.0411, car 0.4372, van 0.2169, truck 0.2009, tricycle 0.0854, awning-tricycle 0.0901, bus 0.3391, motor 0.1213.

### 분석 (노이즈 바닥 ±0.007 APtiny / ±0.010 AP50 / ±0.005 AP 기준)
- **APtiny = 0.0557**: baseline(0.0564)·InnerSIoU(0.0568) 대비 **노이즈 내 동률, 실질 개선 0**. 결정 기준(>0.0668) **미달**.
- **전반 지표(AP@.5:.95 0.1708, AP50 0.3089, AR 0.3250, APmedium 0.2668)는 6개 로스 중 최고** — 하지만 InnerSIoU 대비 차이는 전부 노이즈 내(AP@.5:.95 +0.0005, AP50 +0.0050).
- 즉 **WIoU의 focusing은 "전반 검출·recall"엔 미세하게 도움, "소형 객체(APtiny)"엔 무효**.
- 가설("focusing이 tiny가 무조건 hard-sample로 취급돼 망가지는 걸 완화")은 **APtiny 기준 기각**.

### 결론
- **IoU 회귀 로스 트랙 종료.** CIoU/InnerCIoU/InnerSIoU/ratio/NWD/WIoU 6종 모두 **APtiny를 노이즈 이상으로 못 올림** → 로스 함수는 소형 객체 병목을 못 건드림(견고한 negative result, 논문 motivation으로 활용).
- 로스 최종 후보: **WIoU v3** 또는 **InnerSIoU r=1.0** (전반 지표 동률 최고, 둘 다 노이즈 내). WIoU가 recall/AP50 미세 우위 + "dynamic focusing" 서술 깔끔.
- **짬뽕(Wise-SIoU) 진행 안 함**: 전제(WIoU focusing이 tiny에 효과)가 기각됨 → SIoU에 곱해도 노이즈. 아래 예정 항목 취소.

## ~~WIoU v3 (Wise-IoU) — 학습 진행 중 (브랜치 `1.2-loss_WIoU`)~~ ↓ 구현 상세

### 출처
> Tong, Chen, Xu, Yu, "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism" (2023), arXiv:2301.10051. (수식은 ar5iv 원문과 1:1 대조함)

### 원논문 수식 (검증 완료)

**WIoU v1 — distance attention**
```
L_WIoU_v1 = R_WIoU · L_IoU
R_WIoU    = exp( ((x − x_gt)² + (y − y_gt)²) / (W_g² + H_g²)* )
L_IoU     = 1 − IoU
```
- `(W_g, H_g)`=최소 외접박스, `*`=computational graph에서 **detach** (수렴 방해 gradient 차단)
- `R_WIoU ∈ [1, e)` — 중심 멀수록 L_IoU 증폭, 중심 일치 시 1

**WIoU v2 — 단조 focusing (v3에 흡수, 미사용)**
```
L_WIoU_v2 = ( L_IoU* / L̄_IoU )^γ · L_WIoU_v1
```

**WIoU v3 — 비단조 동적 focusing (본 실험)**
```
β = L_IoU* / L̄_IoU  ∈ [0, +∞)        # outlier degree (detach)
r = β / ( δ · α^(β − δ) )              # 비단조 focusing 계수
L_WIoU_v3 = r · R_WIoU · L_IoU
```
- 논문 권장: **α=1.9, δ=3** (AP75 54.50% 최적값)
- 의미: 품질 **중간(β≈δ)** anchor에 gradient 집중. 너무 좋은 박스/outlier 모두 down-weight
  → 소형 객체가 무조건 hard-sample로 취급돼 학습이 망가지는 현상 완화.

### 우리 코드 ↔ 논문 수식 대조

| 논문 항 | 구현 위치 | 코드 | 일치 |
|---|---|---|---|
| L_IoU = 1−IoU | `metrics.py wise_iou` | `iou=inter/union`, 호출부 `1-iou` | ✅ |
| R_WIoU 분자 (center dist², grad O) | `wise_iou` | `center_dist_sq=((Δcx)²+(Δcy)²)` (DIoU rho² 형식) | ✅ |
| R_WIoU 분모 (W_g²+H_g²)* (detach) | `wise_iou` | `c2=(cw²+ch²+eps).detach()` | ✅ |
| R_WIoU=exp(분자/분모) | `wise_iou` | `r_wiou=exp(center_dist_sq/c2)` | ✅ |
| L̄_IoU (EMA) | `BboxLoss` | `iou_mean` 버퍼, `momentum*old+(1-m)*batch_mean` | ✅ |
| β=L_IoU*/L̄_IoU (detach) | `forward` | `beta=(l_iou.detach()/iou_mean).clamp(min=eps)` | ✅ |
| r=β/(δ·α^(β−δ)) | `forward` | `r=beta/(delta*alpha**(beta-delta))` | ✅ |
| L_WIoU_v3=r·R_WIoU·L_IoU | `forward` | `wiou_loss=r*r_wiou*l_iou` | ✅ |

gradient 흐름: `R_WIoU`(분자)·`L_IoU` 통해 흐름, `r`·`β`는 detach된 재가중 계수(focusing). ✅ 논문 일치.

### 구현 노트
- `iou_mean`은 `register_buffer`. **다중 GPU(DDP)에선 GPU별 독립 갱신**(buffer sync 대상 아님) → scalar EMA라 영향 미미.
- `momentum=0.99`(배치당 1% 갱신)는 논문이 구현 디테일로 남긴 부분 → 튜닝 가능. 핵심 수식(β, r)은 그대로 구현.
- YOLO26는 E2ELoss가 `one2many`+`one2one` 두 v8DetectionLoss를 감싸므로 BboxLoss 변경이 **양쪽 모두 적용**.

### 검증 (스모크 테스트)
- `wise_iou`: 동일박스→IoU=1·R_WIoU=1 / 원거리→R_WIoU<e ✅
- BboxLoss v1/v3: loss 유한, gradient 흐름, `iou_mean` EMA 갱신(1.0→0.999…) ✅
- YOLO26s end-to-end forward+loss+backward 정상 (loss items=[box 1.42, cls 1.27, dfl 0.04]) ✅

### 학습 명령어
```bash
yolo detect train \
  model=".../ultralytics/cfg/models/26/yolo26s.yaml" \
  data=".../dataset/VisDrone/cleaning/data.yaml" \
  epochs=300 patience=50 batch=32 imgsz=640 device=6,7 workers=4 seed=0 \
  project=".../ultralytics/runs/LOSS" name="260623_Y26S_WIoU_VISDRONE"
```

### 판단 기준
| 결과 | 다음 |
|---|---|
| APtiny > 0.0568+0.01 (노이즈 초과) | 신호 → 멀티시드 확정 → 짬뽕(Wise-SIoU) 진행 |
| 노이즈 내 | IoU 로스 트랙 종료, InnerSIoU r=1.0 최종 확정 |

### 다음 단계 — 짬뽕 IoU (Wise-SIoU): **취소**
WIoU 단독이 APtiny를 노이즈 이상으로 못 올렸으므로(0.0557), focusing을 SIoU에 곱해도
tiny 개선 기대 불가 → `1.3-loss_wise_siou` **진행 안 함**. 대신 WIoU의 약점을 정조준한 SA-WIoU로 전환.

---

## SA-WIoU (Size-Aware Wise-IoU) — RUN_260624_01 (브랜치 `1.3.1-loss_sawiou`)

### 한눈에 — SA-WIoU가 뭐고 다른 IoU와 뭐가 다른가

**정의:** WIoU v3에서 "outlier 판정(β)"을 **전역 평균이 아니라 객체 크기 그룹별 평균**으로 정규화한 변형. 즉 WIoU의 *샘플 가중치 기준*만 크기 인지형으로 바꾼 것.

**IoU 계열 전체에서의 위치 — 각 loss가 "무엇을" 바꾸는가:**

| Loss | 바꾸는 대상 | 한 줄 설명 |
|---|---|---|
| IoU→GIoU→DIoU→CIoU | **기하 페널티** (겹침 → +중심거리 → +종횡비) | "박스가 얼마나 틀렸나"를 정교화 |
| SIoU | 기하 페널티 (+각도/형태 cost) | 방향성까지 반영 |
| Inner-IoU | **IoU 민감도** (박스를 ratio로 축소) | 겹침 계산 자체를 더 예민하게 |
| NWD | **거리 척도** (겹침 → 2D Gaussian) | 겹침이 없어도 연속 gradient |
| WIoU v3 | **샘플 가중치** (focusing 계수 r) | 기하는 그대로, "어떤 박스에 집중할지" 재분배 |
| **SA-WIoU** | **샘플 가중치의 기준** (outlier 판정을 크기별로) | WIoU가 작은 객체를 억누르던 편향 제거 |

→ 대부분의 IoU 변형은 **"자(尺)" 자체(기하 공식)**를 바꾼다. WIoU는 자는 그대로 두고 **"어디에 주목할지(가중치)"**를 바꾼다. SA-WIoU는 한 단계 더 — 그 주목의 **"기준 프레임"**(전역→크기별)을 바꾼다. 기하에서 가장 멀리 떨어진, WIoU의 가중치 편향을 고치는 메타 수정.

**아이디어 출처 (데이터 기반):**
1. WIoU v3 결과: 전반 지표(AP/AP50/AR) 최고인데 **APtiny만 정체**(0.0557, baseline 동급).
2. 진단: `β = L_IoU / 전역평균`인데, ⓐ 소형 객체는 잘 맞춰도 IoU가 낮아 L_IoU가 본래 높고, ⓑ 전역평균은 쉬운 대형(car/bus)이 지배 → 소형 β가 **구조적으로 부풀려짐** → anti-outlier focusing이 **가장 학습해야 할 하드 tiny(IoU≈0)를 outlier로 억누름**.
3. 수정: outlier를 **같은 크기대 안에서** 판정 → `β = L_IoU / 크기별평균`. tiny끼리 비교 → 억압 해제.

**핵심 이점:** WIoU의 **scale 편향(작을수록 억압)** 제거 → 작은 객체에 gradient 유지.
**실제 효과(정직):** APsmall(16-32px)은 전 로스 최고(0.1252)+vt 회복. 그러나 **APtiny(8-16px)는 무효** → 8-16px는 loss 가중치가 아니라 positive anchor·feature 부재(구조)가 병목임을 역으로 입증.

### 결과 (결론: APtiny 개선 실패 — 가설 기각)

**val (conf=0.25):** mAP50 0.406, mAP50-95 0.251, P 0.526, R 0.290 → WIoU(0.410/0.253)보다 소폭 낮음.

**pycocotools (test, conf=0.001):**

| size band | Baseline | WIoU v3 | **SA-WIoU** |
|---|---|---|---|
| vt(2-8) | 0.0205 | 0.0142 | 0.0167 |
| **tiny/APT(8-16)** | 0.0564 | 0.0557 | **0.0558** |
| small(16-32) | 0.1191 | 0.1222 | **0.1252** |
| medium(32+) | 0.2558 | **0.2668** | 0.2657 |
| **AP@[.5:.95]** | 0.1646 | **0.1708** | 0.1707 |
| AP50 | 0.2918 | **0.3089** | 0.3054 |
| AR | 0.3168 | 0.3250 | 0.3250 |

클래스별 AP@.5:.95: pedestrian 0.1170, people 0.0598, bicycle 0.0394, car 0.4423, van 0.2163, truck 0.1924, tricycle 0.0846, awning-tricycle 0.0954, bus 0.3414, motor 0.1182.

### 분석
- **APtiny = 0.0558**: baseline(0.0564)·WIoU(0.0557)와 노이즈 내 동률. 결정 기준(>0.0668) **미달**. **가설 기각** — size-stratified outlier 정규화가 APtiny를 풀지 못함.
- **APsmall(16-32) = 0.1252는 전 로스 중 최고**, vt도 WIoU보다 회복(0.0167). → size 정규화가 16-32px엔 미세 긍정 효과, 그러나 8-16px(APT)엔 무효.
- 전반(AP@.5:.95 0.1707, AR 0.3250)은 WIoU와 사실상 동률.
- 해석: WIoU의 anti-small bias 제거는 **중간 크기(16-32px)까지만** 도움. 진짜 tiny(8-16px)는 positive anchor 부족·feature 부재가 병목이라 **loss 재가중만으로는 한계**(= 구조 문제).

### 최종 판정
- **IoU 로스 트랙 종료 확정.** 7종(CIoU/InnerCIoU/InnerSIoU/ratio/NWD/WIoU/SA-WIoU) 모두 **APtiny를 노이즈 이상으로 못 올림**.
- 로스 최종 후보: **WIoU v3**(전반 최고) 또는 **SA-WIoU**(APsmall 최고 + size-aware 분석 서술 가치). 둘은 노이즈 내 동률.
- 다음 레버는 **구조(FPN/P2 head)** — APtiny가 실제로 움직일 곳. `../00_roadmap/SOD_ROADMAP_PAPER_PROCESS.md`.

### 동기 (WIoU 결과 분석에서 도출)
WIoU v3는 outlier degree `β = L_IoU / L̄_IoU(전역)`로 고손실 박스를 down-weight한다.
그런데 **전역 평균은 쉬운 대형 객체가 지배**하고, 소형 객체는 잘 맞춰도 IoU가 낮아(L_IoU 높음)
β가 구조적으로 커진다 → WIoU가 **하드 tiny 객체(IoU≈0)를 outlier로 보고 억눌러** APtiny 개선 실패(0.0557).

→ WIoU가 전반 지표(AP/AP50/AR)는 올리면서 APtiny만 못 올린 **메커니즘적 원인**.

### 아이디어
outlier 판정을 **크기 구간 안에서** 정규화한다.
```
WIoU :     β = L_IoU / L̄_IoU(전역)            # 대형 지배 평균
SA-WIoU :  β = L_IoU / L̄_IoU[size_bin]         # 같은 크기대 평균
```
- tiny 객체는 다른 tiny와 비교해 outlier 판정 → 하드 tiny가 더는 억눌리지 않음 → gradient 유지.
- WIoU의 bounded focusing 틀 안이라 gradient 폭발 위험 없음 (naive size-weight의 불안정 회피).

### 크기 빈 = FPN 레벨(stride)
> **중요**: BboxLoss 내부 박스는 `target_bboxes / stride_tensor`로 들어와 **stride(그리드) 단위**다.
> 따라서 sqrt(area)로 빈을 나누면 레벨이 섞여 물리 크기 분리가 안 됨.
> → **anchor의 stride로 빈 분할**(= FPN 레벨 = scale). YOLO26 P3/P4/P5 = stride 8/16/32:

| bin | stride | 의미 |
|---|---|---|
| 0 | 8 (P3) | 소형 |
| 1 | 16 (P4) | 중형 |
| 2 | 32 (P5) | 대형 |

`iou_mean` 버퍼를 scalar → `[3]`으로 확장, 각 박스의 stride로 빈 선택해 bin별 EMA 갱신.

### 구현 (WIoU 대비 변경점)
| 위치 | 변경 |
|---|---|
| `BboxLoss.__init__` | `wiou_size_aware=True`, `wiou_stride_edges=(16,32)` 추가; `iou_mean` 버퍼 `[3]`(빈 수) |
| `BboxLoss.forward` | stride로 `bin_idx` 계산 → bin별 L̄_IoU EMA 갱신 → `β = L_IoU / L̄_IoU[bin_idx]` |
| `v8DetectionLoss` | `BboxLoss(..., use_wiou=True, wiou_version=3, wiou_size_aware=True)` |
| `metrics.py wise_iou` | 변경 없음 (WIoU와 공유) |

- `size_aware=False`면 `iou_mean`이 `[1]`로 줄어 vanilla WIoU와 동일 (하위호환).

### 검증 (스모크 테스트)
- 3-bin `iou_mean`이 scale별로 독립 갱신(예: [0.997, 0.995, 0.997]) ✅
- BboxLoss: loss 유한, gradient 흐름, vanilla(`size_aware=False`, n_bins=1) fallback 정상 ✅
- YOLO26s end-to-end forward+loss+backward 정상, one2many/one2one 양쪽 `size_aware=True n_bins=3` ✅

### 학습 명령어
```bash
yolo detect train \
  model=".../ultralytics/cfg/models/26/yolo26s.yaml" \
  data=".../dataset/VisDrone/cleaning/data.yaml" \
  epochs=300 patience=50 batch=32 imgsz=640 device=6,7 workers=4 seed=0 \
  project=".../ultralytics/runs/LOSS" name="260624_Y26S_SAWIoU_VISDRONE"
```

### 판단 기준
| 결과 | 다음 |
|---|---|
| APtiny > 0.0568 + 0.01 (노이즈 초과) | **성공** → 멀티시드 확정, 빈 경계/개수 ablation |
| APtiny 노이즈 내 but 전반 ≥ WIoU | WIoU와 동급 → WIoU 최종 채택 (SA는 부록) |
| 전반 < WIoU | size 정규화가 역효과 → WIoU 최종 채택 |

### 솔직한 승산
APtiny 노이즈 초과 ~35-40%. WIoU의 anti-small-object bias를 직접 제거하는 설계라 근거는 가장 강함.
실패해도 WIoU(전반 지표 개선)가 안전망으로 남음.

### 개선 (v2): stride 빈 → 물리 픽셀 크기 빈

**v1의 약점(진단):** v1은 **anchor의 stride(FPN 레벨)로 3-bin** 분할했다. 그런데 P3(stride 8) 한 레벨이 물리적으로 **약 8~64px를 모두 담당**한다 → bin 0 안에 8px와 50px가 섞임. 결과적으로 **8-16px 객체가 자기 bin 안에서도 여전히 상대적 고손실 outlier**로 남아 억압이 덜 풀림. → APsmall(16-32)은 개선됐지만 APtiny(8-16)는 정체된 이유.

**우아한 해결:** 빈 기준을 stride → **복원한 물리 픽셀 √area**로 바꾸고, 경계를 **평가 band와 정렬(8/16/32px → 4-bin: vt/tiny/small/medium)**.
- BboxLoss 내부 박스는 stride 단위이므로 물리 크기 = `√(w_grid·h_grid) × stride[fg]`로 복원.
- 이제 **8-16px가 독립 정규화 그룹**을 가짐 → 더 큰 객체에 β가 부풀려지지 않음 → 하드 tiny 억압 추가 제거.
- 함수 형태 가정 없음(빈별 EMA 그대로) + 평가 지표와 정렬 → 해석/서술 깔끔.

**정직한 기대치:** v1보다 8-16px 정규화가 정밀해져 APtiny에 v1보다 유리할 수 있으나, **구조 병목(positive anchor·feature 부재)이 상한**이라 큰 도약은 기대난. APsmall 추가 개선 + APtiny 소폭 가능성. (학습 후 검증)

**구현:** `wiou_stride_edges=(16,32)`(stride) → `wiou_size_edges=(8,16,32)`(픽셀 √area), forward에서 `size_px = √(w_grid·h_grid)·strd_fg`로 빈 할당. 브랜치 `1.3.1-loss_sawiou`.

#### v2 결과 (RUN_260624_02) — **의도와 정반대, 역효과 (기각)**

**val:** mAP50 0.403, mAP50-95 0.253, P 0.524, R 0.286 (v1 0.406보다 낮음).

**pycocotools (test):**

| size band | baseline | SA-v1(stride) | **SA-v2(픽셀)** |
|---|---|---|---|
| **vt(2-8)** | 0.0205 | 0.0167 | **0.0236** (역대 최고 ↑) |
| **tiny/APT(8-16)** | 0.0564 | 0.0558 | **0.0538** (↓, baseline 이하) |
| small(16-32) | 0.1191 | 0.1252 | 0.1222 |
| medium(32+) | 0.2558 | 0.2657 | 0.2639 |
| **AP@[.5:.95]** | 0.1646 | 0.1707 | 0.1692 (↓) |
| AP50 | 0.2918 | 0.3054 | 0.3006 |

**분석 (왜 역효과인가):** 빈을 잘게 나누니 vt(2-8px)가 독립 그룹을 얻어 **억압이 풀리며 gradient가 vt로 쏠림**(vt 0.0236, 전 로스 최고). 그러나 2-8px는 feature가 거의 없어 **학습해도 거의 안 됨** → 고정된 모델 용량을 near-hopeless vt에 빼앗기며 **정작 헤드라인 8-16px가 더 떨어짐**(0.0538). 전반(AP@.5:.95)도 v1보다 하락.

**결론:** "더 정밀한 size 정규화"가 더 좋은 게 아니다. loss 재가중은 **고정 gradient 예산을 크기대 사이에서 재분배**할 뿐 — 더 잘게 나누면 가장 작고 안 풀리는 곳(vt)으로 새어나가 오히려 손해. **SA-WIoU v2 기각.** 이는 "소형 객체 병목은 loss로 못 푼다(=구조 문제)"를 **더 강하게 입증**.

> 솔직 평: v2는 우아한 해결이 아니었다. EMA·빈·하이퍼파라미터가 늘어난 **국소 땜질**이었고 결과도 나빴다. 진짜 우아한 해결은 loss 재가중이 아니라 **구조(P2 head/고해상 feature)**. → loss 트랙 **완전 종료**, 구조 트랙으로 전환.

---

## 구현 위치

| 파일 | 변경 내용 |
|---|---|
| `ultralytics/utils/metrics.py` | `bbox_iou`에 `InnerCIoU`, `InnerSIoU` 플래그 (`1-loss`) / `wasserstein_nwd()` (`1.1-loss_nwd`) / `wise_iou()` (`1.2-loss_WIoU`) |
| `ultralytics/utils/loss.py` | `BboxLoss.__init__`에 `use_nwd/nwd_C`(NWD), `use_wiou/wiou_version/alpha/delta/momentum`+`iou_mean` 버퍼(WIoU), `wiou_size_aware/wiou_stride_edges`+멀티빈 `iou_mean`(SA-WIoU); `forward` 분기 |
| `ultralytics/utils/tal.py` | 변경 없음 (anchor 할당은 CIoU 유지, 전 브랜치 공통) |

브랜치 구조:
- `1-loss`: InnerCIoU, InnerSIoU 구현
- `1.1-loss_nwd`: NWD 구현 (main 기반)
- `1.2-loss_WIoU`: WIoU v1/v3 구현 (`1-loss` 기반)
- `1.3.1-loss_sawiou`: SA-WIoU 구현 (`1.2-loss_WIoU` 기반) — size-stratified outlier 정규화
- ~~`1.3-loss_wise_siou`~~: 취소 (WIoU가 APtiny 개선 못 함)
