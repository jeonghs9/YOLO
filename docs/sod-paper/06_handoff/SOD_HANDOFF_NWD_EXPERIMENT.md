# SOD 논문 — NWD 단독 검증 실험 인계 문서

> 이 문서는 연구 방향 논의(Claude 채팅)에서 Claude Code로 작업을 인계하기 위한 문서입니다.
> 작업 디렉토리: `/home/hsjeong/workspace/SOD-PAPER/`

---

## 1. 연구 전체 맥락 (요약)

- **목표**: SOD(Small Object Detection) 논문, MDPI 게재 목표
- **베이스 모델**: YOLO26s/n (Ultralytics 8.4.6, editable install)
- **데이터셋**: VisDrone, DOTA(HBB, tiled), AI-TOD
- **연구 방향 (방향 B 확정)**: 구조 개선(FPN shallow-deep semantic gap 해소 모듈) + 학습 개선(size-adaptive loss) 을 함께 제안하는 복합 기여 논문
- **현재 단계**: 구조 모듈(FPN) 작업 전, **Loss 설계를 먼저 고정**하는 단계. 이유: ablation 매트릭스(`Baseline → +FPN모듈 → +Loss모듈 → +Full`)에서 Loss가 baseline 위에서 먼저 깨끗하게 검증되어야 추후 FPN 모듈과의 상호작용을 분리해서 볼 수 있음.

### Loss 트랙 진행 상황 (`../02_loss/SOD_LOSS_EXPERIMENTS.md` 참조)

| 실험 ID | Loss | 모델 | 데이터셋 | mAP50 | mAP50-95 | P | R | 상태 |
|---|---|---|---|---|---|---|---|---|
| BASELINE | CIoU | YOLO26s | VisDrone | 0.396 | 0.246 | 0.517 | 0.283 | 완료 |
| RUN_260617_03 | InnerCIoU (ratio=1.0) | YOLO26s | VisDrone | 0.403 | 0.251 | 0.525 | 0.285 | 완료 |
| RUN_260618_01 | InnerSIoU (ratio=1.0) | YOLO26s | VisDrone | 0.406 | 0.253 | 0.528 | 0.285 | 완료 |
| **NEW** | **NWD (단독 적용)** | YOLO26s | VisDrone | — | — | — | — | **이번 작업 대상** |

### Baseline 크기별 AP (pycocotools, test split, conf=0.001, max_det=500) — `../05_eval/SOD_EVAL_PIPELINE.md` 참조

YOLO26s (test):
| band | AP | AR |
|---|---|---|
| vt(2-8px) | 0.0205 | 0.0583 |
| **tiny/APT(8-16px)** | **0.0564** | 0.1682 |
| small(16-32px) | 0.1191 | 0.2925 |
| **medium(32px+)** | **0.2558** | 0.4415 |
| cocoS(<32) | 0.0845 | 0.2274 |
| cocoM(32-96) | 0.2441 | 0.4302 |
| cocoL(>96) | 0.3342 | 0.5113 |

클래스별 AP@[.5:.95] (작은 객체일수록 낮음 — bicycle 0.0337, people 0.0556 / 큰 객체 — car 0.4379, bus 0.3394)

---

## 2. 이번 작업의 목적 — NWD 단독 적용 검증

### 연구 가설

> CIoU 계열 IoU loss는 박스가 작을수록 동일한 픽셀 단위 위치 오차가 IoU 값을 훨씬 크게 흔든다 (위치 민감도 문제). NWD(Normalized Wasserstein Distance)는 박스를 2D Gaussian 분포로 모델링하여 IoU=0인 비겹침 상태에서도 연속적인 gradient를 제공하므로 이 문제를 완화할 수 있다.
>
> 단, NWD를 **모든 크기의 객체에 균일하게 적용**하면 이미 잘 맞는 큰 객체의 학습 신호까지 왜곡되어 큰 객체 성능이 저하될 가능성이 있다. 이것이 검증되면 "크기에 따라 적응적으로 NWD와 IoU를 혼합하는 size-adaptive loss"의 설계 동기가 정당화된다.

### 이번 실험에서 검증해야 할 것 (성공 조건)

```
가설이 맞다면 NWD 단독 적용 시 다음과 같이 나와야 함:

  APtiny(8-16px):   BASELINE 0.0564 → NWD 결과가 더 높음 (개선)
  APmedium(32px+):  BASELINE 0.2558 → NWD 결과가 더 낮음 (저하)

  클래스 레벨에서도:
  bicycle/people/tricycle (작음) → 개선
  car/bus (큼) → 저하 또는 정체

만약 두 구간이 같은 방향으로 움직이면(둘 다 개선/둘 다 저하):
→ "균일 적용의 부작용" 가설 기각
→ size-adaptive 결합의 동기가 약해짐 → 연구 방향 재논의 필요
```

이 결과가 이후 size-adaptive loss 설계(NWD + Inner-IoU 결합 등)의 핵심 근거 자료가 되므로, **크기별 AP까지 반드시 측정**해야 함 (ultralytics 내장 val의 mAP50/mAP50-95만으로는 검증 불가).

---

## 3. NWD 수식 (구현 시 참고)

원 논문(Wang, Xu, Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection")의 정의를 따른다.

### 3.1 박스를 2D Gaussian 분포로 모델링

bounding box `(cx, cy, w, h)`를 다음 2D Gaussian 분포로 근사:

```
μ = (cx, cy)
Σ = [[(w/2)^2, 0], [0, (h/2)^2]]
```

### 3.2 두 Gaussian 간 2nd-order Wasserstein Distance

박스 A = (cx_a, cy_a, w_a, h_a), 박스 B = (cx_b, cy_b, w_b, h_b)에 대해:

```
W2^2(Na, Nb) = || (cx_a, cy_a, w_a/2, h_a/2) - (cx_b, cy_b, w_b/2, h_b/2) ||_2^2
```

(對角 공분산 가정 시 closed-form으로 위와 같이 단순화됨)

### 3.3 Normalized Wasserstein Distance (NWD)

```
NWD(Na, Nb) = exp( -sqrt(W2^2(Na, Nb)) / C )
```

- `C`는 데이터셋에 따른 정규화 상수 (원 논문은 평균 box size 등으로 경험적 설정. AI-TOD 논문 기준 C=12.7 사용 사례 있음 — VisDrone에 맞게 재산정 필요할 수 있음, 아래 4번 항목 참고)
- NWD 값은 [0, 1] 범위 → IoU와 동일한 스케일로 loss에 사용 가능

### 3.4 Loss 적용

```
L_NWD = 1 - NWD(pred_box, target_box)
```

기존 `BboxLoss.forward()`의 IoU 기반 loss(`1 - iou`)를 이 값으로 교체(단독 적용 시).

---

## 4. 구현 가이드라인

### 4.1 구현 위치 (기존 `InnerCIoU` 구현 패턴 따름 — `../02_loss/SOD_LOSS_EXPERIMENTS.md` 참조)

| 파일 | 변경 내용 |
|---|---|
| `ultralytics/utils/metrics.py` | `bbox_iou()`와 별도로 `wasserstein_nwd(box1, box2, C=...)` 함수 신규 추가 권장 (NWD는 겹침 기반이 아니므로 기존 `bbox_iou` 시그니처에 억지로 끼워 넣지 말 것) |
| `ultralytics/utils/loss.py` | `BboxLoss.forward()`에 `NWD=True` 플래그 분기 추가. 기존 `CIoU=True`, `InnerCIoU=True` 분기와 동일한 패턴 유지 |
| `ultralytics/utils/tal.py` | **변경 없음** (기존 InnerCIoU 때와 동일하게 anchor 할당(TAL)은 CIoU 유지 — NWD는 loss 계산에만 적용하고 label assignment는 건드리지 않음. 이 결정은 기존 실험과의 일관성을 위해 고정) |

### 4.2 정규화 상수 C 결정 필요

- 원 논문(NWD-RKA, AI-TOD 기준)은 C=12.7을 사용하나 이는 AI-TOD 데이터셋의 평균 객체 크기에 맞춘 값.
- **VisDrone은 평균 객체 크기가 다르므로 C를 그대로 쓰면 안 됨.** 다음 중 하나로 결정:
  1. VisDrone train set의 평균 box 크기(width, height 평균 또는 area sqrt 평균)를 계산해서 C로 사용 (원 논문 방법론과 동일한 절차)
  2. 1차로는 문헌값(C=12.7 또는 C=8 등 다른 논문 값)으로 빠르게 돌려보고, 추후 ablation으로 C값 민감도 확인
- **권장: 우선 절차 1번(VisDrone 평균 box 크기 기반 계산)으로 C를 산정**하고 실험 시작. 코드에 C 산정 스크립트나 계산 근거를 주석으로 남길 것 (논문 재현성을 위해 필수).

### 4.3 학습 조건 (기존 baseline과 완전히 동일하게 고정)

```
model: yolo26s
data: VisDrone (dataset/VisDrone/cleaning/data.yaml)
epochs: 300, patience: 50, batch: 32, imgsz: 640, seed: 0
optimizer: SGD 계열 (muSGD — 기존 baseline과 동일하게)
mosaic: 1.0, copy_paste: 0.0, multi_scale: 0.0  (기존 baseline과 동일 — 절대 변경 금지)
```

> 기존 baseline/InnerCIoU 실험과 데이터·증강 조건이 다르면 비교 자체가 무효가 됨. `../02_loss/SOD_LOSS_EXPERIMENTS.md`의 BASELINE/RUN_260617_03 행과 정확히 같은 조건 사용.

### 4.4 평가 — 반드시 두 가지 모두 수행

1. **ultralytics 내장 val** (val split, conf=0.25) — 기존 `loss.md` 표 포맷과 맞춰서 기록 (P, R, mAP50, mAP50-95, 클래스별 mAP50)
2. **pycocotools 크기별 AP** (`../05_eval/SOD_EVAL_PIPELINE.md`의 `eval/eval_size_ap.py` 사용, test split, conf=0.001, max_det=500, imgsz=640)
   ```bash
   python eval/eval_size_ap.py \
     --weights ultralytics/runs/<NWD실험 weights 경로>/weights/best.pt \
     --data dataset/VisDrone/cleaning/data.yaml --split test \
     --imgsz 640 --device <GPU번호> --max-det 500 --conf 0.001 \
     --tag NWD_VISDRONE
   ```
   → vt/tiny/small/medium 구간별 AP, AR 전부 기록 필요. **이게 이번 실험의 핵심 결과물.**

---

## 5. 실험 후 기록 포맷 (`../02_loss/SOD_LOSS_EXPERIMENTS.md`에 추가할 내용)

다음 정보를 `../02_loss/SOD_LOSS_EXPERIMENTS.md`에 NWD 실험 섹션으로 추가:

1. val 결과 표 (BASELINE, InnerCIoU와 같은 행 포맷)
2. 클래스별 mAP50 표
3. **크기별 AP 비교표** (BASELINE vs NWD, vt/tiny/small/medium 전 구간)
4. 사용한 정규화 상수 C 값과 산정 근거
5. 가설 검증 결과 요약 (APtiny 개선 여부, APmedium 저하 여부 — 명시적으로 O/X 기록)

---

## 6. 이번 작업 범위 — 명확히 할 것

- **포함**: NWD 단독 적용 구현 + VisDrone 1개 데이터셋 학습 + 2종 평가(val + 크기별 AP) + 결과 기록
- **포함 안 됨 (다음 단계)**:
  - NWD와 Inner-IoU/SIoU의 결합(size-adaptive loss) 설계 — 이번 실험 결과를 본 후 진행
  - FPN 구조 모듈 개발 — 별도 트랙, 이번 작업과 무관
  - DOTA/AI-TOD에서의 NWD 검증 — VisDrone 결과로 가설 1차 확인 후 확장 여부 결정
  - InnerSIoU 실험 — 완료 (RUN_260618_01, mAP50=0.406, mAP50-95=0.253)

---

## 7. 외부 코드 사용 여부 — 결정 사항

**외부 GitHub repo의 NWD 구현을 가져오지 않고, 본 문서의 수식(3절)을 기반으로 기존 코드베이스 패턴(`InnerCIoU`와 동일한 방식)에 맞춰 직접 구현한다.**

이유:
- 기존 `bbox_iou()` / `BboxLoss.forward()` 플래그 분기 패턴과 통합성 유지 (추후 size-adaptive 결합 시 필요)
- 외부 repo는 ultralytics 버전/클래스 구조가 달라 이식 비용 및 의존성 충돌 위험
- C값 산정 등 세부 가정을 직접 추적 가능해야 논문 재현성 확보 가능
- 그러나 NWD 원 논문에서 주장하는 수식은 그대로 구현한다. 수식이 완성이 되면, 원논문을 확인하여 수식과 일치하는지 비교분석을 진행한다.

---

## 8. 작업 완료 후 보고해야 할 것

1. 구현한 코드 위치 및 핵심 diff
2. 사용한 C값과 산정 방법
3. val 결과 (P/R/mAP50/mAP50-95, 클래스별)
4. 크기별 AP 결과 (vt/tiny/small/medium, BASELINE 대비)
5. 가설 검증 결론 (APtiny 개선 O/X, APmedium 저하 O/X) — 이 결론에 따라 다음 연구 단계가 갈라지므로 반드시 명시적으로 정리
