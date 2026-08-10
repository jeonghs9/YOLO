# SOD 논문 연구 흐름

목표: Small Object Detection 개선 논문, MDPI 게재.  
베이스: YOLO26s/n (ultralytics 8.4.6, editable install), 데이터셋 VisDrone / DOTA / AI-TOD.

> **⚡ 현재 유효 스토리 (2026-07-09) — 이것만 유효, 아래 옛 서사는 히스토리**
> P2 로 고해상 detection head 를 확보한 뒤, small 병목을 **recall(feature 보존 = SPD-Conv)** 과
> **precision(quality ranking = SAQ)** 으로 **분해**한다. fusion/loss/refine/assignment probe 는 **negative ablation**.
> ⚠ 아래 §"구안(2026-06-25)" 이하 "FPN gap module + loss" 서사는 **반증·폐기(기록용)**. 혼동 주의.

---

## 연구 방향 (확정: 방향 B)

**구조 개선(FPN shallow-deep semantic gap 해소 모듈) + 학습 개선(localization loss)** 복합 기여 논문.

**Ablation 매트릭스:**
```
Baseline → +FPN모듈 → +Loss → +Full (FPN+Loss)
```
(순서 변경: loss 단독이 baseline에서 무효임을 8종으로 확인 → 구조 먼저, loss는 구조 위 refine.)

---

## ★ 현재 논문 스토리 (2026-07-09 최신 — 코덱스 인수인계)

- **① naive-P2 고해상 헤드** — 확정 양성(핵심 기여). 유지.
- **② small(16-32) 개선 — 2갈래 진행**:
  - **진단(실측, `saq_premise_diag.py`)**: oracle q-rescore `cls·IoU^γ` → small AP **0.140→0.260(+0.12)**,
    AP50 +0.19 (**ranking headroom 큼**). 단 **AR 0.3221 불변** = **recall 은 assignment/rescore 로 못 뚫는 벽**.
    cls-IoU misalignment small Spearman **0.501** < medium 0.637.
  - **갈래 A (precision/ranking) = SAQ**: VFNet/GFL **차용** quality head(reg feature→q, `score=cls·q^γ`).
    설계 v6 완성, `SAQDetect` head 구현 + smoke 무해성 통과, **L_q 미구현 → ⏸ 보류**(`3-saq-quality` 커밋).
    설계 `../03_modules/SOD_MODULE_SAQ.md`.
  - **갈래 B (recall/feature) = SPD-Conv ★현재 진행**: downsample 정보손실 허점을 **space-to-depth 로 보존**
    (기존 차용, 코드=Focus 재사용). backbone-only + full(backbone5+neck3) yaml, **build 통과, 본학습 대기(사용자 CLI)**.
    `../03_modules/SOD_MODULE_CANDIDATES.md`.
- **negative ablation** (정직한 기여): tiny(8-16 정보한계) · fusion 3방향(SGI/GSA/AFFG) · P2Refine · WIoU ·
  **NWD assignment probe(2026-07-09, AR 불변으로 negative)**.
- **지표**: mAP50/mAP50-95 + COCO APs/APm + small AP50/75/AR.

## 코덱스 인수인계 체크포인트 (2026-07-09)

- **핵심 발견**: small 병목이 **recall(AR 0.32 벽) + precision(ranking headroom +0.12)로 분리**됨. probe(assignment)로
  recall 안 열림 → recall 은 **feature 정보(SPD-Conv)**, precision 은 **quality ranking(SAQ)** 로 각각 공략(상보).
- **현재 액션**: SPD-Conv 본학습(사용자 CLI, `260709_Y26S_P2_SPDfull_VISDRONE` 등) → naive-P2 대비 small AR/AP 판정.
- **보류**: SAQ(L_q 구현 남음) — SPD 결과 후 재개.
- **브랜치**: `2.4-probe_nwd_assign`(probe) · `3-saq-quality`(SAQ head) · `4-spdconv`(SPD, 현재) · `2-fpn`(naive-P2 베이스).
- **코덱스 답 반영 (2026-07-09)**:
  - ① **backbone-only 먼저** — 가설(downsample 정보손실) 직접검증. full 은 neck+용량 교란("SPD 좋다"vs"용량 증가" 반박).
    ablation 순서: **P2 → +SPD-backbone → +SPD-full**(backbone 이 AR/AP 올렸을 때만 full 확장).
  - ② recall 벽 "feature 한계" **단정 금지** — 현 실험군서 assignment/rescore 로 AR 안 열림 = feature/representation 볼
    근거일 뿐. 다른 가능성(conf/maxDet·class confusion·label/occlusion·head capacity) 남음. SPD 실패해도 "SPD 로는 못 건드림".
    **필수 로그**: small AR·AP50·matched/unmatched count·class별 recall·**후보 recall vs 최종 recall 분리**·P2/P3 level 분포.
    ⚠ **내 정정(코덱스에 회신)**: 우리 YOLO26-p2 는 **end2end(NMS-free, one2one head)** → 코덱스의 "NMS 전/후 recall"은
    그대로 안 맞음. 대신 **one2many(assigner topk10 후보) recall vs one2one(inference 최종) recall** 로 분리(+conf/maxDet 영향).
  - ③ 결합 스토리 **타당(조건부)**: SPD 단독 small AR/AP50↑ AND SAQ 단독 AP/AP75↑·FP↓ AND 무충돌. SPD 가 AR 못 올리면
    → SPD=negative ablation, SAQ 중심으로. **판정 핵심 = AP 아니라 small AR 이 실제 움직이는가.**

---

## 구안 (2026-06-25 — fusion 3방향 negative 로 **반증·폐기**, 히스토리 보존)

> ⚠ 아래 "FPN shallow-deep gap 모듈(기여1) + loss 시너지" 원안은 반증됨. **기록용으로만** 남김.

### 문제 정의 — 소형 객체 검출 실패의 **두 독립 원인**

| 원인 | 정체(메커니즘) | 해결 도구 | 우리 근거 |
|---|---|---|---|
| ① **정보 부재** | stride-8에서 8px 객체 = feature map ~1px → 인식할 feature·붙을 positive anchor 부재. CNN 백본은 대형 객체용으로 aggressive downsampling(stride 32). | **FPN 갭 모듈** (고해상 + 의미 주입) | pycocotools vt/tiny AP가 구조적으로 바닥 |
| ② **위치 신뢰성** | feature가 있어도 작은 박스는 1~2px 어긋남에 IoU 급락 → 앵커와 높은 IoU 달성 난, 정밀 회귀 불안정 → 위치 신뢰성↓ → 전체 탐지 정확도↓ | **회귀 loss** | loss 8종 실험 |

### 핵심 논거 — 왜 둘 다 필요한가 (시너지)
- **loss 8종(CIoU/InnerCIoU/InnerSIoU/NWD/WIoU/SA-WIoU 등)이 baseline에서 APtiny를 분산 이상 못 올림** → ②를 풀려 했으나 ①(정보 부재)이 먼저 막고 있었기 때문. **맞출 feature가 없으면 정밀 회귀 loss는 무의미.**
- **두 원인은 직교(orthogonal)**: 모듈은 "정보를 만들고", loss는 "그 정보 위에서 정밀히 맞춘다."
- → **모듈이 ①을 풀어야 비로소 loss(②)가 효과를 가짐** (상보적). 이것이 복합 기여의 정당화.

### 기여 분업
- **기여 1 (구조):** FPN shallow-deep semantic gap 해소 + 미세특징 보존 모듈.
  - 얕은층=위치 정확/의미 빈약, 깊은층=의미 풍부/위치 흐림 → 갭 완화 + detail 보존.
  - **novelty 경계**: PAN(이미 YOLO 내장)·BiFPN·AugFPN·CARAFE와 차별되는 구체 메커니즘 필요 (단순 "갭 줄임"은 기여 아님).
- **기여 2 (학습):** 소형 객체 위치 신뢰성 loss — 구조 위에서 ②를 refine.
  - loss 8종 연구 = "loss 단독 불가" motivation + 최종 후보(WIoU/SIoU 계열) 재사용.

### Ablation (기여 분리 입증)
```
baseline → +FPN모듈 → +loss → +둘다
```
각 단계 **vt/tiny/small/medium 구간별 AP**로 측정 → "모듈이 ①, loss가 ②, 둘이 시너지"를 수치로 분리.

### 정직한 리스크 / 미검증 가정
- "loss가 모듈 위에서 도움"은 **합리적 가설(SOD 문헌 뒷받침)이나 우리 데이터로 미검증** → ablation에서 검증 대상.
- 구조 모듈의 novelty가 기존 FPN 변형 대비 약하면 incremental 위험 → 빈틈 분석 선행 필요.
- 모든 "best"/"개선" 주장은 노이즈 바닥(APtiny ±0.007) 때문에 **multi-seed 필수**.

---

## 완료 단계

### Phase 0 — 환경 및 데이터셋 구축 (완료)

- ultralytics 8.4.6 editable install (`SOD-PAPER/ultralytics/`)
- VisDrone cleaning (10클래스, train/val/test)
- DOTA v1.5 HBB tiled 전처리 (16클래스, crop_size=1024, gap=200)
- AI-TOD cleaning (8클래스)
- 개인 repo: `jeonghs9/SOD-PAPER` push 구조 확립

### Phase 1 — Baseline 학습 (완료)

모든 결과: `../01_datasets/SOD_DATASET_VISDRONE.md`, `../01_datasets/SOD_DATASET_DOTA.md`, `../01_datasets/SOD_DATASET_AITOD_RESULTS.md` 참조.

| 모델 | 데이터셋 | mAP50 | mAP50-95 |
|---|---|---|---|
| YOLO26s (muSGD) | VisDrone val | 0.396 | 0.246 |
| YOLO26n (muSGD) | VisDrone val | 0.377 | 0.232 |
| YOLO11s (SGD) | VisDrone val | 0.405 | 0.253 |
| YOLO11n (SGD) | VisDrone val | 0.365 | 0.224 |
| YOLO26s (muSGD) | DOTA val | 0.604 | 0.391 |
| YOLO26n (muSGD) | DOTA val | 0.581 | 0.371 |
| YOLO11s (SGD) | DOTA val | 0.616 | 0.405 |
| YOLO11n (SGD) | DOTA val | 0.593 | 0.375 |
| YOLO26s (muSGD) | AI-TOD val | 0.570 | 0.299 |
| YOLO26n (muSGD) | AI-TOD val | 0.493 | 0.238 |

pycocotools 크기별 AP (VisDrone test, YOLO26s): APtiny(8-16px)=0.0564, APsmall=0.1191, APmedium=0.2558.

### Phase 2 — Loss 트랙 (완료·종료, 2026-06-24)

전체 결과: `../02_loss/SOD_LOSS_EXPERIMENTS.md`, `../04_results/SOD_RESULT_AP_SIZE.md` 참조. YOLO26s + VisDrone 고정. **7종 검증 완료.**

| 실험 | Loss | val mAP50 | pycoco APtiny | pycoco AP@.5:.95 | 브랜치 |
|---|---|---|---|---|---|
| BASELINE | CIoU | 0.396 | 0.0564 | 0.1646 | main |
| RUN_260617_03 | InnerCIoU r=1.0 | 0.403 | 0.0632* | 0.1697 | `1-loss` |
| RUN_260618_01 | InnerSIoU r=1.0 | 0.406 | 0.0568 | 0.1703 | `1-loss` |
| RUN_260622_01/02 | Inner r=0.7 | 0.400~0.406 | — | — | `1-loss` |
| RUN_260619_01 | NWD C=14.5 | 0.410 | 0.0498 | 0.1570 | `1.1-loss_nwd` |
| RUN_262623_01 | WIoU v3 | 0.410 | 0.0557 | **0.1708** | `1.2-loss_WIoU` |
| RUN_260624_01 | SA-WIoU | 0.406 | 0.0558 | 0.1707 | `1.3.1-loss_sawiou` |

\* InnerCIoU r=1.0 ≡ CIoU(수학적 동일)인데 baseline과 0.0068 차 → **노이즈 바닥 ±0.007**.

**Loss 트랙 최종 결론:**
- **APtiny를 노이즈(±0.007) 이상 올린 로스 0개.** → 로스 함수는 소형 객체 병목을 못 건드림 (견고한 negative result).
- 전반 지표(AP@.5:.95/AP50/AR) 최고는 WIoU v3 (AP50 +0.017 vs baseline, 노이즈 초과) — 논문용 로스 기여로 활용 가능 (멀티시드 확정 필요).
- SA-WIoU: WIoU의 anti-small bias 제거 → APsmall(16-32) 최고(0.1252), 그러나 APtiny(8-16)는 무효. loss 재가중은 16-32px까지만 닿음.
- **로스 최종 후보: WIoU v3 또는 SA-WIoU.** 다음 레버는 구조(FPN/P2 head).

---

## Phase 3 — 구조 트랙 (P2 fusion 모듈) — **종료 (2026-07-02): fusion 계열 negative**

### 확정된 결과
- **naive-P2 (RUN_260625_01)**: P2/4 헤드 추가. AP@.5:.95 0.1646→**0.1841**(+0.0195, 노이즈 초과) → **구조(고해상 헤드)는 확실한 이득.** (상세 `../04_results/SOD_RESULT_AP_SIZE.md`)
- **tiny matched-IoU 진단**: tiny 실패 = **두 겹** — 검출(AP50/recall) + 정밀도(AP75, 찾은 것도 평균 IoU 0.65, 기하 한계). naive-P2는 recall↑(0.168→0.184)이나 **matched IoU 정체(0.655→0.662)** → 찾긴 더 찾아도 정확도는 그대로.
- **SGI(Semantic-Guided Injection) 5변형 — negative** (`../03_modules/SOD_MODULE_SGI.md`, 브랜치 `2-fpn`): V0(attn만)/V3noattn(소스만)/V1~V3(둘다). **소스 주입만(V3noattn)=naive-P2** → 의미 주입 무효(자기 대조군이 반박). 이유: P2(L19)는 이미 top-down PAN으로 의미 보유 → 추가 주입 중복.
- **GSA(Guided Sharp Attention) — negative** (`../03_modules/SOD_MODULE_GSA.md`, 브랜치 `2.2-fpn_gsa`, RUN_260701_01): tiny 0.0631=naive-P2와 동일, **γ=−0.91(모델이 빼는 방향)**. ⚠️ 코덱스 정정: 코드는 nearest 업샘플 **후** local attention(=coarse prior 재혼합, native detail 복원 아님) → 실패와 일치.

> **Phase 3 결론:** **"깊은 의미를 P2로 가져오는 fusion 모듈" 계열은 대조군까지 포함해 종결.** 병목은 P2 feature 표현/융합이 아님(갭은 top-down이 이미 메움). **코덱스·Claude 두 AI 합의.** SGI/GSA는 논문에서 **negative ablation**(fusion은 tiny 무효 → 구조 기여는 P2 헤드 자체)으로 활용.

### 다음 구조 모듈을 한다면 (Phase 3 교훈)
- deep→P2 주입 ❌. 유망 신호: **SGI V0(P2 self-attention만)이 tiny 최고** → **P2 self-refinement / foreground sharpening / backbone-side 고해상 강화** 방향. (HRNet식 multi-resolution backbone은 정석이나 규모 큼.)

---

## Phase 4 — assignment·해상도·기하 트랙 (2026-07-02~, 코덱스 검토 반영)

병목이 fusion이 아님이 확정 → 레버를 **학습 신호(누구를 학습)·입력 해상도·bbox 기하**로 이동. (assignment = 모듈도 loss도 아닌 **제3축**: `tal.py` 배정 규칙, 파라미터·추론비용 0.)

> **⚠️ 정정 (2026-07-02): STAL(size-adaptive assignment)은 8.4.6에 이미 존재·활성.** `tal.py:302-308 select_candidates_in_gts`: GT w/h < `stride[0]`이면 박스를 `stride[1]`로 확대 → 극소 GT도 positive 후보 포함. `TaskAlignedAssigner(stride=self.stride.tolist())`로 실제 stride 전달. P2 모델(stride 4/8/16/32)은 **<4px(주로 vt 2-8px)** 확대. → **"assignment 미시도" 주장 철회**: 후보 assignment 바닥은 이미 처리됨. 이것이 loss/fusion이 tiny를 못 올린 이유의 일부(쉬운 이득은 이미 baseline에 반영).

### 다음 레버 (STAL 정정 반영, 우선순위 재정렬)
1. **bbox 기하 (승격, 코덱스 제기):** **reg_max=1**이 tiny AP75 천장 의심 → DFL/regression 표현력·quality score·box quantization 점검. matched IoU 정체(0.655→0.662)와 직결. → **구현 착수: 아래 「Phase 4 · 레버① 구현」(P2Refine, 브랜치 `3-head_refine`). oracle 상한 APtiny 0.208로 방향 뒷받침.**
2. **해상도/타일링 (병목 진단용):** imgsz 640/1024/1280 사다리. ⚠️ "모듈 효과"가 아니라 "해상도 효과" — 640에서도 이득 나야 모듈, 1024에서만 살면 해상도가 답. VisDrone tiling/crop도 현실적.
3. **구조 모듈 B — P2 self-refinement:** deep→P2 주입 아님. P2 자신의 고해상 feature를 self-attention + 고주파 detail 보존으로 정제(V0가 tiny 최고였던 신호). layer 20 배치 → bottom-up으로 P3/P4/P5에도 전파. (사용자 선택)
4. **assignment 잔여 레버 (좁음):** 후보 선택은 STAL이 처리하나 최종 topk 지표는 여전히 CIoU(`cls⁰·⁵·CIoU⁶`). tiny가 순위에서 밀릴 수 있어 CIoU→NWD in align-metric은 아직 유효하나 좁은 레버.
5. **loss는 보조:** WIoU/SA-WIoU는 전체/small(16-32)만 유효 — tiny 핵심 해결책으로 세우면 같은 벽.

**성공 기준(naive-P2 대비):** tiny AP50·recall **또는** tiny AP75(기하) 상승이 노이즈(±0.007) 초과. 640에서의 순효과 우선.

---

## 브랜치 구조

```
main (ultralytics 8.4.6 기본)
├── 0-moe_loss          (MoE loss 실험, 별도)
├── 1-loss              (InnerCIoU, InnerSIoU 구현)
├── 1.1-loss_nwd        (NWD 구현, main 기반)
├── 1.2-loss_WIoU       (WIoU v1/v3 구현, 1-loss 기반 — 완료)
├── 1.3.1-loss_sawiou   (SA-WIoU, 1.2-loss_WIoU 기반 — 완료, 로스 트랙 마지막)
├── 1.3-loss_wise_siou  (취소)
├── 2-fpn               (naive-P2 + SGI 5변형 — 학습·평가 완료, negative)
├── 2.2-fpn_gsa         (GSA 모듈 — 학습·평가 완료, negative)
├── 2.1-assigner        (NWD assignment — 예정, Phase 4-1)
├── 3-head_refine       (P2Refine head refine — 2026-07-06 학습완료 **negative**)
└── 2.3-fpn_affg        (ScaleConcat α=0.5 억제 — 2026-07-08 **negative**; AFFG 게이트 미구현/폐기)
```
> **⛔ 구조(P2 fusion) 트랙 종료 (2026-07-08):** 주입(SGI/GSA)·재배치(P2Refine)·억제(α=0.5/AFFG) **3방향
> 전부 negative** → **naive-P2 가 sweet spot, 구조 모듈로 tiny 해결 불가 확정.** tiny 병목 = 정보(해상도).
> 해상도/tiling 은 전처리라 모델 연구서 제외(사용자 결정). 상세 `../04_results/SOD_RESULT_AP_SIZE.md`.
>
> **▶ 방향 전환 (2026-07-08): tiny 접고 small(16-32) 위치 정밀도로.** small 은 feature 있음(P2 4~8px,
> tiny 와 결정적 차이) + 위치 정밀도 여지 큼(AP75 0.1147, oracle 0.3067). Phase 1 = **P2+SA-WIoU**
> (WIoU 는 small 홀대 → SA-WIoU 로 재시도), Phase 2 = small-전용 boundary refine 모듈.
> 설계·인수인계: `../03_modules/SOD_MODULE_SMALL_REFINE.md`. (marine/BrackishMOT 는 예비 후보)
> **md 문서는 `workspace/md/sod-paper/` 관리**(github 미업로드). 실험 코드는 포크 각 브랜치.

---

## 핵심 판단 기준

| 결정 포인트 | 기준 | 현재 상태 |
|---|---|---|
| Loss 트랙 | APtiny 노이즈 초과 | **완료**(8종, 0개 초과 → 구조로 전환) |
| 구조 필요성 | naive-P2가 APtiny/전반 개선 | **확인**(전반 +0.0195) |
| P2 fusion 모듈 | naive-P2 대비 tiny↑ | **완료·negative**(SGI 5변형+GSA, 대조군 포함 무효) |
| STAL(size-adaptive assign) | 극소 GT 후보 포함 | **이미 존재·활성**(8.4.6 `tal.py`, 미시도 아님) |
| 다음 레버 | naive-P2 대비 tiny↑(노이즈 초과) | **미정**: 기하(reg_max)·해상도·모듈B 중 |
| 논문 기여 주장 | 3개 데이터셋 전체 ablation + multi-seed | 미완 |

---

## 논문 서술 전략 (2026-06-24 확정)

Loss 7종 결과를 어떻게 서술할지에 대한 전략. **핵심: "최고 loss를 골라 개선했다"가 아니라 "loss로는 안 풀린다는 진단 → 구조 기여 동기"로 프레이밍.**

### 왜 단순 "WIoU 최고 → SA-WIoU로 개선" 서술은 안 되나
- **"WIoU가 최고"는 노이즈 내**: WIoU(AP@.5:.95 0.1708) vs InnerSIoU(0.1703) 차이는 노이즈 바닥(±0.005~0.010) 안. 단일 seed로 단정 불가 → **multi-seed(평균±std) 필수**.
- **"SA-WIoU가 WIoU 단점 보완"은 헤드라인(APtiny) 기준 사실 아님**: SA-WIoU APtiny=0.0558(정체), APsmall(16-32)만 0.1252로 개선. "보완했다"로 쓰면 표와 모순 → 리뷰 거절 사유.

### 채택 프레이밍 (B): loss 연구 = 진단, 구조 = 메인 기여
> "IoU 회귀 loss 7종(CIoU, InnerCIoU, InnerSIoU, ratio<1, NWD, WIoU, SA-WIoU)을 체계적으로 검증했다. **어떤 loss도 AP_tiny를 run-to-run 분산 이상으로 개선하지 못했다.** 특히 SA-WIoU는 small(16-32px)은 개선하나 tiny(8-16px)는 개선하지 못했는데, 이는 극소 객체의 병목이 loss 가중치가 아니라 **구조적 요인(positive anchor 부족·feature 부재)**에 있음을 보여준다. 이것이 본 논문의 구조적 기여(FPN/P2)의 직접적 동기다."

- SA-WIoU의 "APsmall↑ / APtiny→" 비대칭이 **구조 기여를 정당화하는 결정적 증거**.
- 약점이 논리적 무기가 됨: "loss를 끝까지 밀어봤지만 한계 → 그래서 구조".

### 논문 표 구성 (안)
1. **Table (loss ablation)**: 7종 × {AP, AP50, AP_tiny, AP_small, AP_medium}. multi-seed 평균±std. → "AP_tiny는 어느 것도 분산 이상 못 넘음" 시각화.
2. **본문 분석**: SA-WIoU 사례로 "loss 가중치는 16-32px까지만 닿는다" 논증.
3. **구조 섹션**: FPN/P2가 AP_tiny를 움직임을 보여 대비.

### 필수 조건 (서술 전 충족)
- [ ] 최종 후보 loss(WIoU 또는 SA-WIoU) + baseline **multi-seed(≥2~3) 재학습** → "분산 내/외" 정량화.
- [ ] loss ablation 표를 multi-seed 평균±std로 재작성.

### 로스 최종 선택
- **WIoU v3**: 전반 지표 최고(서술 단순). 또는
- **SA-WIoU**: APsmall 최고 + "scale 편향 진단·교정" 분석 가치(진단 프레이밍과 더 잘 맞음).
- 권장: **SA-WIoU를 진단 도구로 본문에 두고**, 최종 적용 loss는 multi-seed로 동급이면 서술 단순한 WIoU 채택.

---

## Phase 3.5 — injection 통계 재확인 (2026-07-03, 도구화)

`/home/hsjeong/workspace/SOD-PAPER/.claude/skills/sod-research-method/analyze_result.py`(대조군 교체 재검):
- **SGIV3 APtiny: baseline 대비 +0.0102(개선처럼) → P2(직전단계) 대비 +0.0034(노이즈내).**
  SGI의 "개선"은 전부 P2 헤드 공로 → injection 순수기여 무효 **재확인**.
- SGIV3noattn(주입만) P2 대비 +0.0003 → 주입 완전 무효.
- 교훈: **대조군은 baseline 이 아니라 직전 단계(P2).** 노이즈바닥 실측(InnerCIoU r=1.0≡CIoU) tiny ±0.0068.

## Phase 4 · 레버① 구현 — Localization refinement (head, P2Refine) (2026-07-03)

> 위 **Phase 4 레버①(bbox 기하, matched IoU 0.655→0.662 정체)의 구체 구현** — 별도 Phase 아님.
> 레버③(P2 self-refinement)과 같은 **head/기하** 방향이나, 레버③은 feature 정제이고 여기는 **박스 offset 보정**(디코딩된 박스를 P2 고해상 feature 로 미세보정).

### 방향 근거 — oracle 상한(학습 전 저비용, `oracle_ceiling.py`, VisDrone Y26S, maxDets=1500)
| band | baseline | loc상한(위치완벽) | det상한(recall완벽) |
|---|---|---|---|
| APtiny(8-16) | 0.0564 | **0.2083 (+0.152, 노이즈~20배)** | 0.7752 |
- 위치를 완벽히 고치면 tiny 0.056→0.208 → **localization 방향 천장 큼(파볼 가치)**.
- det상한>loc상한: recall 여지가 더 크나 검출은 P2가 이미 일부 공략 → **우선 위치**.

### 트랙 정의
- 원인 ①(의미부재)=injection → **닫힘**. 원인 ②(위치 정밀도, matched IoU 0.65)=head refinement → **개척**.
- **neck(semantic gap)이 아니라 head(regression)를 건드린다.** tiny는 레이어19=P2/4에서 예측(`Detect.cv2`).
- 설계 문서: `../03_modules/SOD_MODULE_HEAD_REFINE.md`. 브랜치 관례 확장: loss=1.x, fpn=2.x, **head=3.x** (`3-head_refine`).
- 검증 흐름: verify_module(빌드) → verify-sod-math(offset 수식) → run-sod-paper(학습·평가) → analyze_result(**P2 대비** 판정).

---

## 파일 맵

| 파일 | 내용 |
|---|---|
| `../01_datasets/SOD_DATASET_VISDRONE.md` / `dota.md` / `aitod.md` | 데이터셋별 baseline + pycocotools AP |
| `../02_loss/SOD_LOSS_EXPERIMENTS.md` | Loss 8종 실험 전체 기록 + 수식 |
| `../04_results/SOD_RESULT_AP_SIZE.md` | pycocotools 구간별 AP 비교 (loss + 구조) + tiny matched-IoU 진단 + oracle 상한 |
| `../03_modules/SOD_MODULE_FPN_SURVEY.md` | FPN/PAN 변형 비교 + SGI novelty 포지셔닝 |
| `../03_modules/SOD_MODULE_SGI.md` | **SGI 모듈** 설계·수식·창작성·검증·성공기준·점검절차 |
| `../03_modules/SOD_MODULE_GSA.md` | **GSA 모듈** 설계·negative 기록 |
| `../03_modules/SOD_MODULE_HEAD_REFINE.md` | **위치정밀화(head) 모듈** 설계·가설명세·계획 (Phase 4, negative) |
| `../03_modules/SOD_MODULE_AFFG.md` | **AFFG(neck 억제 모듈)** 설계·novelty·검증 (Phase 0 실패·폐기) |
| `../03_modules/SOD_MODULE_SMALL_REFINE.md` | **small 위치 정밀도 트랙**(P2+SA-WIoU → small refine 모듈) 설계·Codex 인수인계 |
| `md/인수인계_260626{,_2,_3}.md` | 코덱스 검토 요청·반영 보고 (3회) |
| `../06_handoff/SOD_HANDOFF_NWD_EXPERIMENT.md` | NWD 실험 설계(가설 기각) |
| `../05_eval/SOD_EVAL_PIPELINE.md` | pycocotools 평가 파이프라인 + AI-TOD maxDets + 상한/통계 도구 |
| `../00_roadmap/SOD_ROADMAP_PAPER_PROCESS.md` | 이 파일 — 전체 연구 흐름·핵심 스토리·서술 전략 |
| `eval/eval_size_ap.py` / `eval/inspect_sgi.py` | 크기별 AP 측정 / SGI γ·src_w 점검 |
| `/home/hsjeong/workspace/SOD-PAPER/.claude/skills/*` | run-sod-paper / develop-sod-module / verify-sod-math / sod-research-method |
| 코드(`2-fpn`) | `block.py:SGI`, `cfg/models/26/yolo26-p2-sgi{V0,V1,V2,V3,V3noattn}.yaml` |
