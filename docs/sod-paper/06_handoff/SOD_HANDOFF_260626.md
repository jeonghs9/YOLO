# SOD 논문 인수인계 — 검토 요청 문서 (2026-06-26)

> **목적:** 다른 AI/연구자에게 **현재 연구의 타당성·창작성·리스크를 비판적으로 검토받기 위한** 자기완결 문서.
> 칭찬보다 **빈틈·반박·대안**을 원합니다. 특히 (a) 모듈 novelty가 충분한지, (b) 실험 설계가 결론을 지지하는지, (c) 놓친 함정이 있는지.

---

## 0. 한 문단 요약

YOLO26s 기반 Small Object Detection(SOD) 논문(MDPI 목표). **IoU 회귀 loss 8종을 전부 검증했으나 소형 객체(APtiny, 8-16px)를 측정 노이즈(±0.007) 이상 개선하지 못함**을 확인 → "소형 객체 병목은 loss가 아니라 구조"라는 결론 도출. **naive-P2(고해상 검출 헤드)를 추가하니 전반 AP는 크게 올랐으나(+0.0195) APtiny는 여전히 정체, 단 tiny recall은 상승** → "찾되 못 맞춤 = semantic gap" 진단. 이를 해결하는 **SGI(Semantic-Guided Injection) 모듈**을 설계·구현(검증 완료, 학습 대기). 최종적으로 **구조(SGI) + loss**의 복합 기여 논문을 목표.

---

## 1. 연구 세팅 (사실관계)

- 베이스: **YOLO26s** (ultralytics 8.4.6, editable install, end-to-end/NMS-free 헤드).
- optimizer **muSGD**, 300ep, patience=50, batch=32, imgsz=640, seed=0 (전 실험 고정).
- 데이터: **VisDrone**(10클래스, 주 실험), DOTA, AI-TOD(일반화용).
- 평가 2종:
  1. ultralytics val (conf=0.25) — mAP50/95.
  2. **pycocotools 크기별 AP** (test split, conf=0.001, max_det=500) — √area px 구간: vt(2-8)/tiny(8-16)/small(16-32)/medium(32+).
- **노이즈 바닥**: InnerCIoU(ratio=1.0)는 CIoU와 **수학적으로 동일**한데 baseline과 APtiny가 0.0068 차이 → 단일 seed 변동폭 **APtiny ±0.007, AP@.5:.95 ±0.005, AP50 ±0.010**. (이 값이 모든 "개선" 판정의 기준선.)

---

## 2. Loss 트랙 결과 (8종, pycocotools test)

| Loss | AP@.5:.95 | APtiny(8-16) | 비고 |
|---|---|---|---|
| Baseline (CIoU) | 0.1646 | 0.0564 | |
| InnerCIoU r=1.0 (≡CIoU) | 0.1697 | 0.0632 | 수학적으로 CIoU와 동일 → 차이=노이즈 |
| InnerSIoU r=1.0 | 0.1703 | 0.0568 | |
| InnerCIoU/SIoU r=0.7 | — | — | val 기준 무효/역효과 |
| NWD (C=14.5) | 0.1570 | 0.0498 | 전 구간 저하 |
| WIoU v3 | **0.1708** | 0.0557 | 전반 최고(노이즈 내) |
| SA-WIoU v1 (stride bin) | 0.1707 | 0.0558 | APsmall 최고 |
| SA-WIoU v2 (픽셀 bin) | 0.1692 | 0.0538 | 역효과(vt만↑) |

**결론:** APtiny를 노이즈 초과로 올린 loss = **0개/8종**. WIoU/SA-WIoU 등이 small(16-32)·전반은 약간 올리나 진짜 tiny(8-16)는 못 건드림.
**해석:** loss(샘플 재가중/기하 변형)는 **고정된 gradient 예산을 크기대 사이에서 재분배**할 뿐, 없는 feature·positive anchor를 만들지 못함.

> 상세: `../02_loss/SOD_LOSS_EXPERIMENTS.md`, `../04_results/SOD_RESULT_AP_SIZE.md`.

---

## 3. 구조 트랙 — naive-P2 결과 (핵심 전환점)

stock `yolo26-p2.yaml`로 P2/4 검출 헤드 추가 (baseline loss=CIoU).

pycocotools test (AP / AR 동시):

| band | base AP | P2 AP | ΔAP | base AR | P2 AR | ΔAR |
|---|---|---|---|---|---|---|
| tiny(8-16) | 0.0564 | 0.0631 | **+0.0067 (~노이즈)** | 0.1682 | 0.1841 | **+0.0159** |
| small(16-32) | 0.1191 | 0.1402 | +0.0211 | 0.2925 | 0.3221 | +0.0296 |
| medium(32+) | 0.2558 | 0.2853 | +0.0295 | — | — | — |
| **AP@.5:.95** | 0.1646 | **0.1841** | **+0.0195** | 0.3168 | 0.3452 | +0.0284 |

추가 진단 (tiny를 IoU 임계로 분해):
| band | tiny AP50 | tiny AP75 | AP75/AP50 |
|---|---|---|---|
| baseline | 0.147 | 0.033 | 22% |
| naive-P2 | 0.161 | 0.036 | 22% |
| (참고) medium P2 | 0.465 | 0.303 | 65% |

**진단 결론:**
- 구조(P2)는 전반 AP를 노이즈 한참 초과로 올림(+0.0195) → "소형=구조" 1차 입증.
- 그러나 **tiny AP는 여전히 노이즈 내**, **tiny recall만 명확히↑** → **찾되 못 맞춤**.
- tiny AP50 자체가 낮음(0.16) + AP75 붕괴(0.036) → **분류·검출(의미)이 1차 병목, 정밀도 2차**.
- = **semantic gap** (P2는 고해상이나 의미 빈약).

> 상세: `../04_results/SOD_RESULT_AP_SIZE.md` 구조 트랙.

---

## 4. 제안 모듈 — SGI (Semantic-Guided Injection)

**목표:** 깊은 층(P3/P4/P5) 의미를 고해상 P2에 **선택적 주입**해 tiny의 recall→AP 전환.

### 수식 (코드와 1:1 검증됨)
입력 `[base, *sources]`, `P_i`=1×1 conv, `U`=upsample to base:
```
base  = P_0(x_0)
F_sem = base                         (V0: self) | mean_i P_i(U(x_i))  (V1~V3)
a_c   = σ(W2·ReLU(W1·GAP(F_sem)))     # 채널 attention (what)
a_s   = σ(Conv7x7([mean_c F_sem; max_c F_sem]))  # 공간 attention (where)
out   = base + γ·( a_s ⊙ (a_c ⊙ F_sem) )    # γ init 0 → 시작은 항등(안전)
```

### 기존 대비 차별 (주장하는 novelty)
- PAN(단순 concat)/BiFPN(스칼라 가중)과 달리 **의미 소스가 직접 채널·위치별(per-pixel)로 주입을 guide**.
- AugFPN(최상위 보강)과 달리 **최하위 고해상(tiny가 사는 곳) 보강**.
- CBAM(self-attention)과 달리 **cross-level**(깊은 층이 P2를 guide; V0가 그 대조군).
- residual 주입으로 **detail 보존**.
- **소형객체 타깃 + size별 AP 검증**.

### 검증 (구현 정확성/안전성 — 효과는 미검증)
- **수식==코드** allclose=True / 게이트∈[0,1] / **γ=0→out==base(안전 초기화)** / 4변형 빌드+E2E loss+backward 정상.
- 비용: GFLOPs naive-P2 27.8 → SGI V0~V3 **28.0~28.2 (+0.8~1.5%)**, params ~9.8M. 매우 가벼움.

### 실험 사다리 (학습 대기)
| 실험 | SGI from | 구성 | 증명 대상 |
|---|---|---|---|
| naive-P2 | — | P2+P3 concat | 기준 |
| V0 | [19] | P2 self만 | "attention만으로 되나?" (대조군) |
| V1 | [19,16] | +P3 | "가까운 의미 주입" |
| V2 | [19,16,13] | +P3+P4 | "중간 깊이" |
| V3 | [19,16,13,10] | +P3+P4+P5 | "최심부까지" |

성공 판정: naive-P2 대비 **APtiny가 노이즈(±0.007) 초과 상승**.

> 상세: `../03_modules/SOD_MODULE_SGI.md`. 코드: `2-fpn` 브랜치 `ultralytics/nn/modules/block.py: SGI`.

---

## 5. 논문 구성 (계획)

- **기여 1 (구조, 메인):** SGI 모듈 — semantic gap 해소, APtiny 개선.
- **기여 2 (학습):** loss 8종은 "loss로 안 풀림" 진단 + 최종 후보(WIoU/SIoU) 구조 위 refine.
- 핵심 논거: 소형객체 실패 = **두 독립 원인** (①정보 부재=구조, ②위치 신뢰성=loss). 직교·상보.
- ablation: `baseline → +P2 → +SGI → +loss`, **구간별 AP**로 분리 입증.
- **multi-seed(≥2~3) 필수** (단일 seed 차이는 노이즈 내).

> 상세: `../00_roadmap/SOD_ROADMAP_PAPER_PROCESS.md` 핵심 스토리 + 서술 전략.

---

## 6. ★ 검토자에게 묻고 싶은 것 (비판 요청)

1. **Novelty 충분한가?** SGI는 "attention 융합 FPN"의 한 변형이다. PAN/BiFPN/AugFPN/CARAFE/FaPN/CBAM 대비 **정말 차별적인가, 아니면 incremental인가?** MDPI(Remote Sensing/Sensors) 기준 통과 가능한가? 차별성을 더 키울 방법은?
2. **실험이 결론을 지지하는가?**
   - "loss로는 APtiny 못 푼다"를 8종 단일-seed로 결론내는 게 정당한가? (노이즈 바닥 논증의 허점은?)
   - naive-P2의 "recall↑/AP→ = semantic gap" 해석이 과도한가? 다른 설명(예: 단순 anchor 불균형, conf 보정 문제)은 배제됐나?
3. **놓친 대안/함정**:
   - SGI 대신 더 단순/강력한 대안이 있나? (예: 입력 고해상 1280, P2에 deformable/FaPN 정렬, query-based sparse head)
   - SGI가 학습 시 실패할 시나리오? (γ가 0에 갇힘, P5→P2 8× 업샘플의 의미 왜곡, mean 집계가 노이즈 소스에 취약 등)
4. **평가 방법론**: pycocotools 크기 구간 정의/노이즈 바닥 추정 방식이 타당한가? multi-seed 외에 필요한 통제는?
5. **전체 스토리**: "두 독립 원인(구조+loss) 복합 기여" 프레이밍이 설득력 있나, 아니면 loss 파트가 약해서 빼는 게 나은가?

---

## 7. 파일 맵 (검토 시 참조)

| 파일 | 내용 |
|---|---|
| `../00_roadmap/SOD_ROADMAP_PAPER_PROCESS.md` | 전체 흐름 + 핵심 스토리 + 서술 전략 |
| `../02_loss/SOD_LOSS_EXPERIMENTS.md` | loss 8종 상세 + 수식 |
| `../04_results/SOD_RESULT_AP_SIZE.md` | pycocotools 구간별 AP 전체 비교 (loss + 구조) |
| `../03_modules/SOD_MODULE_SGI.md` | SGI 설계·수식·창작성·검증 |
| `../03_modules/SOD_MODULE_FPN_SURVEY.md` | FPN 변형 비교 + novelty 포지셔닝 |
| `../06_handoff/SOD_HANDOFF_NWD_EXPERIMENT.md` | NWD 실험 설계(가설 기각됨) |
| 코드(2-fpn) | `ultralytics/nn/modules/block.py:SGI`, `cfg/models/26/yolo26-p2-sgiV0~V3.yaml` |

**브랜치:** loss=`1-loss`/`1.1-loss_nwd`/`1.2-loss_WIoU`/`1.3.1-loss_sawiou`, 구조=`2-fpn`, 문서=`main`. (remote: jeonghs9/SOD-PAPER)
