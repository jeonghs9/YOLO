# 크기별 AP 평가 결과

평가 조건: pycocotools, VisDrone **test split**, conf=0.001, max_det=500, imgsz=640.  
크기 구간: √area(px) 기준 — vt[2,8) / tiny/APT[8,16) / small[16,32) / medium[32,∞).

---

## YOLO26s Baseline (CIoU) — 참조

| 지표 | 값 |
|---|---|
| AP@[.5:.95] | 0.1646 |
| AP50 | 0.2918 |
| AP75 | 0.1647 |
| AR | 0.3168 |

| size band | AP | AR |
|---|---|---|
| all | 0.1646 | 0.3168 |
| vt(2-8) | 0.0205 | 0.0583 |
| **tiny/APT(8-16)** | **0.0564** | 0.1682 |
| small(16-32) | 0.1191 | 0.2925 |
| medium(32+) | 0.2558 | 0.4415 |
| cocoS(<32) | 0.0845 | 0.2274 |
| cocoM(32-96) | 0.2441 | 0.4302 |
| cocoL(>96) | 0.3342 | 0.5113 |

| class | AP |
|---|---|
| pedestrian | 0.1147 |
| people | 0.0556 |
| bicycle | 0.0337 |
| car | 0.4379 |
| van | 0.2035 |
| truck | 0.1926 |
| tricycle | 0.0766 |
| awning-tricycle | 0.0765 |
| bus | 0.3394 |
| motor | 0.1158 |

---

## InnerCIoU (ratio=1.0) — RUN_260617_03

| 지표 | 값 |
|---|---|
| AP@[.5:.95] | 0.1697 |
| AP50 | 0.3020 |
| AP75 | 0.1691 |
| AR | 0.3219 |

| size band | AP | AR |
|---|---|---|
| all | 0.1697 | 0.3219 |
| vt(2-8) | 0.0169 | 0.0627 |
| **tiny/APT(8-16)** | **0.0632** | 0.1674 |
| small(16-32) | 0.1241 | 0.2996 |
| medium(32+) | 0.2638 | 0.4482 |
| cocoS(<32) | 0.0896 | 0.2326 |
| cocoM(32-96) | 0.2541 | 0.4372 |
| cocoL(>96) | 0.3250 | 0.5418 |

| class | AP |
|---|---|
| pedestrian | 0.1160 |
| people | 0.0593 |
| bicycle | 0.0367 |
| car | 0.4411 |
| van | 0.2122 |
| truck | 0.2011 |
| tricycle | 0.0838 |
| awning-tricycle | 0.0862 |
| bus | 0.3454 |
| motor | 0.1155 |

---

## InnerSIoU (ratio=1.0) — RUN_260618_01

| 지표 | 값 |
|---|---|
| AP@[.5:.95] | 0.1703 |
| AP50 | 0.3039 |
| AP75 | 0.1696 |
| AR | 0.3232 |

| size band | AP | AR |
|---|---|---|
| all | 0.1703 | 0.3232 |
| vt(2-8) | 0.0149 | 0.0526 |
| **tiny/APT(8-16)** | **0.0568** | 0.1578 |
| small(16-32) | 0.1221 | 0.2937 |
| medium(32+) | 0.2660 | 0.4563 |
| cocoS(<32) | 0.0868 | 0.2248 |
| cocoM(32-96) | 0.2550 | 0.4451 |
| cocoL(>96) | 0.3177 | 0.5118 |

| class | AP |
|---|---|
| pedestrian | 0.1155 |
| people | 0.0572 |
| bicycle | 0.0391 |
| car | 0.4392 |
| van | 0.2166 |
| truck | 0.1975 |
| tricycle | 0.0896 |
| awning-tricycle | 0.0845 |
| bus | 0.3461 |
| motor | 0.1177 |

---

## NWD (C=14.5) — RUN_260619_01

| 지표 | 값 |
|---|---|
| AP@[.5:.95] | 0.1570 |
| AP50 | 0.2886 |
| AP75 | 0.1523 |
| AR | 0.2843 |

| size band | AP | AR |
|---|---|---|
| all | 0.1570 | 0.2843 |
| vt(2-8) | 0.0137 | 0.0395 |
| **tiny/APT(8-16)** | **0.0498** | 0.1406 |
| small(16-32) | 0.1117 | 0.2564 |
| medium(32+) | 0.2476 | 0.4042 |
| cocoS(<32) | 0.0792 | 0.1953 |
| cocoM(32-96) | 0.2362 | 0.3938 |
| cocoL(>96) | 0.3268 | 0.4536 |

| class | AP |
|---|---|
| pedestrian | 0.1005 |
| people | 0.0594 |
| bicycle | 0.0329 |
| car | 0.4181 |
| van | 0.1862 |
| truck | 0.1841 |
| tricycle | 0.0905 |
| awning-tricycle | 0.0757 |
| bus | 0.3208 |
| motor | 0.1019 |

---

## WIoU v3 (α=1.9, δ=3) — RUN_262623_01

| 지표 | 값 |
|---|---|
| AP@[.5:.95] | 0.1708 |
| AP50 | 0.3089 |
| AP75 | 0.1684 |
| AR | 0.3250 |

| size band | AP | AR |
|---|---|---|
| all | 0.1708 | 0.3250 |
| vt(2-8) | 0.0142 | 0.0487 |
| **tiny/APT(8-16)** | **0.0557** | 0.1620 |
| small(16-32) | 0.1222 | 0.2934 |
| medium(32+) | 0.2668 | 0.4587 |
| cocoS(<32) | 0.0867 | 0.2262 |
| cocoM(32-96) | 0.2573 | 0.4493 |
| cocoL(>96) | 0.3226 | 0.5159 |

| class | AP |
|---|---|
| pedestrian | 0.1167 |
| people | 0.0590 |
| bicycle | 0.0411 |
| car | 0.4372 |
| van | 0.2169 |
| truck | 0.2009 |
| tricycle | 0.0854 |
| awning-tricycle | 0.0901 |
| bus | 0.3391 |
| motor | 0.1213 |

---

## SA-WIoU (3-bin stride, 8/16/32) — RUN_260624_01

| 지표 | 값 |
|---|---|
| AP@[.5:.95] | 0.1707 |
| AP50 | 0.3054 |
| AP75 | 0.1690 |
| AR | 0.3250 |

| size band | AP | AR |
|---|---|---|
| all | 0.1707 | 0.3250 |
| vt(2-8) | 0.0167 | 0.0585 |
| **tiny/APT(8-16)** | **0.0558** | 0.1663 |
| small(16-32) | 0.1252 | 0.2988 |
| medium(32+) | 0.2657 | 0.4555 |
| cocoS(<32) | 0.0890 | 0.2313 |
| cocoM(32-96) | 0.2554 | 0.4449 |
| cocoL(>96) | 0.3325 | 0.5412 |

클래스별 AP@.5:.95: pedestrian 0.1170, people 0.0598, bicycle 0.0394, car 0.4423, van 0.2163, truck 0.1924, tricycle 0.0846, awning-tricycle 0.0954, bus 0.3414, motor 0.1182.

---

## 종합 비교

### 전체 AP 지표

| Loss | AP@[.5:.95] | AP50 | AP75 | AR |
|---|---|---|---|---|
| Baseline (CIoU) | 0.1646 | 0.2918 | 0.1647 | 0.3168 |
| InnerCIoU (r=1.0) | 0.1697 | 0.3020 | 0.1691 | 0.3219 |
| InnerSIoU (r=1.0) | 0.1703 | 0.3039 | **0.1696** | 0.3232 |
| NWD (C=14.5) | 0.1570 | 0.2886 | 0.1523 | 0.2843 |
| **WIoU v3** | **0.1708** | **0.3089** | 0.1684 | **0.3250** |
| SA-WIoU v1 (stride) | 0.1707 | 0.3054 | 0.1690 | **0.3250** |
| SA-WIoU v2 (픽셀) | 0.1692 | 0.3006 | 0.1695 | 0.3231 |

### 크기별 AP 비교 (핵심 지표)

| size band | Baseline | InnerCIoU | InnerSIoU | NWD | WIoU v3 | SA v1(stride) | SA v2(픽셀) |
|---|---|---|---|---|---|---|---|
| vt(2-8) | 0.0205 | 0.0169 | 0.0149 | 0.0137 | 0.0142 | 0.0167 | **0.0236** |
| **tiny/APT(8-16)** | 0.0564 | **0.0632** | 0.0568 | 0.0498 | 0.0557 | 0.0558 | 0.0538 |
| small(16-32) | 0.1191 | 0.1241 | 0.1221 | 0.1117 | 0.1222 | **0.1252** | 0.1222 |
| **medium(32+)** | 0.2558 | 0.2638 | 0.2660 | 0.2476 | **0.2668** | 0.2657 | 0.2639 |
| cocoS(<32) | 0.0845 | 0.0896 | 0.0868 | 0.0792 | 0.0867 | **0.0890** | 0.0865 |
| cocoM(32-96) | 0.2441 | 0.2541 | 0.2550 | 0.2362 | **0.2573** | 0.2554 | 0.2533 |
| cocoL(>96) | 0.3342 | 0.3250 | 0.3177 | 0.3268 | 0.3226 | 0.3325 | **0.3394** |

> **SA-WIoU v2(픽셀 4-bin)는 역효과**: vt(2-8)만 최고로 끌어올리고(0.0236) 정작 APtiny(8-16)는 baseline 이하로 하락(0.0538). loss 재가중은 고정 gradient 예산을 크기대 사이에서 **재분배**할 뿐 — 잘게 나눌수록 near-hopeless vt로 새어나가 손해. **8종째 APtiny 노이즈 초과 0개.**

### Δ vs Baseline

| size band | InnerCIoU | InnerSIoU | NWD | WIoU v3 | SA-WIoU |
|---|---|---|---|---|---|
| vt(2-8) | -0.0036 | -0.0056 | -0.0068 | -0.0063 | -0.0038 |
| **tiny/APT(8-16)** | **+0.0068** | +0.0004 | **-0.0066** | -0.0007 | -0.0006 |
| small(16-32) | +0.0050 | +0.0030 | -0.0074 | +0.0031 | **+0.0061** |
| **medium(32+)** | +0.0080 | +0.0102 | -0.0082 | **+0.0110** | +0.0099 |
| AP@[.5:.95] (all) | +0.0051 | +0.0057 | -0.0076 | **+0.0062** | +0.0061 |

> **APtiny에서 노이즈(±0.007)를 넘은 로스는 7종 중 0개.** SA-WIoU는 APsmall(16-32)만 최고(+0.0061), tiny(8-16)는 무효 → loss 재가중은 16-32px까지만 닿고 진짜 tiny는 구조 문제.

> **노이즈 바닥**(InnerCIoU r=1.0 ≡ CIoU인데 baseline과 차이로 측정): APtiny ±0.0068, AP@.5:.95 ±0.0051, AP50 ±0.0102.
> → **APtiny에서 노이즈를 넘은 로스는 없음**(전부 ±0.007 내). 전반 지표 최고는 WIoU지만 InnerSIoU와 차이는 노이즈 내.

---

## 분석 및 가설 검증

### HANDOFF 가설 검증 결과

> 가설: NWD 단독 적용 시 APtiny↑ + APmedium↓

| 가설 | 예측 | 실제 | 결과 |
|---|---|---|---|
| APtiny(8-16) 개선 | ↑ | **↓ (-0.0066)** | **기각** |
| APmedium(32+) 저하 | ↓ | **↓ (-0.0082)** | **확인** |

**가설 일부 기각**: NWD는 APmedium은 떨어지지만 APtiny도 함께 떨어짐. Gaussian loss가 작은 객체에 유리할 것이라는 예상과 반대.

### 핵심 발견

1. **InnerCIoU가 APtiny 최고 (0.0632)**: Baseline(0.0564) 대비 +0.0068. CIoU를 inner box로 개선하는 것이 tiny 객체에 가장 효과적.

2. **InnerSIoU가 AP@[.5:.95] 및 APmedium 최고**: 전반적 localization 정밀도 우수.

3. **NWD는 모든 크기 구간에서 Baseline 이하**: val mAP50은 0.410으로 최고였지만 test pycocotools에서 역전. val(conf=0.25) vs test(conf=0.001) 평가 조건 차이가 원인 — NWD는 high-confidence 예측에 집중하므로 conf=0.25 기준 val에서 유리하고, 저임계값 full-recall test에서는 불리.

4. **vt(2-8px) 구간**: 전 모델 Baseline 이하 — 2~8px 극소 객체는 loss 변경으로는 한계.

### 최종 결론 (IoU 로스 트랙 종료, 2026-06-24)

7종 IoU 계열 로스(CIoU, InnerCIoU, InnerSIoU, ratio<1, NWD, WIoU, SA-WIoU)를 모두 검증한 뒤:

1. **소형 객체(APtiny)를 노이즈(±0.007) 이상으로 올린 로스는 없음.** InnerCIoU의 +0.0068은 noise-twin(InnerCIoU r=1.0 ≡ CIoU) 검증으로 노이즈임이 드러남. → **로스 함수는 SOD 병목(소형 객체)을 건드리지 못함** (견고한 negative result).
2. **WIoU가 전반 지표(AP@.5:.95 0.1708, AP50 0.3089, AR 0.3250) 최고**이나 InnerSIoU 대비 차이는 노이즈 내. recall/AP50에 미세 우위.
3. **NWD만 명확히 저하** (전 구간 baseline 이하).

4. **SA-WIoU**(WIoU의 anti-small bias를 size-stratified 정규화로 제거)는 **APsmall(16-32) 최고(0.1252)**이나 **APtiny(8-16)는 무효(0.0558)** → loss 재가중은 16-32px까지만 닿고, 진짜 tiny(8-16px)는 positive anchor 부족·feature 부재(=구조)가 병목임을 시사.

**→ 로스 최종 후보: WIoU v3 (전반 최고) 또는 SA-WIoU (APsmall 최고 + size-aware 분석 서술 가치).** 단일 seed 노이즈 내 동률이므로 논문 확정은 멀티시드 필요.
**→ 다음 레버는 로스가 아니라 구조(FPN/P2 head) 또는 해상도.** 로스 트랙 탐색 완료(7종 소진). 상세 방향: `../00_roadmap/SOD_ROADMAP_PAPER_PROCESS.md`.

---

# 구조 트랙

## naive-P2 (yolo26s-p2, baseline CIoU) — RUN_260625_01

P2/4 검출 헤드 추가(stock `yolo26-p2.yaml`). 비용: params 9.77M, GFLOPs 27.8(baseline 22.8, +22%).
val: mAP50 0.421, mAP50-95 0.268, P 0.542, R 0.305 (baseline 0.396/0.246/0.517/0.283 대비 전부 큰 상승).

### pycocotools (test) — AP와 AR을 함께 봐야 함

| size band | base AP | **P2 AP** | ΔAP | base AR | **P2 AR** | ΔAR |
|---|---|---|---|---|---|---|
| vt(2-8) | 0.0205 | 0.0203 | -0.0002 | 0.0583 | 0.0696 | +0.0113 |
| **tiny(8-16)** | 0.0564 | 0.0631 | **+0.0067 (~노이즈)** | 0.1682 | 0.1841 | **+0.0159** |
| small(16-32) | 0.1191 | **0.1402** | +0.0211 | 0.2925 | 0.3221 | +0.0296 |
| medium(32+) | 0.2558 | **0.2853** | +0.0295 | 0.4415 | 0.4790 | +0.0375 |
| **AP@.5:.95** | 0.1646 | **0.1841** | **+0.0195** | 0.3168 | 0.3452 | +0.0284 |

클래스별 AP@.5:.95: pedestrian 0.1380, people 0.0692, bicycle 0.0492, car 0.4605, van 0.2269, truck 0.2096, tricycle 0.1052, awning-tricycle 0.0936, bus 0.3598, motor 0.1292. (bicycle·tricycle 등 소형이 loss 대비 크게 상승)

### 핵심 발견 (정교화: 2026-06-26 진단)
- **구조가 loss로 불가능했던 도약**: AP@.5:.95 +0.0195 = loss 최고(WIoU +0.006)의 3배, 노이즈 한참 초과. → "소형 객체는 구조" thesis 1차 입증.
- **그러나 APtiny(8-16)는 +0.0067로 여전히 노이즈 내**, tiny recall만 명확히↑(+0.016). → "찾되 못 맞춤".

### tiny matched-IoU 진단 (2026-06-26, 저장된 예측 재분석)

tiny(8-16px) GT에 대해 "찾은 비율"과 "찾은 것의 localization 품질"을 분리 측정:

| 모델 | 찾은 비율(@IoU.5) | 찾은 것 평균 IoU | 그중 IoU≥0.75 |
|---|---|---|---|
| baseline | 48.9% | 0.655 | 20% |
| naive-P2 | 54.8% (+5.9%p) | 0.662 (≈동일) | 22% |

**→ semantic gap "하나"가 아니라 별개의 두 문제로 갈림:**
1. **검출(detectability, AP50/recall)**: tiny의 ~45%를 아예 못 찾음. P2가 일부 개선(+5.9%p). → **"의미 부족"이 맞는 영역, SGI 공략 대상.**
2. **정밀도(localization, AP75)**: 찾은 것조차 평균 IoU 0.65에 갇힘(P2도 못 올림). 8-16px는 1px 오차로 IoU 급변 → **localization 품질이 거의 안 오름**(원인은 geometric sensitivity 추정, annotation/assignment 등과 아직 미분리 — "기하 한계"로 단정 금지).

**함의 (정직하게):**
- **loss 8종이 APtiny 못 올린 이유와 일관**: tiny AP75(정밀도) 쪽이 막혀 있어 회귀 loss로 뚫기 어려웠던 것으로 보임.
- **헤드라인 지표는 APtiny/APsmall 유지** (SOD 관례, cherry-picking 방지). **AP50/AP75/AR/matched-IoU는 "APtiny 결과를 해부"하는 진단 지표**로 사용 (성공 기준을 AP50로 바꾸는 게 아님).
- SGI가 노리는 곳 = **검출(AP50/recall)**; 정밀도(AP75)는 한계가 보여 기대 제한적.
- "semantic gap" 단정 → **"tiny 실패 = 검출(의미) + 정밀도 두 겹, 그중 검출이 actionable"**로 표현 정교화.

### 누적 비교 (pycocotools AP@.5:.95 / APtiny / AP50)
| | AP@.5:.95 | APtiny | tiny AP50 | tiny recall@.5 | 비고 |
|---|---|---|---|---|---|
| baseline | 0.1646 | 0.0564 | 0.147 | 48.9% | |
| best loss (WIoU) | 0.1708 | 0.0557 | — | — | loss는 APtiny 무효 |
| **naive-P2** | **0.1841** | 0.0631 | 0.161 | 54.8% | 전반↑, tiny는 검출만↑ |

## ★ SGI 결과 (RUN_260626~260629) — **negative: 의미 주입 무효** (2026-06-30)

5종 학습(V0~V3 + V3noattn) + pycocotools test:

| 구성 | AP | AP50 | **tiny** | tiny AR | γ (학습값) |
|---|---|---|---|---|---|
| naive-P2 (구조만) | 0.1841 | 0.3208 | 0.0631 | 0.1841 | — |
| V0 = attention만 (소스X) | 0.1866 | 0.3250 | **0.0676** | 0.1799 | +0.69 |
| V3noattn = 소스 주입만 (attn X) | 0.1846 | 0.3220 | 0.0635 | 0.1744 | +0.65 |
| V1 (+P3) | 0.1851 | 0.3213 | 0.0659 | 0.1880 | +0.76 |
| V2 (+P3P4) | 0.1826 | 0.3202 | 0.0673 | 0.1918 | −1.12 |
| V3 (+P3P4P5) | 0.1882 | 0.3271 | 0.0666 | 0.1830 | +1.18 |
| **GSA** (sharp attn, 동일입력) | 0.1851 | 0.3221 | **0.0631** | 0.1835 | **−0.91** |

V3 학습 src_w: P3 0.39 / P4 0.32 / P5 0.29 (고루 사용). γ 4종 모두 0에서 벗어남 → **모듈은 정상 학습**(미학습 아님). V2는 γ 음수(주입을 빼는 방향).

**GSA (RUN_260701_01, 2026-07-02) — 후속 모듈도 negative:** SGI-V3와 **동일 입력**. 결과 **tiny 0.0631 = naive-P2와 정확히 동일**, 전체 AP는 SGI-V3보다 낮음, **γ = −0.91(음수)** = 모델이 GSA 출력을 빼는 방향으로 학습. → **무효**. ⚠️ **코덱스 정정:** GSA 코드는 deep source를 **nearest 업샘플(흐림) 후** 그 위에서 local attention → "흐리기 전 선명 복원"이 아니라 **coarse prior 재혼합**(새 detail 없음, 실패와 일치). 상세 `../03_modules/SOD_MODULE_GSA.md`.

**3-way 대조군 분해 (코덱스가 요구한 핵심):**
- **소스 주입만(V3noattn) = naive-P2** (tiny 0.0635 vs 0.0631 = **+0.0004, 사실상 0**, AP +0.0005).
  → **cross-level 의미 주입은 그 자체로 효과 없음.**
- **attention만(V0)이 최고**(tiny 0.0676) → SGI의 (노이즈 수준) 미세 이득은 **주입이 아니라 P2 self-attention**에서.
- 모든 차이 노이즈 바닥(tiny ±0.007, AP ±0.005) 안.

**결론 (정직):** **SGI의 novelty(의미 주입)는 자기 대조군에 의해 반박됨.** 이유 = naive-P2의 P2 feature(L19)가 **이미 top-down PAN으로 P3/P4/P5 의미를 받음** → "semantic gap"은 기존 구조가 이미 메워놨고 추가 주입은 중복.

**확정된 것 (2026-07-02, SGI+GSA 통합):** ① P2 헤드(구조)는 baseline 대비 확실(AP +0.0195). ② **"깊은 의미를 P2로 가져오는 fusion 모듈" 계열 전부 무효** — 주입(SGI), 선명 주입(GSA), attention/소스 각 대조군까지 포함. 갭(의미)은 top-down PAN이 이미 메웠기 때문. ③ tiny 병목 = 검출(recall)+정밀도(기하). → 다음 레버는 **P2 fusion이 아니라 assignment(누구를 학습)·해상도·기하.**

---

## Oracle 상한 실험 (2026-07-03) — 학습 전 방향 판정

도구 `/home/hsjeong/workspace/SOD-PAPER/.claude/skills/sod-research-method/oracle_ceiling.py`. VisDrone Y26S test, iou_thr=0.1, **maxDets=1500(AI-TOD식)**. 캐시 예측(`pred_Y26S`)만으로 "완벽히 풀면 상한"을 **학습 없이** 측정.

| size band | baseline | loc상한(위치완벽) | Δloc | det상한(recall완벽) |
|---|---|---|---|---|
| all | 0.1646 | 0.3426 | +0.1779 | 0.7763 |
| vt(2-8) | 0.0205 | 0.1424 | +0.1219 | 0.8639 |
| **tiny/APT(8-16)** | **0.0564** | **0.2083** | **+0.1519** | 0.7752 |
| small(16-32) | 0.1191 | 0.3067 | +0.1876 | 0.7242 |
| medium(32+) | 0.2558 | 0.4648 | +0.2090 | 0.7644 |

- **loc상한** = 예측 박스를 매칭 GT 위치로 스냅(위치만 완벽). **관대한 천장**(실제<상한) — 도달목표 아니라 방향판정용.
- **APtiny loc상한 0.2083 = 노이즈(±0.007)의 ~20배 → 위치 정밀화(head refinement, Phase 4) 방향 유효.**
- det상한(recall완벽) > loc → 검출 여지 더 크나 검출은 P2가 이미 일부 공략 → **우선 localization**.

## SAQ q-rescore 상한 진단 (2026-07-08) — naive-P2(P2v2) 기준, SAQ 전제 검증

도구 `scratchpad/saq_premise_diag.py`. 캐시 예측(`pred_P2v2`)에 **완벽 quality head** 가정 `score=cls·IoU^γ`
적용 시 small AP 상한. "quality-aware ranking(SAQ)이 얼마나 여지 있나"를 **학습 없이** 측정. (연계 `../03_modules/SOD_MODULE_SAQ.md` §9-K)

| setting | small AP | AP50 | AP75 | **AR** | all AP |
|---|---|---|---|---|---|
| baseline (naive-P2) | 0.1402 | 0.2953 | 0.1147 | 0.3221 | 0.1841 |
| cls·IoU^2 | 0.2490 | 0.4882 | 0.2213 | **0.3221** | 0.2899 |
| cls·IoU^4 | 0.2602 | 0.4929 | 0.2408 | **0.3221** | 0.2986 |
| perfect rank(q>0.5, 이진) | 0.1567 | 0.3462 | 0.1201 | 0.3221 | 0.1861 |

- **misalignment 실존**: cls↔matched-IoU Spearman small **0.501** < medium 0.637 (small 이 위치품질 덜 반영).
- **rescore 여지 큼**: small AP **+0.12(0.14→0.26)**, AP50 **+0.19** → SAQ ranking/FP억제 방향 유효.
- **⚠ AR(recall) 0.3221 불변** = **rescore 는 후보를 못 늘림 → SAQ 는 recall 못 올린다**(ranking/precision only).
  SAQ 천장 = rescore ~0.26 (loc 0.336/det 0.762 아님). **recall 은 probe/topk/assignment 몫** — SAQ 와 상보.
- **연속값 필수**: `q>0.5` 이진 rescore 는 +0.016 만(small 은 IoU>0.5 TP 가 적어 이진화 시 AP50~75 TP 버림).

## Phase 2 probe — NWD assignment blend 판정 (2026-07-09) ★ negative

`260708_Y26S_P2_NWDPROBE_VISDRONE` (2.4-probe_nwd_assign: tal.py align_metric small-GT CIoU/NWD blend, λ=0.25,
sqrt-area 16-32, topk 고정). test @maxDets1500, naive-P2(P2v2) 대비.

| 지표 | naive-P2 | probe(NWD) | Δ |
|---|---|---|---|
| small AP | 0.1402 | 0.1392 | −0.0010 |
| small AP50 | 0.2953 | 0.2913 | −0.0040 |
| small AP75 | 0.1147 | 0.1127 | −0.0020 |
| small AR | 0.3221 | 0.3205 | −0.0016 |
| all AP | 0.1841 | 0.1847 | +0.0006 |

- **판정 negative**: small 전부 노이즈 밴드(±0.007) 안, 미세 하락. **assignment(NWD blend)로 small 못 올림.**
- **핵심 함의 (표현 신중, Codex)**: **이 실험군(naive-P2+STAL+NWD-probe) 안에서 small raw recall 은 0.32 부근에서
  안 움직임**(AR −0.0016) → **이 실험군 내 recall 축 ROI 낮음.** (⚠ 단정 금지: "feature 근본 벽"인지 STAL 중복·
  NWD-AP 불일치·topk 엇갈림인지 원인 미분리. probe 하나로 candidate 전체가 막혔다 결론은 과함.)
  oracle q-rescore 도 AR 불변 → **ranking headroom(+0.12)은 recall 과 독립**, probe 실패가 이를 없애지 않음.
- **원인 추정**: STAL 이 이미 극소 GT 후보 확대(Codex 중복 우려 현실화) + λ=0.25 약함 + NWD 가 AP(IoU) 정렬 아님.
- **→ NWD-in-assignment 폐기.** NWD 는 SAQ 의 A3 auxiliary 로만 재검토. SAQ(inference score-quality)는 별개 축(§SAQ md).

## SGIV3 대조군 재분석 (2026-07-03)

도구 `analyze_result.py`. 노이즈 바닥 실측(InnerCIoU r=1.0 ≡ CIoU): **APtiny ±0.0068**(문서값과 일치).
- SGIV3 APtiny: baseline 대비 **+0.0102(개선 착시)**, **P2 대비 +0.0034(무효)**.
- SGIV3noattn(주입만) P2 대비 **+0.0003** → 주입 순수기여 완전 무효.
- → injection 최종 종결. **대조군을 baseline 으로 잡으면 P2 공로를 새 모듈 공로로 오인**(방법론 교훈, `sod-research-method`).

## P2Refine (head localization refinement) — negative (2026-07-07)

RUN=`260706_Y26S_P2REFINE_VISDRONE` (batch32/300ep/seed0, device 2,3,4). 최종 **γ=+0.060**
(학습 내내 0.02→0.06 증가 = refine 살아있음). 대조군 = naive-P2(260625). 노이즈 바닥 band ±0.007.

| size band | naive-P2 | P2Refine | Δ | 판정 |
|---|---|---|---|---|
| AP@[.5:.95] | 0.1841 | 0.1801 | −0.0040 | 노이즈내 |
| AP50 | 0.3208 | 0.3131 | −0.0077 | 노이즈내 |
| AP75 | 0.1844 | 0.1814 | −0.0029 | 노이즈내 |
| AR | 0.3452 | 0.3354 | −0.0098 | ▼ 악화 |
| vt(2-8) | 0.0203 | 0.0275 | +0.0072 | ▲ 개선(near-hopeless) |
| **tiny/APT(8-16)** | **0.0631** | **0.0605** | **−0.0026** | **무효** |
| small(16-32) | 0.1402 | 0.1358 | −0.0044 | 노이즈내 |
| medium(32+) | 0.2853 | 0.2801 | −0.0052 | 노이즈내 |

**판정: negative (폐기).** APtiny 무효(−0.0026), AR 유의 악화. γ 는 살아있으나 학습된 offset 이
tiny 위치를 못 고침. → **localization refinement(Phase 4 레버①)도 tiny 무효** — SGI·GSA 에 이은
head 트랙 negative ablation. 원인·인사이트: `../03_modules/SOD_MODULE_HEAD_REFINE.md`.

## ScaleConcat α=0.5 (top-down 억제 = AFFG Phase 0) — negative (2026-07-08)

RUN=`260707_Y26S_SCALE05_VISDRONE` (naive-P2 의 layer 18 만 ScaleConcat α=0.5, **파라미터 0**).
대조군 = naive-P2(`P2v2` 재eval, by-size AP50/75 포함). 학습 세팅 동일(batch32/300ep/seed0).

| size band | naive-P2 (AP/AP50/AP75/AR) | α=0.5 | 핵심 Δ |
|---|---|---|---|
| **tiny(8-16)** | 0.0631/0.1613/0.0364/0.1841 | 0.0497/0.1308/0.0296/0.1558 | AP **−0.0135**, AP50 −0.0305, AR −0.0283 ▼ |
| small(16-32) | 0.1402/0.2953/0.1147/0.3221 | 0.1098/0.2382/0.0857/0.2857 | AP −0.0304 ▼ |
| medium(32+) | 0.2853/0.4649/0.3025/0.4790 | 0.2410/0.4037/0.2530/0.4435 | AP −0.0443 ▼ |
| **전체(all)** | 0.1841/0.3208/0.1844/0.3452 | 0.1534/0.2722/0.1538/0.3145 | AP **−0.0307** ▼ |

**판정: 실패.** tiny 를 못 올린 정도가 아니라 **하락(−0.0135)** + 전체 붕괴(−0.0307). top-down 억제는
VisDrone tiny 에 독 — classification confidence 가 무너져 AP50/AR 급락(Codex #3 리스크 현실화).
**"top-down 과잉" 가설 기각**; Fusion Factor(TinyPerson/CityPersons)는 VisDrone 미성립.
⚠ val 은 억제에도 0.216 으로 버팀 → test 붕괴 = **val≠test 교훈**. → **AFFG 게이트 폐기**(Phase 0 관문
실패, 게이트 코드 미작성 = 방법론적 낭비 0). 설계 보존: `../03_modules/SOD_MODULE_AFFG.md`.

## ★ 구조(P2 fusion) 트랙 최종 종료 (2026-07-08)

P2 top-down fusion 을 **3방향 전부** 시도 → **모두 negative**:
**주입(SGI/GSA) · 재배치(P2Refine) · 억제(α=0.5/AFFG).**
→ 더 넣어도·다시 맞춰도·빼도 나빠짐 = **naive-P2 가 sweet spot**. **구조 모듈로 tiny 해결 불가 확정.**
tiny 병목 = fusion 이 아니라 **정보(해상도)**. (해상도/tiling 은 전처리라 모델 연구서 **제외** — 사용자 결정.)
확정 기여: **naive-P2 헤드**(전체 AP +0.0195, small/medium 큰 향상) + loss(WIoU/SA-WIoU, 전체·small 소폭).

## P2+WIoU (구조+loss 결합) — small 무효, large 만 유의 (2026-07-08)

RUN=`260708_Y26S_P2_WIoU_VISDRONE` (yolo26s-p2 + WIoU v3, `1.2-loss_WIoU` 코드). 대조군=naive-P2(P2v2).

| 지표 | naive-P2 | P2+WIoU | Δ | 판정 |
|---|---|---|---|---|
| mAP50 | 0.3208 | 0.3268 | +0.0060 | 노이즈 · |
| mAP50-95 | 0.1841 | 0.1869 | +0.0028 | 노이즈 · |
| APs(<32) | 0.0999 | 0.1007 | +0.0008 | 무효 · |
| APm(32-96) | 0.2744 | 0.2779 | +0.0035 | 노이즈 · |
| **APl(>96)** | 0.3617 | 0.3805 | **+0.0189** | **유의 ▲** |
| small(16-32) AP75 | 0.1147 | 0.1149 | +0.0002 | 무효 |

**판정: small 개선 실패.** 유일한 유의 개선은 **APl(large)** — WIoU 는 큰 객체(feature 충분)의 위치
정밀도만 올림. ⚠ **WIoU v3 는 "어려운 박스 down-weight"라 small(어려움)을 홀대** → small AP75 못 올림.
→ small 은 **SA-WIoU**(size 편향 제거)로 재시도 필요.

## → small object 위치 정밀도 트랙 착수 (2026-07-08)

tiny(정보 한계) 접고 **small(16-32) 집중**. 병목 = **위치 정밀도 + detection/ranking 공존**(단독 아님):
- small AP50 0.2953 vs AP75 0.1147(비율 39%; medium 65%) → 위치 정밀도 여지 강한 신호.
- **naive-P2 기준 oracle(small): loc 상한 0.3360, det 상한 0.7619 → det≫loc = recall/ranking 여지가 더 큼.**
- small 은 tiny 와 달리 **feature 있음**(P2 4~8px). 단 **P2Refine 은 small AP75 도 못 올림(−0.0036)** →
  단순 offset 금지. 순서 = **P2+SA-WIoU → ranking/quality → (조건부)refine**.
설계·계획: `../03_modules/SOD_MODULE_SMALL_REFINE.md`.

## ★ SPD-Conv (recall 축 probe) — backbone·full 모두 negative (2026-07-10)

RUN=`260709_Y26S_P2_SPD_VISDRONE2`(backbone, RUN_260710_01, 15.2M, best ep112) ·
`260709_Y26S_P2_SPDFULL_VISDRONE`(full, RUN_260710_02, 17.5M, best ep147).
SPD-Conv(arXiv:2208.03641) = strided conv 의 space-to-depth 대체(정보손실 없는 다운샘플). **가설=small feature 보존→recall↑.**
- backbone = naive-P2 의 backbone stride-2 Conv 5개만 SPDConv. full = + neck down-path 3개까지(총 8개).
- 학습 세팅 naive-P2 동일(batch32/300ep/seed0/imgsz640). eval: test @maxDets1500, **naive-P2 재평가가 P2v2 와 소수점 일치**(조건·재현성 검증).

| size band | naive-P2 (AP/AP50/AP75/AR) | SPD-bb | SPD-full |
|---|---|---|---|
| all | 0.1841/0.3208/0.1844/0.3452 | 0.1846/0.3230/0.1855/0.3460 | 0.1841/0.3214/0.1844/0.3428 |
| vt(2-8) | 0.0203/0.0666/0.0043/0.0696 | 0.0197/0.0630/0.0053/0.0599 | 0.0199/0.0716/0.0034/0.0604 |
| tiny(8-16) | 0.0631/0.1613/0.0364/0.1841 | 0.0659/0.1685/0.0390/0.1832 | 0.0655/0.1697/0.0341/0.1790 |
| **small(16-32)** | **0.1402/0.2953/0.1147/0.3221** | 0.1377/0.2927/0.1110/**0.3181** | 0.1409/0.2950/0.1155/**0.3146** |
| medium(32+) | 0.2853/0.4649/0.3025/0.4790 | 0.2863/0.4701/0.3070/0.4819 | 0.2830/0.4642/0.2989/0.4773 |

### Δ small vs naive-P2 (핵심 판정 밴드, 노이즈 ±0.007 AP / ±0.01 AR)
| 지표 | SPD-bb Δ | SPD-full Δ |
|---|---|---|
| small AP | −0.0025 | +0.0007 |
| small AP50 | −0.0026 | −0.0003 |
| small AP75 | −0.0037 | +0.0008 |
| **small AR** | **−0.0040** | **−0.0075** |

**판정: negative (null), backbone·full 모두.** 전 밴드 전부 노이즈 안, 결정 지표 **small AR 이 0.3221 에서 안 올라감**
(오히려 미세↓). tiny AP 미세↑(+0.0028/+0.0024)은 노이즈 내 + tiny AR 은 오히려↓. full(neck까지, 17.5M)은 더 무겁기만 하고 small AR 최저 → 확실히 폐기.

**왜 (정직, 단정 회피):**
1. **SPD 전제 ↔ 우리 병목 어긋남**: SPD 이득은 "다운샘플이 small 신호를 죽일 때". 그러나 **P2/4 헤드가 이미 고해상 경로**로 small(16-32)을 검출 → SPD 가 고치려는 정보손실을 P2 가 이미 상당부분 차단. **SGI 가 죽은 논리와 동일**("P2 가 이미 갭 메움").
2. **recall 벽은 feature 보존 문제 아님 — 3연속 방증**: oracle rescore(AR 불변)·NWD-assign probe(AR 불변)·**SPD bb/full(AR 불변)**. recall 겨냥 서로 다른 3레버 전부 small AR 0.32 못 건드림.
   ⚠ **단, det-oracle 상한 0.762** → recall 은 원리적으로 여지 큼. 정확한 표현 = **"근본 불가"가 아니라 "우리 가진 레버로는 안 열림"**.
3. (약한 caveat) SPD 조기종료(best 112/147)이나 val 곡선 naive-P2 와 학습내내 겹침·plateau → undertraining 으로 보기 어려움.

**→ SPD 폐기, negative ablation 등재**(SGI·GSA·P2Refine·AFFG·NWD-assign 에 이어). **recall 축 ROI 소진 → precision/ranking(SAQ) 로 피벗.**
연계 `../03_modules/SOD_MODULE_SAQ.md`.

## ★ SAQ A1a-1 (quality head + L_q, **rescoring 없음**) — learnability PASS (2026-07-13)

RUN=`260710_Y26S_P2_SAQ_A1a1_VISDRONE3`(RUN_260710_03, SAQDetect+L_q, 9.42M, 161ep best111). q head 를 IoU-target-all
(t=max plain IoU, VFL+Σw)로 학습, **inference 미변경**(A1a-1). 목적=① q 가 IoU 를 배우나(learnability) ② head 가 base 교란하나.

### ① size-AP vs naive-P2 (test @maxDets1500) — base 교란 없음
| band | naive-P2 AP/AR | SAQ-A1a1 AP/AR | Δ |
|---|---|---|---|
| all | 0.1841/0.3452 | 0.1872/0.3501 | AP +0.0031 / AR +0.0049 |
| tiny(8-16) | 0.0631/0.1841 | 0.0636/0.1833 | AP +0.0005 / AR −0.0009 |
| **small(16-32)** | 0.1402/0.3221 | 0.1413/0.3223 | AP +0.0011 / AR +0.0002 |
| medium(32+) | 0.2853/0.4790 | 0.2883/0.4856 | AP +0.0030 / AR +0.0066 |

→ 전 밴드 노이즈(±0.007) 내, **미세 양(+)**. rescoring 없으니 예상대로 naive-P2 와 동급 = **cv_q head + L_q(no-detach)가 base 를 안 다침**(§5 ⚠ 우려 미현실화).

### ② learnability (진단 `eval/saq_quality_diag.py`, val 후보 anchor t>0.1, **핵심**)
| band | n | Sp(q,IoU) | Sp(cls,IoU) | q>cls? |
|---|---|---|---|---|
| all | 1.33M | **0.483** | 0.379 | ✓ |
| tiny(8-16) | 0.32M | **0.702** | 0.623 | ✓ |
| **small(16-32)** | 0.47M | **0.504** | 0.362 | ✓ **(+0.142)** |
| medium(32+) | 0.45M | 0.065 | −0.005 | (둘다 ~0) |

- **learnability 성립(강)**: q 가 IoU 를 잘 랭킹(small 0.504, tiny 0.702) + **모든 밴드에서 cls 이김, small 에서 최대(+0.142)**
  = SAQ 가설(cls 가 small 위치품질 못 잡음)을 q 가 메움. 캘리브레이션 고-q 구간 우수(q≈0.865→IoU 0.864).
- ⚠ **캐비엇1 (A1b 최대 리스크)**: **Sp(q,cls)=0.775** — q·cls 고중복. rescoring `cls·q^γ` 이득은 q 가 cls **너머** 정보 줄 때만.
  small 우위(+0.142)가 그 "너머"이나, oracle +0.12 중 얼마 회수할지 **미지 → A1b 실측 필요**.
- ⚠ **캐비엇2 (설계 gap, 7Q 발견1)**: q head 는 **one2many** 경로에서 학습. 그러나 inference 는 **one2one**. A1a-1 은 무관(rescoring 없음)이나
  **A1b 는 one2one 에 q 를 얹어야** 함(별도 head or 이동). 이 위 learnability 는 one2many 기준값.
- ⚠ **캐비엇3**: 저-q 구간 캘리브레이션 비단조(q≈0.03→IoU 0.44) → 저-q 억제가 일부 좋은 박스 죽일 수 있음(γ·저cls 상관이 완화 기대).

**판정: A1a-1 learnability PASS(강). base 무해.** SGI·GSA·SPD 등 연속 negative 뒤 **첫 양성 신호(올바른 축=ranking)**.
**→ 다음: A1b(rescoring) 실측이 진짜 판정.** 단 A1b 전 (a) one2many/one2one 결정 (b) 값싼 rescoring probe 로 실제 q 가 small AP 올리나 선확인 권장.
설계·상세 `../03_modules/SOD_MODULE_SAQ.md`.

### rescoring probe (2026-07-13) — **INCONCLUSIVE(경로 부적합), 방법론 교훈**

`eval/saq_rescore_probe.py`: A1a-1 의 **one2many 후보**(val, 548장·54.3만 det)에 cls / cls·q^γ / cls·IoU^γ(oracle) 3-way 재점수화.

| 랭킹 | small AP | Δ vs cls |
|---|---|---|
| cls (기준) | 0.1044 | — |
| cls·q^γ (학습 q) | best 0.1040 | −0.0004 |
| **cls·IoU^γ (oracle=완벽 q)** | best 0.1011 | **−0.0033** |

- **핵심: oracle(완벽 q)조차 여지 0(오히려 −)** → q 가 약해서가 아니라 **one2many 경로 자체에 rescoring 여지 없음**.
  이유 = one2many 는 장당 ~990 후보(초과밀) → pycocotools 매칭이 중복 흡수, 랭킹이 AP 를 못 바꿈(포화).
- 대조: premise 진단의 oracle **+0.12 는 one2one(최종검출)** 에서. → **rescoring 여지는 one2one 에만 존재.**
- **⚠ 방법론 교훈**: probe 에 **oracle 대조군**을 넣어 "음성=q 실패"와 "음성=경로 부적합"을 분리 → 오판 방지(probe 는 폐기 아니라 **판단불가**로 처리). 이게 없었으면 SAQ 오폐기 위험.
- **→ 값싼 지름길 없음**(여지 있는 one2one 엔 아직 q 미존재) → **A1b(one2one 에 q + 추론 rescoring) 본구현이 유일한 유효 판정.**

## ★ SAQ A1b (q on one2one + 추론 rescoring cls·q^γ) — ★★ **NEGATIVE (원인 규명)** (2026-07-14)

RUN=`260713_Y26S_P2_SAQ_A1b_VISDRONE`(RUN_260713_01). SAQDetect `saq_rescore=True`: q 를 **one2one reg feature**
에서 L_q 학습 + 추론 `_inference_rescore`(score=cls·q^γ). naive-P2 동일세팅(batch32/300ep/seed0). 246ep, best ep196.

> **한줄**: rescoring 이득 0(전 결합식 ≤ 기준). 원인 = **q 가 one2one 에서 IoU 와 *반대로* 학습됨**(Sp small **−0.335**,
> A1a-1 one2many 의 +0.504 대비 부호 반전). one2many/one2one 딜레마(7Q 발견1)가 치명타. **개념 여지(oracle +0.046)는 실재.**

### ① val 최종 (max_det 300, rescored) — 수렴, 미세 열세
| 지표 | naive-P2(best ep191) | **A1b**(best ep196) | Δ |
|---|---|---|---|
| mAP50-95 | 0.2611 | 0.2587 | **−0.0024** (동률·노이즈) |
| mAP50 | 0.4264 | 0.4210 | −0.0054 |
| recall | 0.4158 | 0.3949 | **−0.0209** (score 압축 흔적) |

- **중간판독 추이(mAP50-95 Δ)**: ep50 −0.0041 → ep75 −0.0014 → ep96 −0.0010 → 최종 −0.0024. **초기 열세가 닫힘**
  = L_q 가 검출망을 크게 교란 안 함(base 대체로 무해 시사, detach 필요성↓). 단 **recall −0.02 는 잔존**(rescoring 곱셈 압축).
- ⚠ val 은 **max_det 300**. 판정 지점(**test max_det 1500, conf0.001**)과 다름 — rescoring 여지(+0.12 oracle)는 저conf·고후보 지점.

### ② test 판정 3종 (test @maxDet1500, 재학습 0)

**base 체크(γ0, rescoring OFF) — 검출망 무해 확인 ✅**
| band | naive-P2 AP/AR | A1b γ0(base) | → base 교란 없음 |
|---|---|---|---|
| small | 0.1402/0.3221 | 0.1404/0.3200 | ≈ 동일 (L_q 가 검출망 안 망침) |
| all | 0.1841/0.3452 | 0.1852/0.3452 | ≈ 동일 |

**as-deployed(γ1 rescoring) — 이득 없음**
| band | naive-P2 | A1b γ1 | Δ |
|---|---|---|---|
| small AP | 0.1402 | 0.1391 | **−0.0011** (노이즈, 미세 음성) |

**결합식 probe(one2one, 결정적) — small AP:**
| 결합식 | small AP | Δ vs cls |
|---|---|---|
| cls (기준) | 0.2591 | — |
| 학습 q 최선(small<32·q^0.5) | 0.2579 | **−0.0012** |
| cls·q^1 / q^2 | 0.2545 / 0.2504 | −0.0046 / −0.0086 |
| **ORACLE cls·IoU^2 (완벽 q)** | **0.3052** | **+0.0461** |

→ **완벽 q 는 +0.046 (여지 실재!), 학습 q 는 모든 결합식(γ·boost·small-only)에서 ≤ 기준(회수 ~0%).**
(one2one probe 는 장당 ~400 det = 배포 operating point, oracle 정상작동 = 경로 유효 = one2many probe 의 INCONCLUSIVE 와 대조.)

### ③ 원인 규명 (smoking gun) — one2one q 가 거꾸로 학습됨
`saq_quality_diag.py`(A1b, one2one 자동선택) Sp(q,IoU):
| | small | medium | Sp(q,cls) |
|---|---|---|---|
| A1a-1 (q on **one2many**) | **+0.504** | 0.065 | 0.775 |
| **A1b (q on one2one)** | **−0.335** | −0.501 | 0.581 |

- **확정된 것 = "one2one q 가 IoU 와 역정렬"(부호 반전 +0.50→−0.33)** 까지. **왜인지는 미분리(Codex 지적)**:
  ⓐ 억제 anchor reg feature 정보결핍(유력 가설) ⓑ one2one assigner 가 좋은 duplicate 를 의도적 저score 화 ⓒ VFL non-pos weight
  상호작용 ⓓ q 가 "대표(top) anchor 여부"를 학습(t>0.1 집합) — **확정 표현 금지, 가설로 유지.**
- **딜레마(확정)**: q 를 **잘 배우는 곳(one2many, +0.50)은 추론에 안 쓰이고, 추론경로(one2one)에선 역정렬**. = 7Q **발견1** 현실화.

### ④ 판정 & 다음 (Codex 라운드2 반영)
**A1b = NEGATIVE** (SGI·GSA·P2Refine·AFFG·SPD·NWD-assign 에 이어 ranking 축 첫 실패).
- **정확한 표현(Codex)**: "실패=개념 문제" 아님, "**oracle 기준 ranking 여지는 남으나 현재 SAQ 의 learnable q 로 회수 못함**".
- ⚠ **oracle +0.046 은 combo probe *내부* 상대 headroom** (letterbox640·raw one2one 후보, base small AP 0.2591 ≠ 공식 0.1404).
  **공식 AP headroom 으로 인용 금지.** ranking 축 공식 여지 근거 = premise 진단 **+0.12(naive-P2, 공식 operating point)** 쪽.
- **rescue = 낮은 ROI(Codex)**: 공식 small AP γ0 0.1404→γ1 0.1391(배포효과 0), combo 학습 q 전부 음수, q-cls 중복 0.58~0.78 →
  기대 +0.02(노이즈·기여성 애매). → **full train 강추 아님.**
- **하려면 값싼 smoke-gate 먼저(Codex)**: shared/distill q 가 **one2one box IoU 와 small Spearman ≥ +0.2~0.3** 나오는지 선확인.
  후보: **R1**(공유 feature) < **distill**(one2many q teacher→one2one q student) or **top-candidate 중심 target/weight** (더 정합적).
- **신규 툴**: `eval_size_ap.py --saq-gamma`, `eval/saq_combo_probe.py`.

**정직 판정: A1b negative. ranking 여지는 (probe/premise 상) 남으나 learnable q 로 회수 실패. rescue 는 값싼 smoke-gate 통과 시에만.** 대결 `../03_modules/SOD_SAQ_A1B_CODEX_BRIEF.md`.

### ⑤ cross-path gate + AP (2026-07-14~15, 재학습0) → **SAQ ranking 축 최종 종료**

Codex #2(box mismatch 치명?) 검증 위해 A1a-1 모델(one2many q, +0.50)을 `saq_rescore=False` 로 로드해 **one2many q 로
one2one 을 평가**. 도구 `eval/saq_crosspath_gate.py`(랭킹), `eval/saq_crosspath_ap.py`(AP).

**gate (랭킹 상관, one2many q vs one2one 박스 IoU, val):**
| band | Sp(q, IoU_o2o) | Sp(cls_o2o, IoU_o2o) |
|---|---|---|
| small | **+0.686** | −0.051 |
| tiny/medium | +0.735 / +0.448 | +0.280 / −0.023 |
→ **one2many q 가 one2one 박스 IoU 를 훌륭히 예측(+0.69), one2one cls 는 무상관(−0.05).**
**Codex #2(box mismatch) 반박**: q transfer 잘 됨. A1b 실패는 mismatch 아니라 **"one2one 에서 q 학습" 자체**로 국소화.

**AP (one2many q 로 one2one 검출 rescore, test @maxDet1500, probe-내부 상대 AP):**
| 결합식 | small AP | Δ |
|---|---|---|
| cls (기준) | 0.2599 | — |
| 학습 q 최선(small<32·q^1) | 0.2622 | **+0.0022 (노이즈)** |
| ORACLE cls·IoU² | 0.3075 | +0.0476 |

→ **랭킹 +0.69 에도 실제 AP 는 +0.0022(노이즈), oracle 여지의 ~5%만 회수.**
**원인(핵심 교훈)**: **랭킹 상관 ≠ AP.** one2one 은 이미 객체당 최선 박스를 top 에 올림 → q 로 재정렬해도 채택 detection 거의
불변. oracle(완벽)만 숨은 더 좋은 박스 승격(+0.048), Spearman0.69 q 는 top 케이스서 오차가 이득 상쇄 → 순이득 ≈0.

### ★ SAQ 최종 판정 (2026-07-15): **ranking 축 종료 (negative, 완전 규명)**
4단계 전부 확인: ① q 학습가능(o2m +0.50) ② one2one 학습 시 붕괴(−0.33) ③ 좋은 q 는 o2o 박스도 잘 랭킹(+0.69, box mismatch 아님)
④ **그래도 AP +0.002(노이즈)** = learnable q 로 여지 회수 불가. **oracle +0.048 여지는 실재하나 도달 불가.**
- **논문 가치(정직 negative)**: "quality-aware rescoring 은 oracle headroom 있으나 **learnable q 로 회수 안 됨 — 랭킹상관이 AP 로
  전이 안 되는 이유(이미 정렬된 top 지배)**". Codex #4(ROI 낮음) 옳음 / #2(box mismatch) 반증됨.
- **→ SAQ 폐기.** recall 축(SPD·NWD)·precision 축(SAQ) **양축 모두 벽**. 확정 기여 = **naive-P2 헤드**. 다음 = 전략 재검토.
