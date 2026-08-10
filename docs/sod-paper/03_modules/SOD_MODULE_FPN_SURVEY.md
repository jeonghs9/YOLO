# FPN/PAN 계열 비교 분석 — SOD 관점에서 우리 모듈의 빈틈 찾기

목적: 구조 모듈(기여 1) 설계 전, 기존 multi-scale fusion 방법이 **①갭 완화 ②detail 보존**을 어떻게 했는지 정리하고, **극소 객체(8-16px) 관점의 빈틈**을 특정한다. (단순 "갭 줄임"은 기여 아님 — 구체적·차별적 메커니즘 필요.)

---

## 1. 핵심 방법 비교

| 방법 (연도) | 핵심 아이디어 | 갭 완화 방식 | detail 보존 | SOD 관점 한계 |
|---|---|---|---|---|
| **FPN** (Lin 2017) | top-down + lateral 1×1 | 깊은 의미를 위로 전달 | nearest/bilinear upsample | top-down 중 의미 희석; 얕은 레벨은 여전히 저의미. 고해상 레벨 없음 |
| **PANet/PAN** (Liu 2018) | + bottom-up path | 위치 정보를 아래→위로 재전달 | 단순 concat | **이미 YOLO26 내장.** 융합이 단순 concat, 레벨 동등 취급 |
| **BiFPN** (Tan 2020, EfficientDet) | 가중 양방향 융합 반복 | 입력별 **학습 가능 스칼라 가중치** | 반복 블록 | 가중치가 **채널/공간 인지 아님**(스칼라). 반복 비용. 고해상 P2 미포함 |
| **NAS-FPN** (Ghiasi 2019) | 토폴로지 탐색 | 탐색된 융합 경로 | — | 불투명, SOD 비특화, 무거움 |
| **AugFPN** (Guo 2020) | consistent supervision + RFA + soft-RoI | 레벨 간 의미 일관성 강제 | residual feature augmentation | 2-stage용. RFA가 **최상위 레벨** 보강 — 극소 객체(최하위 고해상)엔 직접적이지 않음 |
| **CARAFE** (Wang 2019) | content-aware upsampling | (업샘플 연산자) | **예측 커널로 detail 보존** | 업샘플만 개선 — 융합 전체 재설계는 아님 (보완재) |
| **고해상 P2 head** (YOLO-p2 등) | stride-4 검출 레벨 추가 | — | 고해상 자체 | **얕은 P2(저의미) + 단순 융합 → 갭 그대로**. 비용 급증 |
| **RFB / attention fusion** | 수용영역·attention 융합 | 채널/공간 attention | 일부 | 일반 검출용, 극소 특화·size별 검증 부족 |

---

## 2. 정리 — 두 축으로 보면

- **해상도 축 (①정보 부재 해결):** 대부분 변형은 **P3가 최하위** → 8-16px엔 셀이 부족. P2를 더해야 하지만 단독 P2는 비용↑·저의미.
- **융합 품질 축 (①갭 완화 + ②detail):** BiFPN(스칼라 가중)·AugFPN(상위 보강)·CARAFE(업샘플만) — **고해상 레벨에서 "얕은 고해상 + 깊은 저해상"을 의미적으로 정렬해 융합**하는, **극소 객체 전용** 설계는 비어 있음.

---

## 3. 우리가 비집고 들어갈 빈틈 (novelty 포지셔닝)

> **"고해상(P2/P3) 레벨에서, 깊은 의미를 content/공간 인지 방식으로 주입하면서 미세 detail을 보존하는, 극소 객체 전용 융합 모듈."**

기존 대비 차별점 (리뷰어 "BiFPN/AugFPN과 뭐가 다르냐" 대응):
1. **스칼라 가중치(BiFPN)가 아니라 content/공간 인지 융합** — 어느 위치·채널에 의미를 주입할지 선택.
2. **최상위 보강(AugFPN)이 아니라 최하위 고해상 레벨 보강** — 극소 객체가 사는 곳을 직접 공략.
3. **업샘플만(CARAFE)이 아니라 융합+detail 보존 결합** (CARAFE/high-freq residual을 모듈 내부 요소로 흡수 가능).
4. **검증 차별:** 대부분 FPN 논문이 AP만 보고 — 우리는 **vt/tiny/small/medium 구간별 AP**로 "진짜 8-16px를 올린다"를 입증 (우리 평가 파이프라인의 강점).

---

## 4. 설계 후보 메커니즘 (Step 3에서 구체화)

- **의미 주입(semantic injection):** 깊은 feature를 고해상으로 올릴 때 channel/spatial attention 또는 gating으로 "선택적 주입" (단순 concat/합산 X).
- **detail 보존:** content-aware upsample(CARAFE류) 또는 high-frequency residual skip (백본 얕은 feature의 고주파 성분 보존).
- **정렬(alignment):** 얕은-깊은 feature 간 misalignment 보정(deformable/offset) — 옵션.
- → 위 중 1~2개를 우리 데이터(8-16px 병목)에 맞춰 조합. 과욕 금지(ablation 가능하게 최소 구성).

---

## 5. 참고 (정식 인용은 작성 시 보강)
- FPN: Lin et al., CVPR 2017. PANet: Liu et al., CVPR 2018. BiFPN/EfficientDet: Tan et al., CVPR 2020.
- NAS-FPN: Ghiasi et al., CVPR 2019. AugFPN: Guo et al., CVPR 2020. CARAFE: Wang et al., ICCV 2019.
- SOD 서베이/NWD-RKA 등은 `../06_handoff/SOD_HANDOFF_NWD_EXPERIMENT.md`·loss.md 참조.

> 다음(Step 2): naive P2 ablation으로 "고해상만의 천장·비용"을 측정 → 이 모듈의 순수 가치를 분리할 기준점 확보. (`2-fpn` 브랜치)

---

## 6. 추가 서베이 — adaptive/selective fusion (2026-07-07, head refine negative 이후)

SGI/GSA(단순 주입) negative + P2Refine(정보 재배치) negative 후, **"gap 줄이기"가 아니라 "gap 조절/선택"** 계열 재조사.

| 방법 | 핵심 메커니즘 | tiny 관점 | 우리 데이터와 연결 |
|---|---|---|---|
| **Effective Fusion Factor** (WACV2021, [2011.02298](https://arxiv.org/abs/2011.02298)) | top-down 융합량을 **스칼라 fusion factor α** 로 조절(데이터셋 통계로 추정, 고정) | top-down 이 tiny 에 **양면적** — 과하면 해로움, α 낮춰야 tiny↑ | **SGI 실패와 정확히 일치** — P2 의 deep 의미가 과잉(주입 중복) |
| **ASFF** (2019, [1911.09516](https://arxiv.org/abs/1911.09516)) | 픽셀별 학습 spatial weight(1×1+softmax)로 레벨 융합 | conflicting 억제, scale-invariance(모든 크기 대칭) | 위치별 선택 메커니즘(단 tiny 특화 아님) |
| **EFPN / FTT** (2020, [2003.07021](https://arxiv.org/abs/2003.07021)) | deep 을 sub-pixel conv **super-res** + texture 복원(정보 생성) | 고해상 레벨 신설 | refine 실패(정보부족)의 대안이나 "생성"은 어렵고 SGI류 위험 |
| BiFPN (2020) | 학습 **스칼라** 가중, 양방향 반복 | 채널/공간 인지 아님 | 스칼라 한계 |
| PANet(YOLO 내장)·MSE-FPN·TL-AFPN·IHP·Rethinking-FPN | bottom-up / semantic enhance / adaptive fusion 변형 | 대체로 범용 | tiny-specific 억제는 비어 있음 |

### ★ 핵심 통찰 (방향 전환)

이 논문들 + 우리 데이터(SGI negative, oracle "P2 의미 과잉")가 **같은 결론**을 가리킨다:

> **tiny 엔 semantic gap 을 "무조건 줄이면(SGI식 주입)" 오히려 해롭다. "얼마나·어디에" 융합할지 조절/억제해야 한다.**

→ SGI(주입)의 **정반대 = 선택적 억제** 모듈. Fusion Factor(전역 스칼라)를 **크기조건부 공간 게이트**로
확장. 설계·인수인계: **`../03_modules/SOD_MODULE_AFFG.md`**.
