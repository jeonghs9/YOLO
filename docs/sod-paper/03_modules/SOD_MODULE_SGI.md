# SGI — Semantic-Guided Injection (구조 모듈)

브랜치: `2-fpn`. 코드: `ultralytics/nn/modules/block.py: SGI` / config: `yolo26-p2-sgiV0~V3.yaml`.

> ## ⚠️ 최종 판정 (2026-06-30): negative — 의미 주입 무효
> 5종(V0~V3+V3noattn) 학습 결과(`../04_results/SOD_RESULT_AP_SIZE.md`): **소스 주입만(V3noattn) = naive-P2** (tiny +0.0004),
> attention만(V0)이 오히려 최고. → **cross-level 의미 주입(이 모듈의 novelty)은 효과 없음.**
> 이유: naive-P2의 P2 feature가 이미 top-down PAN으로 의미를 받아 "semantic gap"이 이미 메워져 있어 추가 주입이 중복.
> → **SGI는 논문에서 negative ablation**("명시적 의미 주입은 top-down이 이미 제공해 불필요")으로 활용.
> 아래 설계/검증 기록은 보존(방법론·코드 정확성은 유효). 다음 레버는 assignment/detail/해상도.

---

## 1. 아이디어 (왜 이 모듈인가)

naive-P2 진단: 작은 물체를 **찾기는 하나(recall↑) 못 맞춤(AP 정체)** = **semantic gap**. P2(고해상)는 의미가 빈약해 "찾은 것"을 확신 있게 분류·정밀화 못 함.

**SGI의 임무:** 깊은 층(P3/P4/P5, 의미 풍부)의 의미를 **고해상 P2에 "선택적으로" 주입**해 recall을 AP로 전환.

> **초급자 설명:** P2는 "시력은 좋은데 글을 못 읽는 학생". 깊은 층은 "글은 잘 읽는 선배". SGI는 **선배의 이해를 학생에게, 그것도 "중요한 내용을 중요한 자리에" 콕 집어** 전달해주는 모듈이에요.

---

## 2. 수식 (코드와 1:1)

입력: `[base, *sources]` (base=P2, sources=깊은 층). `P_i`=1×1 conv 투영, `U`=base 크기로 upsample.

```
base   = P_0(x_0)
w      = softmax(src_w)                    # 소스(P3/P4/P5)별 학습 가중치, 합=1 (init 균등=mean)
F_sem  = base                              (소스 없음 = V0: 자기참조)
       = Σ_i w_i · P_i(U(x_i))             (V1: P3 / V2: P3+P4 / V3: P3+P4+P5)
a_c    = σ( W2 · ReLU( W1 · GAP(F_sem) ) )         # 채널 attention "무엇"(what), (B,C,1,1)
a_s    = σ( Conv7x7( [mean_c F_sem ; max_c F_sem] ) )  # 공간 attention "어디"(where), (B,1,H,W)
out    = base + γ · ( a_s ⊙ ( a_c ⊙ F_sem ) )
```
- `σ`=sigmoid, `⊙`=원소곱(브로드캐스트), `GAP`=global average pool, `mean_c/max_c`=채널축 평균/최대.
- **`src_w` 초기값 0** → softmax 균등 → init시 단순 평균(mean)과 **수치 동일**. 학습으로 해로운 소스(예: 8× 업샘플된 P5)를 알아서 down-weight. 비용 0(스칼라 n개).
- **base path = Identity** (P2 채널=출력 채널이라 무변환): `γ=0`이면 `out = 원본 P2(x_0)`와 **정확히 일치**. → "+SGI"는 1×1 conv를 덧댄 게 아니라 **순수 residual add-on**(naive-P2와 init시 완전 동일 — 논문 방어 핵심).
- **`γ`(gamma) 초기값 0** → 시작 시 `out = 원본 P2` (주입항=0, 안전 초기화). γ는 **제약 없는 파라미터**라 학습 중 음수도 가능 → "더하기만"이 아니라 **주입의 세기·방향(부호)을 학습으로 조절**.

> **초급자 설명 (수식 풀이):**
> - `w` = "선배(P3/P4/P5) 중 누구 말을 더 들을지 모델이 스스로 정하는 비중" (처음엔 똑같이, 학습하며 조절).
> - `a_c` = "수백 개 정보 채널 중 의미 있는 것만 키우는 볼륨 조절기".
> - `a_s` = "사진에서 물체가 있을 법한 자리에만 불 켜는 스위치".
> - 마지막 줄 = "원본 P2(base)는 그대로 두고, **선배 의미(F_sem)를 '중요 채널(a_c)×중요 위치(a_s)'로 걸러서 살짝 더해준다**". γ로 얼마나 더할지 학습.

---

## 3. 기존 연구와 무엇이 다른가 (창작성)

| 기존 | 방식 | SGI의 차별점 |
|---|---|---|
| PAN (YOLO 내장) | 얕은+깊은 단순 concat | 단순 결합 아님 — **의미 소스가 직접 "어디·어떤 채널"에 주입할지 guide** |
| BiFPN | 입력별 **스칼라** 가중치 | 스칼라 아님 — **채널별·위치별(per-pixel) content-aware 주입** |
| AugFPN | **최상위** 레벨 보강 | 반대로 **최하위 고해상(P2, tiny가 사는 곳) 보강** |
| CBAM (self-attention) | 자기 자신에 attention | **cross-level** — 깊은 층(의미)이 P2를 guide (자기참조 아님; V0가 그 대조군) |
| CARAFE | upsample만 개선 | **residual 주입으로 detail 보존** + 의미 주입을 한 모듈에 결합 |

**핵심 창작성:** "의미가 풍부한 깊은 층"이 **"의미가 빈약한 고해상 층"의 어디에 무엇을 주입할지 스스로 결정(semantic-guided)**하고, **residual로 고해상 detail은 보존**. 이를 **소형 객체 전용**으로 설계하고 **size별 AP(vt/tiny/small)**로 검증.

> **초급자 설명:** 남들은 "그냥 섞거나(PAN)", "전체 볼륨만 조절(BiFPN)", "큰 물체 쪽을 보강(AugFPN)"했는데, 우리는 **"작은 물체가 사는 고해상 층"에, "똑똑한 선배가 직접 골라서", "원본은 안 망치고" 의미를 넣어줌**. 이게 우리만의 조합이에요. (정직: 각 요소는 알려진 것이나, 이 **결합 + 소형객체 타깃 + 위치별 guide**가 우리 기여.)

---

## 4. 검증 (코드가 수식과 일치하고, 안전한가)

스모크 테스트 (`block.py: SGI`):
- **수식 == 코드**: 수식대로 독립 재계산 → 모듈 출력과 `allclose = True` ✅ (forward가 위 수식을 그대로 구현, 숨은 연산 없음)
- **게이트 범위**: `a_c, a_s ∈ [0,1]` (sigmoid) ✅
- **★ 진짜 안전 초기화**: base=Identity + γ=0 → `out == 원본 P2(x_0)` **정확히 일치** ✅ → init시 naive-P2와 **완전 동일**(전엔 변환된 P2였음, 2차 검토 수정).
- **attention-off 대조군** 경로(`out=base+γ·F_sem`) ✅, **V0(self) + gradient + 유한성** ✅
- **5 변형(V0/V1/V2/V3/V3noattn) 빌드 + E2E(one2many/one2one) loss + backward** 정상, strides [4,8,16,32] ✅

비용 (params / GFLOPs @640, base=Identity로 더 가벼워짐):
| | params | GFLOPs | vs naive-P2 |
|---|---|---|---|
| baseline | 10.01M | 22.84 | — |
| naive-P2 | 9.77M | 27.79 | 기준 |
| SGI-V0 | 9.77M | 27.80 | +0.0% |
| SGI-V1 | 9.78M | 27.91 | +0.4% |
| SGI-V2 | 9.79M | 27.96 | +0.6% |
| SGI-V3 | 9.82M | 27.99 | +0.7% |

→ **모듈이 매우 가벼움** (V0는 사실상 동일). P2 자체 비용(+22%)에 비하면 SGI는 거의 공짜.

> **초급자 설명:** "코드가 수식대로 동작하는지" 두 번 계산해 같은지 확인했고(allclose True), 이번엔 **"끄면 진짜 원본 그대로"**가 되게 고쳐서(전엔 복사기 한 번 거친 P2였음) "그냥 conv 하나 더한 효과"라는 반박을 막았어요. 속도는 거의 안 느려집니다(+1% 미만).

---

## 5. 실험 (V0~V3 ablation 사다리)

방식 M2 고정, **의미 소스 깊이만** 비교:

| 실험 | SGI from | 구성 | 무엇을 증명 |
|---|---|---|---|
| naive-P2 | — | P2 + P3 단순 concat | 기준선(이미 있음) |
| **V0** | `[19]` | P2 self-attention만(소스 X) | "attention만으로 되나?" (대조군 1) |
| **V3noattn** | `[19,16,13,10]`, attn off | P2 + 의미주입(attention X) | "cross-level 소스 자체 효과?" (대조군 2) |
| **V1** | `[19,16]` | P2 + P3 주입 | "가까운 의미 주입이 되나?" |
| **V2** | `[19,16,13]` | P2 + P3+P4 | "중간 깊이까지" |
| **V3** | `[19,16,13,10]` | P2 + P3+P4+P5 | "최심부까지" |

> 대조군 2개로 효과를 분해: V0(attention만)·V3noattn(소스만)·V3(둘 다) 비교 →
> "이득이 attention 때문인가, cross-level 의미 주입 때문인가"를 갈라 novelty 방어.

학습 (8 GPU, 2개씩 병렬), baseline 조건 동일(muSGD, 300ep, seed=0):
```bash
yolo detect train \
  model=".../cfg/models/26/yolo26s-p2-sgiV1.yaml" \
  data=".../dataset/VisDrone/cleaning/data.yaml" \
  epochs=300 patience=50 batch=32 imgsz=640 device=<2개> workers=4 seed=0 \
  project=".../runs/FPN" name="260626_Y26S_SGIv1_VISDRONE"   # V0~V3 각각
```

평가 (4종): val + pycocotools 구간별 AP + **tiny AP50/AP75** + GFLOPs/FPS.

### ★ 성공 판정 (2026-06-26 2차 검토 반영)

tiny matched-IoU 진단(`../04_results/SOD_RESULT_AP_SIZE.md`): tiny 실패는 **두 겹** —
- **검출(AP50/recall)**: ~45% 미검출 → SGI가 공략할 영역 (의미 부족).
- **정밀도(AP75)**: 찾은 것도 평균 IoU 0.65 → localization 품질이 거의 안 오름 (원인은 기하 sensitivity 추정, 아직 미분리).

→ **헤드라인 지표는 APtiny / APsmall 그대로 유지** (SOD 관례, cherry-picking 오해 방지).
**AP50 / AP75 / AR은 "실패 원인 분해" 진단 지표**로 사용 (지표를 바꾸는 게 아님).

성공 = **복합** 기준 (naive-P2 대비):
1. **APtiny 상승** (헤드라인) — 단, 그 상승이 **AP50(검출) 기여인지** AP75(정밀)인지 분해해 보고.
2. tiny recall / AR 유지·상승.
3. 전체 AP 무손상 + 비용 작음.
4. (진단) AP75/AP50 비율 — 오르면 정밀도까지 개선(보너스), 안 올라도 검출 개선이면 SGI 역할 달성.

### ★ 학습 중/후 점검 — gamma · src_w (3차 검토 반영)

γ=0 초기화는 안전하나, **모듈 내부(source/attention) gradient가 1스텝 늦게 열림.** 학습이 실제로 됐는지 확인 필요:
```bash
python eval/inspect_sgi.py ultralytics/runs/FPN/260626_Y26S_SGIv3_VISDRONE/weights/best.pt
# 학습 중간 점검은 last.pt 로
```
출력: 각 SGI의 `gamma`, `src_w(softmax)`(P3/P4/P5 선택 비중).
- **γ가 거의 0으로 남으면** → 모듈 미학습 의심 → **`gamma` init을 0.01로 바꿔 재실험** (안전초기화는 약간 포기, 학습성 확보).
- **src_w**는 "tiny에 어느 깊이가 유리한가" 해석 결과 → 논문에 보고 (예: P3 우세 = 인접 의미가 유효).

### 다음
V0/V3noattn/V1~V3 비교 → 최고 선택 → **baseline·naive-P2·best ×3 seed**(분산 확정) → loss/detail 재도입(AP75 분업은 SGI 결과 본 뒤 결정, 지금은 보류) → DOTA/AI-TOD 확장.

> **학습 시작 체크리스트:** ① `git switch 2-fpn` (yaml/코드는 2-fpn에만) → ② `yolo26s-p2-sgiVx.yaml`(s 접미사) 호출 → ③ V0~V3+V3noattn 학습 → ④ `inspect_sgi.py`로 γ·src_w 확인 → ⑤ pycocotools 구간별 AP + tiny AP50/AP75 + matched-IoU.
