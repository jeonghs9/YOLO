# AFFG — Adaptive Fusion-Factor Gate (P2 top-down 선택적 억제) — 설계·인수인계

> **⛔ 상태: 폐기 (2026-07-08).** Phase 0 α=0.5 관문 **실패** — top-down 억제가 tiny AP **−0.0135**,
> 전체 AP **−0.0307**(`../04_results/SOD_RESULT_AP_SIZE.md`). "top-down 과잉" 가설 기각, VisDrone tiny 는
> top-down 이 오히려 필요. **AFFG 게이트 미구현**(Phase 0 관문 원칙대로 낭비 0). 아래 설계는 방법론·참고용 보존.

작성 2026-07-07 (Codex 인수인계용). `../03_modules/SOD_MODULE_SGI.md`·`../03_modules/SOD_MODULE_GSA.md`(주입, negative)·`../03_modules/SOD_MODULE_HEAD_REFINE.md`
(refine, negative)에 이은 **neck(fpn) 트랙 재도전**. 핵심은 SGI 의 **정반대 — 주입이 아니라 억제**.

> **[2026-07-07 Codex 1~4차 리뷰 반영]** ① identity init: **마지막 conv weight=0 AND bias=0**
> (bias 만으론 부족) → `factor=2σ`=1=naive-P2; `1+β·tanh` 금지(γ=0 닭-달걀) · ② factor∈(0,2)=
> **억제+완만한 증폭 둘 다 허용**(가설 강제 X — 학습된 factor<1 로 검증) · ③ 전역 α 없이 **단일
> pixel-wise factor**(고정 α 는 Phase 0 에만) · ④ Phase 0 **α screening: 0.5→0.25/0.75→0.0**
> (α=1.0=기존 naive-P2) · ⑤ 판정 **tiny AP50/AP75/AR + small/med 분해**(단일 금지) · ⑥ **eval by-size
> AP50/75 + factor-map inspect 도구 선결** · ⑦ P2 융합 = **layer 18, 입력 [17,2]**, AFFG=concat 포함 (**output ch = ch[17]+ch[2] = Concat 동일**,
> args 는 gate hidden 만·c2≠output) · ⑧ Phase 0 는 **ScaleConcat** 모듈로 · ⑨ 1차 free gate(density 금지).

---

## 1. 동기 (왜 AFFG 인가)

우리 데이터가 확정한 것:
- **SGI/GSA(deep→P2 의미 주입) = negative** (`../04_results/SOD_RESULT_AP_SIZE.md`). 이유: P2(L19)는 top-down PAN 으로
  **의미가 이미 과잉** → 추가 주입은 중복.
- **P2Refine(위치 재배치) = negative** (`../03_modules/SOD_MODULE_HEAD_REFINE.md`). 이유: tiny feature 빈약 → offset 못 배움.
- 즉 tiny 병목은 "의미 부족"도 "회귀 방식"도 아니었다.

외부 근거 (`../03_modules/SOD_MODULE_FPN_SURVEY.md` §6):
- **Effective Fusion Factor (WACV2021)**: FPN top-down 이 tiny 에 **양면적** — deep 융합이 과하면 해롭고,
  fusion factor α 를 **낮춰야** tiny AP 가 오른다. **우리 SGI 실패와 같은 현상.**

→ **가설: tiny 위치에서는 P2 로 내려오는 deep 융합을 "억제"해야 한다.** SGI 는 이걸 거꾸로(주입) 해서 실패.

## 2. 설계 (★ identity-init — Codex 1·2차 리뷰 반영 2026-07-07)

YOLO26 neck 의 P2 는 **layer 18** `Concat[Up(P3td), backbone_P2]`(top-down). top-down 브랜치에
**단일 픽셀별 fusion factor** 를 곱한다(**별도 전역 스칼라 α 는 없음** — 고정 α 는 Phase 0 sweep 에만):

```
factor(x,y) = 2 · σ( Conv_last( Conv*([Up(P3td), backbone_P2]) ) )   # (B,1,H,W) ∈ (0,2)
P2_out      = Concat[ factor ⊙ Up(P3td), backbone_P2 ] → C3k2
```

- **★ identity init (결정적, Codex 2차):** 마지막 conv 는 **weight=0 AND bias=0 → logit≡0 →
  factor init=1 → AFFG init == naive-P2**(정확히). ⚠ **bias=0 만으로는 부족** — weight 가 random 이고
  입력이 nonzero 면 logit≠0 이다. 반드시 **last conv weight=0, bias=0** 둘 다.
- **factor ∈ (0,2) = 억제 + 완만한 증폭 둘 다 허용 (Codex):** 억제를 **강제하지 않는다.** 가설은
  hard-code 가 아니라 **학습된 factor 가 tiny 주변에서 <1 로 내려가는지로 검증**한다.
  > *AFFG is initialized as the standard P2 fusion and allows both attenuation and mild
  > amplification. The hypothesis is not hard-coded; it is tested by whether learned factors
  > fall below 1 around tiny objects.* (리뷰어 "왜 >1 허용?" 대응)
- **왜 `2·σ` 인가 (우리 실증):** 대안 `1+β·tanh(logit)`(β init 0)은 β=0 이면 `∂factor/∂logit=0` →
  게이트가 gradient 를 못 받아 **안 배운다** = P2Refine 의 **γ=0 닭-달걀**
  (`../03_modules/SOD_MODULE_HEAD_REFINE.md`). `2·σ`(logit≡0 시작)는 factor=1 이면서 logit gradient
  가 흐른다 → **채택**.
- **1차 구현은 free gate**: density/GT-prior 조건 **금지**(hand-crafted dataset prior 비판, Codex).

파라미터: Conv 소수(경량). **마지막 conv weight=0·bias=0 init**. inspect 용 factor map 노출(§5).

## 3. 기존 대비 무엇이 다르고 창의적인가 (novelty)

| 대비 대상 | 기존 | AFFG | 차별점 |
|---|---|---|---|
| **Fusion Factor** (WACV21) | 전역 **스칼라** α, 데이터 통계로 **고정** | **identity-init pixel-wise factor** f(x,y)=2σ(logit) (별도 global α 없음) | 전역 스칼라→**픽셀별**, 고정→**학습(identity-init)**, 통계→end-to-end |
| **ASFF** (2019) | 모든 레벨 **대칭** 선택 융합, scale-invariance | P2 top-down 만 **비대칭 억제**, tiny 표적 | 융합(무엇을 더할까)→**억제**(deep 을 얼마나 뺄까), 범용→**tiny-specific** |
| **BiFPN** (2020) | 학습 **스칼라** 가중 | 공간 게이트 | 스칼라→**공간 인지** |
| **EFPN/FTT** (2020) | deep 을 super-res **생성**(정보 추가) | deep 을 **억제**(과잉 제거) | 생성 ↔ **억제** (정반대 철학) |
| **SGI/GSA (ours)** | deep 무조건 **주입**(α=1 고정) | tiny 위치에서 **선택적 억제**(identity init factor=1 에서 학습이 <1 로) | 주입 ↔ **억제** (우리 negative 의 반대 처방) |

**창의적 핵심 3가지:**
1. **"억제" 프레이밍** — 대부분의 FPN 개선은 "더 잘 융합(주입·가중)". AFFG 는 *"tiny 위치에서 deep 융합을
   억제"* 라는 반대 발상. 근거는 **우리 SGI/GSA negative + Fusion Factor** 의 이중 지지.
2. **Fusion Factor 의 공간·학습 확장** — 전역 스칼라(고정)를 **픽셀별 학습 게이트**로. Fusion Factor 와
   ASFF 의 결합을 tiny 억제 목적으로 특화.
3. **negative→처방 스토리** — SGI 가 "왜" 실패했는지(P2 과잉)를 규명하고 **그 반대**를 처방. 논문에서
   SGI/GSA 를 motivation(negative ablation)으로, AFFG 를 해답으로 세우는 서사.

**정직한 리스크 (리뷰어 대응 선제):**
- **incremental 위험**: "Fusion Factor + ASFF 조합 아니냐". → 방어: (a) **tiny 억제**라는 반대 방향,
  (b) **공간 게이트**(스칼라 아님), (c) **우리 negative 로 정당화된 문제 정의**. 단순 조합이 아니라
  "왜 억제인가"의 근거가 novelty.
- **미검증**: Fusion Factor 는 TinyPerson/CityPersons. VisDrone tiny 에서 억제가 이득인지 **미확인** → §4 저비용 검증 필수.
- **gate collapse (양방향)**: factor→1 이면 naive-P2(무효), factor→0 이면 semantic 소실. 게다가
  **전체 평균만 낮으면 spatial 이 아니라 global α** → §5 inspect 로 **tiny 주변 factor** 확인 필수.
- **classification confidence 하락(Codex)**: top-down semantic 억제는 AP75↑ 여도 **AP50/recall 을
  깰 수** 있다 → §4 분해 판정 필수(APtiny 단일 지표 금지).

## 4. 검증 계획 (방법론: `sod-research-method`, Codex 리뷰 반영)

대조군 = **naive-P2(260625)**. 반증 = P2 대비 APtiny Δ<0.007.

### ★ 판정은 APtiny 하나로 하지 말 것 (Codex) — 반드시 분해
억제는 top-down semantic 을 줄이므로 **classification confidence 하락 → AP50/recall 붕괴** 위험.
- **tiny AP50**(검출/분류 살아있나) · **tiny AP75**(위치 정밀도 올랐나) · **tiny AR**(후보 줄었나)
- **small/medium AP**(억제가 전체 구조를 망쳤나)
- **성공 패턴**: tiny AP75↑ · tiny AP50 유지/소폭↑ · tiny AR 유지 · small/medium 큰 하락 없음.
  (tiny AP50 하락 + AP75 만 소폭↑ = 실용적으로 **애매** → 재고)
- ⚠ **도구 갭(선결)**: 현재 `eval_size_ap.py` 는 by-size 에 **AP@[.5:.95]만** 준다. **tiny AP50/AP75
  분해는 eval 확장 필요**(COCOeval `t50`/`t75` slice × area band). **Phase 0 전에 도구부터 만들 것.**

### Phase 0 — 고정 α suppression (모듈 전, 성패가 대부분 여기서 결정)
top-down 브랜치에 **고정 스칼라 α** 만 곱해 재학습(억제 가설의 최소 검증).
- **screening 순서(비용 절감, Codex)**: **α=0.5 먼저** → tiny 개선 없으면 **α=0.25, 0.75** 추가 →
  개선 신호 있으면 **α=0.0** 까지. **α=1.0 = 기존 naive-P2(260625)** 라 재학습 불필요(seed 차 보려면 선택).
  α=0.0 = top-down 완전 차단·backbone P2 만.
- 성공 조건(정량, Codex): 어느 α<1 에서 **tiny AP50/AP75 가 노이즈(±0.007) 초과 상승 AND 전체
  AP@[.5:.95] −0.005 이내(+ small/medium 큰 하락 없음)** → "top-down 과잉" 확인 → AFFG 명분 → Phase 1.
  (⚠ tiny 만 오르고 전체 붕괴 = **기각** — α=0.0 극단에서 흔한 함정)
- **α=0.5 포함 어떤 α 에서도 tiny 안 오름 → 억제 가설 기각 → AFFG 접는다.** 다음은 구조 모듈이 아니라
  **해상도/tiling**(정보를 실제로 만드는 방향, `../03_modules/SOD_MODULE_HEAD_REFINE.md` §다음방향).

### Phase 1 — identity-init AFFG (free gate)
`factor = 2σ(logit)`, **마지막 conv weight=0 AND bias=0**(bias 만으론 부족 → logit≡0=factor 1). 학습(free gate).

### Phase 2 — gate 분석 (스토리 성패, §5 inspect)
factor map 을 **tiny/small/medium GT 주변**으로 분해. **"tiny GT 주변에서만 factor<1"** 이어야
spatial gate 명분. 전체 평균만 낮으면 그건 global α 지 spatial gate 가 아니다(Codex).

### Phase 3 — 판정
분해 지표 eval → `analyze_result.py --base apsize_P2_VISDRONE --exp apsize_AFFG_...` 로 naive-P2 대비.

## 5. 구현 지점 (포크)

- **모델 yaml**: `ultralytics/cfg/models/26/yolo26-p2.yaml` 복사 → `yolo26s-p2-affg.yaml`(s scale).
  **P2 융합 지점 = layer 18** `[[-1, 2], 1, Concat, [1]] # cat backbone P2`. **입력은 `[17, 2]`**:
  - `x_top = layer 17`(upsampled P3td) · `x_lat = layer 2`(backbone P2)
  - AFFG 를 **concat 포함 모듈**로: layer 18 을 `[[17, 2], 1, AFFG, [gate_hidden]]` 로 교체.
    출력 = `Concat[factor⊙x_top, x_lat]`, **채널 = ch[17]+ch[2] (Concat 과 동일)**.
    (factor 는 `[x_top, x_lat]` 로 예측 → 두 입력 필수. args 는 gate hidden 채널만, **c2=output 아님**)
  - 멀티인풋(2입력)이라 `nn/tasks.py` parse_model 채널 분기 필요(SGI/GSA 방식 참고).
- **모듈**: `nn/modules/block.py` 에 `class AFFG` (**pixel-wise factor gate Conv**, 전역 α 없음).
  `__init__.py`·`nn/tasks.py` 등록(`develop-sod-module` 참고). 멀티인풋(2입력) → tasks.py 채널 분기.
  - ⚠ **output channels = ch[x_top] + ch[x_lat] = Concat 과 동일**(다음 C3k2 가 기존과 같은 입력 채널).
    args 엔 **gate hidden 채널만**, **c2 를 output channel 로 쓰지 말 것** — `AFFG,[c2]` 로 output 을
    128 등으로 잡으면 Concat 대체가 아니라 **feature projection 모듈**이 되어 실험이 다른 비교가 된다.
- **inspect (새 스크립트 필요)**: `eval/inspect_sgi.py` 는 **스칼라 gamma 만** 읽음 — AFFG 는 **factor
  map** 분석이 필요하므로 별도 스크립트. 출력: 전체 factor 평균, **foreground vs background factor**,
  **tiny GT 주변 factor**, **small/medium GT 주변 factor**. **"tiny 주변만 <1"** 확인이 스토리 핵심
  (전체 평균만 낮으면 global α 지 spatial gate 아님). **best.pt + last.pt 둘 다** 보고(epoch 중 factor
  추이 확인), factor 통계는 **고정 val/test subset** 에서 계산(이미지마다 달라지지 않게).
- **eval 확장(선결)**: `eval_size_ap.py` 에 by-size **AP50/AP75** 추가(COCOeval `t50`/`t75` × area band).
- **검증 도구**: `develop-sod-module/verify_module.py --registered AFFG --model yolo26s-p2-affg.yaml`
  로 빌드+forward → `sod-research-method/analyze_result.py` 로 판정.
- **Phase 0(고정 α)**: 작은 **`ScaleConcat`** 모듈(상수 α, 학습 파라미터 없음)로 `output=Concat[α·x_top, x_lat]`
  만 수행 → layer 18 대체. (YAML 상수곱이 어려우니 "모듈 없이"가 아니라 **ScaleConcat 모듈 또는 block 한 줄**로
  정확히.) ⚠ **ScaleConcat 도 AFFG 와 똑같이 parse_model 에서 `c2 = ch[17]+ch[2]` 직접 지정**(Concat 과
  동일). output channel 이 틀어지면 **α sweep 자체가 오염**된다.

## 6. Codex 인수인계 체크리스트

- [ ] 브랜치 생성: `2.3-fpn_affg` (관례 loss=1.x, **fpn=2.x**, head=3.x). **main 에서 분기** — main 은
      stock `yolo26-p2.yaml` 보유 + 커스텀 등록(SGI/GSA/P2Refine) 0 건 = 깨끗 **확인됨(2026-07-07)**.
      실험 기준 = **stock naive-P2 와 동일 구조에서 layer 18 만 교체**(P2 yaml 차이 나면 2-fpn stock P2 만 cherry-pick).
- [ ] **0. 도구 선결**: `eval_size_ap.py` 에 by-size AP50/AP75 추가(분해 판정용) + factor-map inspect 스크립트.
- [ ] **1. Phase 0 먼저**: **α=0.5 screening** → 신호 있으면 0.25/0.75/0.0 추가(**α=1.0=기존 naive-P2, 재학습 불필요**).
      **α=0.5 포함 어떤 α 에서도 tiny 안 오르면 AFFG 접고 해상도/tiling 으로.** (모듈 만들기 전 필수 관문)
- [ ] **2. AFFG identity init**: `factor=2σ(logit)`, **마지막 conv weight=0 AND bias=0 → factor=1==naive-P2**
      (⚠ bias=0 만으론 부족). `1+β·tanh`(β=0) 금지(P2Refine γ=0 닭-달걀). factor∈(0,2)라 증폭도 허용(가설 강제 X).
- [ ] **3. 판정은 분해로**: tiny AP50/AP75/AR + small/medium. 성공패턴 = AP75↑·AP50 유지·AR 유지·small/med 유지.
      APtiny 단일 지표 금지.
- [ ] **4. gate 분석**: factor map 이 **tiny GT 주변에서만 <1** 인지 확인(전체 평균 낮음 ≠ spatial gate).
- [ ] 1차는 **free gate**(density/GT prior 조건 넣지 말 것). 필요 시에만 weak reg.
- [ ] 학습 세팅 = naive-P2 동일: batch32, 300ep, imgsz640, seed0, MuSGD(auto). **device 3장(assigner OOM 회피).**
- [ ] AMP dtype 주의(`../03_modules/SOD_MODULE_HEAD_REFINE.md` 교훈): 커스텀 모듈 검증은 **DDP+AMP autocast 재현**으로.
- [ ] negative 여도 정직 기록 → `../04_results/SOD_RESULT_AP_SIZE.md`. 논문엔 방법론만, dtype·속도·OOM 제외.

## 7. 참고

- Effective Fusion Factor: Gong et al., WACV 2021 (arXiv:2011.02298).
- ASFF: Liu et al., 2019 (arXiv:1911.09516). EFPN: Deng et al., 2020 (arXiv:2003.07021).
- 우리 negative 근거: `../04_results/SOD_RESULT_AP_SIZE.md`(SGI/GSA/P2Refine), oracle: `/home/hsjeong/workspace/SOD-PAPER/.claude/skills/sod-research-method/oracle_ceiling.py`.
