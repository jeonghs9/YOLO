# SOD 논문 인수인계 #3 — 2차 검토 반영 보고 (2026-06-26)

> **목적:** 2차 검토(코덱스) 피드백을 **어떻게 반영/수정했는지** 보고하고, **학습 시작 직전 최종 확인**을 받기 위한 문서.
> 2차 검토 요지 §1, 우리 반영 §2, 보류·이견 §3, 학습 직전 상태 §4, 최종 확인 요청 §5.

---

## 0. 한 줄

2차 검토의 **필수 수정(base path Identity)**과 **권장(attention-off 대조군 추가)**을 모두 반영. 표현(gamma 음수 가능, "기하 한계" 완화, APtiny 헤드라인 유지)도 정정. 코드 검증 완료, **학습 직전.**

---

## 1. 2차 검토 핵심 (요약)

1. **(필수) base path가 1×1 Conv+BN+SiLU를 거침** → γ=0이어도 out은 "변환된 P2"지 원본 P2가 아님. "안전 초기화=naive-P2 동일" 주장이 엄밀히 틀림. → 채널 같으면 base는 **Identity**, source만 projection.
2. gamma는 unconstrained → 학습 후 **음수 가능**. "더하기만 한다"는 부정확 → "초기 항등, 학습 중 세기·방향 조절".
3. matched IoU 0.65를 "기하 한계"로 **단정 금지** → "localization 품질이 거의 안 올랐다(원인 미분리)" / "geometric sensitivity" 수준.
4. **APtiny를 헤드라인에서 내리지 마라** (cherry-picking 오해). APtiny/APsmall 유지 + AP50/AP75/AR은 **원인 분해 진단**으로.
5. **대조군 하나 더**: attention 없이 source만 주입(`P2 + γ·F_sem`) → "cross-level source 효과 vs attention 효과" 분리.
6. base projection만 고치면 **학습 시작 OK**. V0~V3 사다리 설계는 좋음.
7. (부수) source-weight는 scalar softmax로 충분, size-aware gate는 1차엔 과함 (이미 동의·반영).

---

## 2. 우리가 반영한 것

### 2-1. (필수) base path → Identity ✅
- 코드: `self.base_proj = nn.Identity() if c1[0]==c2 else Conv(c1[0],c2,1)`. source만 `src_proj`로 1×1 변환.
- 우리 경우 P2 feat 채널 = SGI 출력 채널 (s: 64=64) → **base_proj = Identity.**
- **검증: γ=0 → `out == 원본 x[0]` allclose=True** (전엔 변환된 P2라 False였음).
- 효과: "+SGI"가 1×1 conv 덧댐이 아니라 **순수 residual add-on**. init시 naive-P2와 완전 동일 → 깨끗한 ablation delta. (덤: base conv 제거로 GFLOPs도 감소.)

### 2-2. attention-off 대조군 추가 ✅
- 코드: SGI에 `use_attn` 플래그. False면 `out = base + γ·F_sem` (attention 없는 cross-level 주입).
- config: `yolo26-p2-sgiV3noattn.yaml` (V3 소스, attention OFF).
- 분해: **V0**(attention만, 소스X) / **V3noattn**(소스만, attentionX) / **V3**(둘다) → 이득이 attention인지 cross-level 주입인지 분리.

### 2-3. 표현 정정 ✅
- gamma: "더하기만" → "**제약 없음, 학습 중 세기·방향(부호) 조절**" (코드 주석·`sgi.md` 반영).
- "기하 한계" 단정 → "**localization 품질 거의 안 오름(원인 미분리, geometric sensitivity 추정)**" (`AP_result.md`·`sgi.md`).
- 성공기준: "AP50 중심으로 전환" → "**APtiny/APsmall 헤드라인 유지 + AP50/AP75/AR로 원인 분해**" (`sgi.md` 수정).

### 2-4. 검증 결과 (학습 직전)
- γ=0 → out==raw x0 (allclose) ✅ / 수식==코드 ✅ / 게이트∈[0,1] ✅ / attention-off 경로 ✅
- 5변형(V0/V1/V2/V3/V3noattn) 빌드 + E2E loss + backward 정상, strides [4,8,16,32] ✅
- 비용: GFLOPs naive-P2 27.79 → **V0 27.80(동일) ~ V3 27.99 (+0.1~0.2)**, params ~9.8M.

---

## 3. 보류 / 이견

- **multi-seed**: 코드·계획 반영, 아직 미실행. 실험 사다리 3단계에서 baseline·naive-P2·best SGI만 3-seed 예정(전체 재학습 안 함 — loss는 보조로 격하).
- **loss를 AP75 전담 보조 기여로**: 코덱스 권고대로 **지금은 보류.** SGI가 AP50/AR 올리고 AP75 정체로 나온 뒤, loss/detail 1개로 "AP75 보정" 실험할지 그때 결정.
- **size-aware gate / detail(CARAFE)**: 1차 미포함. SGI 결과 보고 필요시 도입.

---

## 4. 학습 직전 상태 (사실관계)

- 코드: `2-fpn`. `block.py:SGI` (base=Identity, src_proj, use_attn, softmax src_w, γ).
- config (s scale로 학습): `yolo26s-p2-sgi{V0,V1,V2,V3,V3noattn}.yaml`.
  - V0=[19](self) / V1=[19,16](P3) / V2=[19,16,13](P3+P4) / V3=[19,16,13,10](P3+P4+P5) / V3noattn=V3 sources, attn off.
- 기준선: naive-P2(`yolo26-p2.yaml`) — pycocotools AP@.5:.95 0.1841 (baseline 0.1646).
- 학습: muSGD/300ep/patience50/batch32/imgsz640/seed0, 8 GPU 2개씩 병렬(5개 → 일부 순차).
- 평가: val + pycocotools 구간별 AP + tiny AP50/AP75 + matched-IoU + 학습된 src_w.

**학습 명령 (scale 검증 완료):** 디스크 파일은 `yolo26-p2-sgiVx.yaml`(scale 글자 없음)이나,
**`model=yolo26s-p2-sgiVx.yaml`**로 호출하면 ultralytics가 `s` 추론(unified→`yolo26-p2-sgiVx.yaml` 로드).
확인: `yolo26s-p2-sgiV0.yaml` → scale='s', params 9.77M (n이면 ~2.5M). **반드시 `s` 접미사로 호출.**
```bash
yolo detect train model=".../cfg/models/26/yolo26s-p2-sgiV0.yaml" \
  data=".../dataset/VisDrone/cleaning/data.yaml" \
  epochs=300 patience=50 batch=32 imgsz=640 device=0,1 workers=4 seed=0 \
  project=".../runs/FPN" name="260626_Y26S_SGIv0_VISDRONE"
# V1/V2/V3/V3noattn: yaml·device·name만 교체
```

### SGI 최종 수식
```
base  = Identity(x0)                  # 채널 같으면 무변환 → γ=0이면 out==원본 P2
w     = softmax(src_w)                # 소스 깊이별 학습 가중치 (init 균등=mean)
F_sem = base                          (V0) | Σ_i w_i·Proj_i(U(x_i))  (V1~V3)
a_c   = σ(W2·ReLU(W1·GAP(F_sem)))      # 채널 attention (use_attn=True)
a_s   = σ(Conv7x7([mean_c;max_c]))     # 공간 attention
inj   = a_s⊙(a_c⊙F_sem)  (attn on) | F_sem  (attn off, 대조군)
out   = base + γ·inj                  # γ init 0(항등), unconstrained
```

### 진단 결과 (참고 — 무엇을 노리는지)
tiny(8-16px): naive-P2가 검출 48.9%→54.8%(recall↑) but 찾은 것 평균 IoU 0.655→0.662(정체).
→ SGI는 **검출(AP50/recall)** 개선을 노림. 정밀도(AP75)는 localization 한계로 기대 제한적.

---

## 5. 최종 확인 요청

1. **base Identity 수정**이 의도대로(순수 add-on, init=naive-P2 동일) 됐다고 보는가? 추가로 볼 게 있나?
2. **대조군 구성**(V0=attn만 / V3noattn=소스만 / V3=둘다)이 "attention vs cross-level injection" 분리에 충분한가? V3noattn을 V3 소스로 둔 게 맞나, 아니면 best 변형에 맞춰 나중에?
3. **성공 기준 표현**(APtiny 헤드라인 유지 + AP50/AP75 분해)이 이제 cherry-picking 우려를 해소하나?
4. 학습 **시작해도 되는가?** 남은 치명적 결함은?

> 결함 없으면 V0/V1/V2/V3/V3noattn 학습 시작 → 결과로 사다리 비교 + multi-seed 진행.

---

## 6. 파일 맵
| 파일 | 내용 |
|---|---|
| `../06_handoff/SOD_HANDOFF_260626.md` / `_2.md` / `_3.md` | 1차 요청 / 1차 반영 / 2차 반영(이 문서) |
| `../03_modules/SOD_MODULE_SGI.md` | SGI 설계·수식(base Identity·use_attn 반영)·검증·성공기준 |
| `../04_results/SOD_RESULT_AP_SIZE.md` | pycocotools 구간별 AP + tiny matched-IoU 진단 |
| `../00_roadmap/SOD_ROADMAP_PAPER_PROCESS.md` | 전체 흐름·핵심 스토리·서술 전략 |
| 코드 | `2-fpn`: `block.py:SGI`, `cfg/models/26/yolo26-p2-sgi{V0,V1,V2,V3,V3noattn}.yaml` |
