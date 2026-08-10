# SOD 논문 인수인계 #2 — 1차 검토 반영 보고 (2026-06-26)

> **목적:** 직전 검토(코덱스) 피드백을 **어떻게 반영/수정했는지** 보고하고, **학습 시작 전 2차 검토**를 받기 위한 문서.
> 1차 검토 원문 요지는 §1, 우리 반영은 §2, 미반영/이견은 §3, 학습 직전 상태는 §4, 재검토 요청은 §5.

---

## 0. 한 줄

1차 검토의 핵심 5개 지적(주장 톤다운 / 노이즈바닥 / semantic gap 단정 / 진단지표 / source 평균 취약)을 **거의 전부 수용**했고, 그중 **(a) 진단지표를 실제로 측정**해 가설을 정교화했고 **(b) SGI 코드를 source-wise 학습가중치로 수정**했다. 학습(V0~V3) 직전 상태에서 2차 검토 요청.

---

## 1. 1차 검토 핵심 지적 (요약)

1. 프레이밍: "모듈을 만들었다"가 아니라 "P2의 recall을 AP로 전환하는 병목을 공략"으로.
2. 노이즈 바닥: InnerCIoU≡CIoU 1회 차이로는 통계적으로 약함 → 핵심 4개만 3-seed.
3. "loss로 APtiny 못 푼다"는 과잉일반화 → "our setting, beyond variance"로 한정.
4. "recall↑/AP→ = semantic gap"은 아직 가설 → 진단지표(AP50/AP75, calibration, matched-IoU) 추가.
5. SGI `F_sem=mean` 취약 → source-wise learnable weight. loss는 보조 기여로 강등. 단계적 실험 + 복합 성공기준.

---

## 2. 우리가 반영한 것 (무엇을·어떻게)

### 2-1. 진단지표 측정 → 가설 정교화 (지적 4) ★가장 중요
저장된 baseline·naive-P2 예측 재분석. tiny(8-16px) "찾은 비율 / 찾은 것 평균 IoU":

| 모델 | 찾은 비율(@IoU.5) | 찾은 것 평균 IoU | IoU≥0.75 비율 |
|---|---|---|---|
| baseline | 48.9% | 0.655 | 20% |
| naive-P2 | 54.8% (+5.9%p) | 0.662 (≈동일) | 22% |

추가로 tiny AP50/AP75 분해:
| | tiny AP50 | tiny AP75 | AP75/AP50 |
|---|---|---|---|
| baseline | 0.147 | 0.033 | 22% |
| naive-P2 | 0.161 | 0.036 | 22% |

**해석(정교화):** "semantic gap 하나"가 아니라 **두 겹**:
- **검출(AP50/recall)**: tiny ~45% 미검출 → 의미 부족 = **SGI 공략 영역.**
- **정밀도(AP75)**: 찾은 것도 평균 IoU 0.65 갇힘, P2도 못 올림 → **기하학적 한계**(8-16px는 1px 오차로 IoU 급변). semantic·loss로 뚫기 어려움.

**부수 효과:** loss 8종이 APtiny 못 올린 게 "필연"으로 설명됨(AP75 기하한계). → "loss로 안 됨"이 미스터리가 아니라 일관된 결론.

### 2-2. SGI 코드 수정 (지적 5)
`F_sem = mean_i P_i(U(x_i))` → `F_sem = Σ softmax(src_w)_i · P_i(U(x_i))`.
- `src_w` init 0 → softmax 균등 → **init시 mean과 수치 동일**(검증 allclose). 학습으로 해로운 소스(8×업샘플 P5 등) down-weight 가능.
- 비용 0(스칼라 n개). 학습된 가중치는 "tiny에 어느 깊이가 유리한가" 해석 결과로 논문 활용 예정.
- 검증: 수식==코드 allclose=True, γ=0→out==base(안전초기화), 게이트∈[0,1], 4변형 빌드+E2E loss+backward 정상, GFLOPs 28.0~28.2(naive-P2 27.8 대비 +0.8~1.5%).

### 2-3. 성공 기준 정교화 (지적 1,4,5)
APtiny 단일 임계(+0.007) → **복합 기준**으로:
1. tiny AP50 / recall 상승(핵심), 2. tiny AP 평균 상승, 3. AR 유지/상승, 4. 전체 AP 무손상, 5. 비용 작음. (AP75/AP50은 보너스, 기하한계라 안 올라도 실패 아님.)
프레이밍도 "P2는 spatial opportunity, SGI는 recall→reliable detection 전환"으로 채택.

### 2-4. 문서 톤다운 (지적 1,3) — 반영 완료
- "loss로 SOD 못 푼다" → "Under our controlled YOLO26s/VisDrone setting, IoU-family losses did not produce a robust APtiny gain beyond observed training variance."
- "semantic gap이다(단정)" → "검출(의미)+정밀도(기하) 두 겹, 그중 검출만 actionable" (가설/진단으로 표현).
- 노이즈 바닥: InnerCIoU≡CIoU는 **정성적 sanity check로 격하**, 정량 바운드는 multi-seed로 대체 예정.

### 2-5. 실험 사다리 채택 (지적 5)
1. baseline·naive-P2·SGI V0~V3 (seed=0)
2. best SGI 선택 → 3. baseline·naive-P2·best ×3 seed → 4. best+WIoU/SIoU 1개 → 5. DOTA/AI-TOD는 best와 best+loss만.
loss는 메인 기여에서 **보조(진단 + 구조 위 AP75 분업 가능성)**로 강등.

---

## 3. 미반영 / 이견 / 보류

- **노이즈 바닥 multi-seed**: 코드·문서엔 반영했으나 **아직 미실행**(학습 사다리 3단계에서 baseline·naive-P2·best SGI만 3-seed 예정). 전체 8 loss 재학습은 비용상 안 함 — loss 결과는 보조 주장으로 격하했으므로 충분하다고 판단.
- **loss를 완전히 뺄지**: 코덱스는 "보조로"라 했고 우리도 동의하나, **추가 단서**가 있음 — 2-1 진단에서 tiny AP75가 기하한계로 보이므로, 만약 SGI가 AP50만 올리고 AP75는 못 올리면 **loss/detail-보존이 "AP75 전담"이라는 측정 가능한 분업**으로 재배치 가능. 즉 loss를 "막연한 refine"이 아니라 "AP75 타깃 보조 기여"로 둘 여지. (이게 과한 욕심인지 검토 요청 — §5.)
- **detail 보존(CARAFE 등)**: 아직 미구현. SGI(검출) 결과를 본 뒤, AP75가 병목이면 도입 검토.

---

## 4. 학습 직전 상태 (사실관계)

- 코드: `2-fpn` 브랜치. `ultralytics/nn/modules/block.py: SGI` (source-weight 반영). config `yolo26-p2-sgiV0~V3.yaml`.
  - V0=[19](self) / V1=[19,16](P3) / V2=[19,16,13](P3+P4) / V3=[19,16,13,10](P3+P4+P5).
- 기준선: naive-P2 (`yolo26-p2.yaml`, stock, baseline CIoU) — RUN_260625_01.
  - val mAP50 0.421(baseline 0.396), pycocotools AP@.5:.95 0.1841(baseline 0.1646, +0.0195).
- 학습 예정: V0~V3, muSGD/300ep/patience50/batch32/imgsz640/seed0, 8 GPU 2개씩 병렬.
- 평가: ultralytics val + pycocotools 구간별 AP + tiny AP50/AP75 + matched-IoU + 학습된 src_w.

### SGI 설계 요약 (수식)
```
base  = P0(x0);  w = softmax(src_w)
F_sem = base                       (V0) | Σ_i w_i·Pi(U(xi))  (V1~V3)
a_c   = σ(W2·ReLU(W1·GAP(F_sem)))           # 채널 attention
a_s   = σ(Conv7x7([mean_c;max_c] of F_sem)) # 공간 attention
out   = base + γ·( a_s ⊙ (a_c ⊙ F_sem) )    # γ init 0 = 항등(안전)
```
차별점: 깊은층(의미)이 P2(고해상)에 **채널·위치별로 주입을 guide** + residual로 detail 보존 + source 깊이를 학습 선택. 대조군 V0가 "attention만 vs cross-level 주입" 분리.

---

## 5. 2차 검토 요청 (이번에 묻고 싶은 것)

1. **진단 해석이 타당한가?** "tiny AP75 = 기하한계(matched IoU 0.65 갇힘)"라는 결론이 데이터로 정당한가, 아니면 다른 통제(annotation noise, P2 assignment, conf calibration)를 더 봐야 하나? matched-IoU 0.65를 "기하한계"로 부르는 게 과한가?
2. **SGI 성공 기준을 AP50/recall 중심으로 옮긴 게 합리적인가?** APtiny@[.5:.95]를 헤드라인에서 내리는 게 리뷰어에게 "metric cherry-picking"으로 보일 위험은?
3. **§3의 "loss를 AP75 전담 보조 기여로 재배치" 아이디어** — 측정 가능한 분업이라 매력적이나, 데이터가 받쳐주지 않으면 억지. 진행할 가치가 있나, 아니면 loss는 순수 진단(motivation)으로만 둘까?
4. **source-wise softmax weight**가 충분한가, 아니면 (코덱스가 언급한) size-aware gate까지 1차에 넣어야 하나? (우리는 리스크 줄이려 scalar만 넣음.)
5. **V0 대조군 설계**가 "attention vs cross-level injection" 분리에 충분한가? 더 필요한 대조군(예: P2에 deep을 단순 concat만, attention 없이)이 있나?
6. **학습 시작해도 되는 상태인가?** 시작 전 반드시 고쳐야 할 치명적 결함이 남아있나?

---

## 6. 파일 맵
| 파일 | 내용 |
|---|---|
| `../06_handoff/SOD_HANDOFF_260626.md` | 1차 검토 요청(전체 상황) |
| `../06_handoff/SOD_HANDOFF_260626_2.md` | 이 문서(1차 반영 보고 + 2차 요청) |
| `../03_modules/SOD_MODULE_SGI.md` | SGI 설계·수식(source-weight 반영)·창작성·검증·성공기준 |
| `../04_results/SOD_RESULT_AP_SIZE.md` | pycocotools 구간별 AP + tiny matched-IoU 진단 |
| `../00_roadmap/SOD_ROADMAP_PAPER_PROCESS.md` | 전체 흐름 + 핵심 스토리 + 서술 전략 |
| `../02_loss/SOD_LOSS_EXPERIMENTS.md` / `../03_modules/SOD_MODULE_FPN_SURVEY.md` | loss 8종 / FPN 변형 비교 |
| 코드 | `2-fpn`: `block.py:SGI`, `cfg/models/26/yolo26-p2-sgiV0~V3.yaml` |
