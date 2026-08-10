# Head Localization Refinement — 위치 정밀화 모듈 (Phase 4 설계)

작성 2026-07-03. `../03_modules/SOD_MODULE_SGI.md`·`../03_modules/SOD_MODULE_GSA.md`(neck fusion, 둘 다 negative)에 이은 **head 트랙** 설계 문서.
SGI/GSA와 **손대는 위치가 다르다**: neck(semantic gap)이 아니라 **head(regression, 위치 예측)**.

## 왜 head인가 (근거)

- **loss는 닫힘**: loss 8종 전부 APtiny 노이즈 초과 0개(`../02_loss/SOD_LOSS_EXPERIMENTS.md`). "재가중은 gradient 예산 재분배일 뿐, 맞출 feature 없으면 정밀 회귀 무의미."
- **neck injection도 닫힘**: SGI/GSA 전부 무효(`../04_results/SOD_RESULT_AP_SIZE.md`). P2는 top-down PAN으로 의미 이미 보유.
- **위치는 미개척 + 천장 높음**: oracle 상한(`oracle_ceiling.py`) — 위치 완벽 시 **APtiny 0.0564 → 0.2083(노이즈 ~20배)**. tiny 실패의 정밀도 축(matched IoU 0.65, AP75 바닥)은 아무도 안 건드림.

## 모듈 개념 (그림)

tiny 는 레이어 19 = **P2/4**(stride 4, 고해상)에서 `Detect.cv2`(DFL 회귀)로 예측된다.

```
[1] 기존 head가 대략 박스 예측 (coarse box) — 물체를 살짝 빗나감 (IoU 0.65)
[2] coarse box의 네 변 위치에서 P2 고해상 feature 를 grid_sample 로 콕 집어 샘플
[3] 경량 conv/MLP 가 각 변의 미세 offset Δ=(Δl,Δr,Δt,Δb) 예측
[4] refined box = coarse box + γ·Δ      (γ=0 초기화 → 처음엔 identity, 안전)
[5] refined box 에 IoU loss (기존 회귀 위에 refine 단계 추가)
```

한 줄: **"대충 그린 박스를 고해상 특징으로 네 변을 밀어 딱 맞추는 경량 보정기"** 를 head(P2 스케일)에 추가.

## 가설 명세 (pre-registration, 학습 전 고정)

| 항목 | 내용 |
|---|---|
| 메커니즘 | coarse 박스 4변을 P2 고해상 feature 로 샘플해 sub-pixel offset 보정 → matched IoU↑ |
| 표적 지표 | **APtiny(8–16), AP75** (전체 AP 아님) |
| 예상 크기 | 노이즈(±0.007)의 최소 3배 이상 (oracle 천장 0.208 근거) |
| 대조군 | **naive-P2** (baseline 아님 — refinement 순수 기여만) |
| 자기 대조군 | **γ=0 고정**(refinement 끔) = P2와 동일해야. γ가 유의하게 켜져야 유효 |
| 반증 조건 | **P2 대비 APtiny Δ < 0.007 이면 폐기** |

## 구현 계획

1. 브랜치 `3-head_refine` (loss=1.x, fpn=2.x, **head=3.x** 관례 확장).
2. `block.py` 에 경량 `P2Refine`(grid_sample + conv 2~3 + γ residual). 등록은 `develop-sod-module` 참고.
3. `head.py` 에서 **P2 스케일 출력에만** 적용(P3/P4/P5 불변).
4. **가장 단순한 버전부터**: 박스 중심 1점 offset → 되면 4변으로 확장(GSA γ=−0.91 반복 방지용 단계적 확장 + γ residual).

## 검증 흐름 (기존 스킬 재사용)

```
develop-sod-module(verify_module.py: 빌드·등록·forward)
  → verify-sod-math(check_correctness.py: offset/box 디코딩 수식이 의도대로인가)
  → run-sod-paper(smoke.sh: 학습 + size-AP 평가)
  → sod-research-method(analyze_result.py: P2 대비 APtiny 유의성 → 반증조건 판정)
```

## 정직한 리스크

- oracle 천장(0.208)이 높다고 **실제가 도달하는 건 아님**. P2 대비 Δ<0.007 이면 깔끔히 폐기.
- 폐기 시 선회처: **det(recall) 방향**(oracle det상한 0.775 로 더 큼). 단 검출은 P2가 이미 일부 공략.
- γ residual + "1점→4변" 단계적 확장 = GSA 실패(γ 음수, 모델이 모듈을 끔) 반복 방지 안전장치.

## 구현 상태 — v0 초안 (2026-07-03, 브랜치 `3-head_refine`)

- **`nn/modules/block.py:P2Refine`** 추가(`__all__` 등록). v0=중심이동(dcx,dcy) / v1=4변(dl,dt,dr,db).
  grid_sample(중심 콕 집기) → MLP → `tanh*max_shift` bounded offset → `refined = boxes + γ·Δ`.
- **단위 검증 통과**(중립 cwd, 더미 P2 feature+coarse box):
  - identity: γ=0 → refined==coarse ✓ (안전 시작)
  - γ.grad=+3.36 ≠ 0 → **identity 시작해도 γ 학습 가능** ✓
  - offset bounded(≤max_shift=4px), v0 2,147 params(경량) ✓
- **head 배선 완료 (v0, 추론 경로)**: `head.py:DetectP2Refine(Detect)` — `_get_decode_boxes`
  오버라이드로 P2(index 0) 디코딩 박스를 `P2Refine` 으로 보정. 등록: `nn/modules/__init__.py`,
  `nn/tasks.py`(import + parse_model frozenset + legacy). yaml `cfg/models/26/yolo26-p2-refine.yaml`
  (`Detect`→`DetectP2Refine`).
  - 검증: `verify_module.py --model yolo26-p2-refine.yaml` → **빌드 OK(params 2,663,523), forward OK.**
  - refine 실동작 확인: γ=5 에서 P2 앵커 **25,600개 박스 실제 보정**(평균 1.99px, 최대 2.65px).
  - 참고: yolo26 은 **end2end(NMS-free)** — 무의미 입력(zeros)에선 postprocess topk 가 refine된
    P2 앵커를 최종 선택에 안 넣어 최종 출력이 안 변할 수 있음(배선 문제 아님, 입력 아티팩트).
## loss.py 학습 연동 — 완료 (2026-07-06)

추론 경로(head.py)에 더해 **학습 경로(loss.py)**에도 refine 연동. 이제 `p2_refine` 이 학습됨.

- **연동 지점** `utils/loss.py:v8DetectionLoss`:
  - `__init__`: `self.p2_refine = getattr(m, "p2_refine", None)` (baseline 은 None → 무시).
  - `get_assigned_targets_and_loss`: assigner(coarse) 뒤·bbox_loss 앞에서 `pred_bboxes` 의 P2
    부분(`[:, :H·W]`)을 `p2_refine` 으로 보정. **pred_bboxes 는 P2 feat 좌표라 좌표변환 불필요.**
- **설계**: **assign 은 coarse, DFL 은 그대로**, IoU loss 만 refined → target 안정 + gradient 연결.
  end2end 는 E2ELoss 가 one2many(학습 주력)에 자동 적용.
- **⚠ 발견·수정 (gamma 초기값)**: gamma=0 으로 시작하면 `∂refined/∂proj = gamma·(…) = 0` 이라
  **offset 예측기(proj)가 안 배우는 닭-달걀**(gamma 가 랜덤 offset 으로 안 켜지면 영영). →
  **gamma 초기값 0 → 0.01(작은 양수)**. 거의 identity(≤0.04px)면서 proj 도 1-step 부터 학습.
- **검증(1-step forward+backward, 작은 박스 40개로 P2 fg 유도)**:
  - fg_mask P2 앵커 321개 → `gamma.grad=−0.040`, `proj.weight |grad|=0.0044` **둘 다 ≠0 ✓**
  - gamma=0.01 최대 박스변화 0.0099px(거의 identity) ✓  · baseline(yolo26-p2) p2_refine=None ✓
- **다음 = Phase 4 마지막**: `run-sod-paper` 로 실제 학습(naive-P2 와 동일 조건) → 학습 후 γ 부호
  감시(음수면 GSA류 경보) → `analyze_result` 로 **naive-P2 대비 APtiny** 판정(Δ<0.007 이면 폐기).

## 학습 진행 + 엔지니어링 로그 (2026-07-06) — ⚠️ 논문 제외/포함 구분

학습 세팅: VisDrone, batch=32, 300ep, patience=50, imgsz=640, seed=0, device=2,3,4, MuSGD(auto).
**naive-P2(260625)와 동일 조건** (대조군 공정 비교). run=`260706_Y26S_P2REFINE_VISDRONE`.

**첫 epoch γ: 0.01 → 0.0245 (양수 증가)** — 모델이 refine 을 쓰기 시작(GSA γ=-0.91 과 정반대). 좋은 초기 신호.

**겪은 엔지니어링(= 논문 제외, 순수 구현):**
- AMP dtype(Half/Float) 충돌 → `P2Refine.forward` 를 `boxes.to(feat.dtype)` 로 feat dtype 에 순응(수정).
  교훈: 검증은 **실제 학습 조건(DDP+AMP autocast)** 을 재현해야 잡힌다.
- refine **fg-only 최적화**(전체 25600 앵커→fg 앵커만): non-fg 는 bbox_loss 가 안 쓰므로. 단
  `TaskAlignedAssigner` CUDA OOM→CPU fallback 은 batch32+P2 앵커 자체 부하라 잔존(속도 ~569s/ep,
  대조군 100s). → device 3장(2,3,4)으로 per-GPU batch↓ 재시도.

**논문 반영 원칙 (이 프로젝트 공통):**
- **제외**: dtype 버그, assigner OOM/CPU fallback, 학습 속도·메모리 최적화 삽질 (엔지니어링 디테일).
- **포함(재현성, 부록/코드)**: 학습 세팅(batch32/300ep/seed0), 모듈 hp(γ0.01, max_shift 4px, fg-refine).
- **포함(방법론, 본문)**: oracle 상한 근거, 대조군=naive-P2, γ 감시, 노이즈(±0.007) 판정.
- **결과**: APtiny 노이즈 초과 → 기여(Phase 4 레버①) / 무효 → negative ablation(SGI·GSA 계열).

## 실험 결과 — negative (2026-07-07)

naive-P2 대비: **APtiny −0.0026(무효)**, AR −0.0098(악화), vt(2-8) +0.0072(개선, near-hopeless).
**폐기**(반증조건 P2 대비 APtiny Δ<0.007 충족). 최종 γ=+0.060(살아있음). 수치표: `../04_results/SOD_RESULT_AP_SIZE.md`.

### 왜 실패했나 (분석)
- **oracle–실제 갭**: oracle 상한 0.208 은 "GT 위치를 안다면"의 정답 스냅. 실제 모듈은 GT 를 모르고
  P2 feature 로 offset 을 추론한다. **tiny feature 가 빈약(1~2px)해 "어디로 밀지" 단서가 없음** →
  γ 는 켜졌어도(refine 사용) 유효한 offset 을 학습하지 못함.
- **AR 악화**: refine 이 박스를 옮기며 TP 가 어긋나고 end2end 순위를 교란 → recall 하락(순이득 아님).
- **vt 개선 vs tiny 무효**: vt(2-8)는 coarse 가 크게 빗나가 refine 이 조금 도움; tiny(8-16)는 이미
  coarse 가 어느정도 맞아(matched IoU 0.65) offset 이 오히려 방해.
- **근본**: 문제는 "회귀 방식"이 아니라 여전히 **"정보 부족"(원인 ①)**. refine 은 있던 정보를
  재배치할 뿐 없는 정보를 못 만든다 — tiny 엔 재배치할 위치 단서 자체가 빈약.

### 다음 방향 (인사이트)
1. **det/recall** (oracle det 0.775 ≫ loc 0.208): AR 악화가 시사 — tiny 병목은 정밀위치보다 **"찾기"**.
2. **입력 해상도/tiling** (paper_process 레버②): tiny feature 빈약의 근본 해소 = **"정보를 만든다"**.
   imgsz↑(1024/1280) 또는 VisDrone tiling.
3. refine 재설계(경계 다점 deformable 샘플 / max_shift 축소 / assign 도 refined 로 일관)는
   feature 빈약을 안 풀면 한계.
- **결론**: "정보 재배치"(refine)로는 tiny 못 푼다. **"정보를 만드는"(해상도·검출) 방향**으로 전환.
