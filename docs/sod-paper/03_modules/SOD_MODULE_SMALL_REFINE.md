# Small-scale Position Refinement — small object 위치 정밀도 트랙 (설계·인수인계)

작성 2026-07-08 (Codex 1~4차 인수인계). tiny(8-16) 정보 한계 확정 후 **목표를 small(16-32)로 이동**.
**핵심 (Codex 4차 — oracle det 0.76≫loc 0.34 로 갱신): small 최대 여지는 위치보다 detection/ranking.
→ 메인 경로 = small-aware assignment/ranking(Phase 2), SA-WIoU 는 보조(Phase 1), refine 은 마지막(Phase 3).**
(문서 제목은 "refine"이 남아 있으나 실제 메인은 assignment/ranking — 파일명 정리는 후속.)

---

## 1. 동기 (왜 small, 왜 위치 정밀도)

- **tiny(8-16) 접음**: fusion 3방향(주입/억제/재배치)·loss 8종 전부 무효. 정보 물리 한계
  (`../04_results/SOD_RESULT_AP_SIZE.md` 구조 트랙 종료). oracle 도 "정보 완벽해야" 오름.
- **small(16-32) 진단** (test, naive-P2):
  - small AP50=0.2953 vs **AP75=0.1147** (비율 **39%**). 비교: medium AP50=0.4649/AP75=0.3025(**65%**)
    → **small 은 AP50→AP75 에서 훨씬 많이 무너짐 = localization quality 문제의 강한 신호.**
  - oracle 위치완벽 상한 0.3067 (현재 AP 0.1402 의 2.2배) → 위치 정밀도 개선 **여지**.
  - **small 은 P2 에서 feature 4~8px** (tiny 1~2px 와 결정적 차이) → 모듈이 배울 정보가 있음.
  - ⚠ **단정 금지 (Codex)**: 병목은 **localization 단독이 아니라 detection/ranking 과 공존**한다.
    oracle 은 GT 스냅한 **관대한 상한**(모듈이 배운다는 보장 X)이고 **det 상한 > loc 상한**이라
    recall/ranking 여지도 남는다. → oracle 을 **naive-P2 예측 기준**으로 재산출(§5 재분석).
  > 정리 문장 (Codex 최종): *small 병목은 localization 과 detection/ranking 이 공존하지만, **oracle 기준
  > 최대 여지는 detection/ranking 쪽이 더 크다**(det 0.76 ≫ loc 0.34). 따라서 **refine 은 후순위이고,
  > small-aware assignment/ranking 이 메인 경로**다.*
- **왜 기존 시도가 small 에 무효였나**:
  - P2Refine: **tiny 타겟**(feature 빈약)이라 실패, small 을 특화 안 함
  - **P2+WIoU: small AP75 0.1147→0.1149 무효.** WIoU v3 는 "어려운 박스 down-weight"라 **small(어려움)을
    홀대** → large(APl +0.0189)만 올림
  - SGI/GSA/억제: fusion 이라 위치와 무관
  - **→ 기존 시도는 small 의 feature fusion·loss·좌표 refine 을 각각 건드렸지만, small 후보의
    positive 선택과 quality-aware ranking 을 직접 정렬하는 모듈은 검증하지 못했다.**
    ⚠ **표현 정정 (2026-07-08)**: "small 위치 정밀도 특화 부재"는 부정확. oracle det 0.76≫loc 0.34 →
    병목은 위치보다 **검출/순위/품질정렬**. 빈 자리 = **quality/ranking 모듈화**.

## ★ 최우선 목표 재확인 (2026-07-08, Codex 5차 + 사용자 강조)

**연구 최우선 목표 = "새 모듈 개발".** loss 튜닝도, assignment 튜닝도 최종 기여가 아니다.
- `tal.py` CIoU/NWD blend 는 **모듈이 아님**(파라미터 0, inference 불변) = **병목이 정말 ranking/quality
  인지 확인하는 저비용 probe**. 위치를 낮춘다. 성공/실패 모두 모듈 설계의 근거로만 쓴다.
- **진짜 기여 = Small-Aware Quality module (가칭 SAQ)** — 상세 설계 `SOD_MODULE_SAQ.md` (v1 초안 작성됨).
  - 입력: P2/P3 feature + 예측 박스 + cls score + **small-scale prior**
  - 출력: small-aware **quality score** (localization quality 추정)
  - inference: `final_score = cls × quality^η` (small 후보) — **파라미터 있는 진짜 모듈, inference 개입**
  - 차별점: GFL/VFNet/IoU-aware head 는 전 스케일 동일 → 우리는 **small-scale 한정 quality gating(SOD 특화)**
  - 기존 실패와 무충돌: 좌표 안 밈(≠P2Refine)·fusion 강제 안 함(≠GSA/AFFG)·loss 재가중 아님(≠WIoU)
- 로드맵: Step0 naive-P2 유지 → Step1 (선택)tal probe → **Step2 ★SAQ 모듈 설계** → Step3 ablation(P2/+probe/+SAQ/full).

## 2. 실험 계획 (Codex 4차: **ranking/assignment 가 메인**, SA-WIoU 는 보조, refine 은 마지막)

새 oracle(small **det 0.76 ≫ loc 0.34**)이 방향을 정함 — 최대 여지는 위치보다 **detection/ranking**.
→ **메인 승부처 = small-aware assignment/ranking**. (자원 하나뿐이면 SA-WIoU 보다 이걸 먼저.)

### Phase 1 — P2 + SA-WIoU (loss, 빠른 보조 sanity — **관문 아님**)
- localization-loss route 확인용. **성공해도 메인 결론 아님, 실패해도 small track 실패 아님**
  (loss route 실패일 뿐 ranking route 는 살아있음).
- 코드: `1.3.1-loss_sawiou` + `yolo26s-p2.yaml`. 성공 기준: 필수 **mAP50 유지** · 핵심 **small AP75 +0.007** ·
  보조 **APs/AP50/AR 하나↑** · 금지 **AP75 만↑ + AP50/AR 붕괴**. (AP75 만 오르면 "정밀도 개선"이지
  "small **detection** 개선"은 아님)

### Phase 2 — small-aware assignment / ranking  ★ 메인
- 근거: oracle det≫loc + P2Refine 이 좌표 건드리자 AP50/AR 붕괴 = **어떤 anchor/point 를 positive 로
  뽑고 어떤 순위를 주는지**가 핵심.

- **첫 실험 (Codex 5차 정밀화): overlaps 통째 교체 금지 → `assign_overlaps`만 분리해 align_metric 에만 blend.**
  현재 `tal.py` 의 `overlaps`(=CIoU, `iou_calculation` line 213)는 **세 군데**서 쓰임:
  ① `align_metric = cls^α × overlaps^β`(ranking) · ② `select_highest_overlaps`(multi-GT 충돌해결)
  · ③ `pos_overlaps` norm(target score 스케일, line 138). **통째 바꾸면 셋이 한꺼번에 변해 해석 불가.**
  → **첫 실험은 ①만 blend, ②·③ 은 CIoU 유지** (= "ranking 만" 순수 검증):
  ```python
  # get_box_metrics 안, align_metric 계산 직전
  # overlaps(=CIoU) 는 그대로 반환 → 충돌해결·norm 은 자동으로 CIoU 유지
  small_gt = size_gate(gt_bboxes)                       # (b, n_gt, 1), sqrt(area)∈[16,32)
  nwd      = wasserstein_nwd(gt_boxes, pd_boxes)        # 1.1-loss_nwd 에서 포팅
  assign_overlaps = torch.where(small_gt,
                                (1-λ)*overlaps + λ*nwd,  # small 만 blend
                                overlaps)                # 나머지 CIoU
  align_metric = bbox_scores.pow(α) * assign_overlaps.pow(β)   # ← ①만 교체
  ```
  - base 는 plain IoU 아님 = **기존과 동일한 clamped CIoU**(line 213 확인). blend = `(1-λ)·CIoU + λ·NWD`.
  - **size gate = sqrt(area) 기준 [16,32)** (w,h 각각 16-32 로 하면 길쭉한 small 누락 → AP/oracle size bin 과 어긋남):
    `wh=(gt[...,2:]-gt[...,:2]).clamp(min=0); gt_size=wh.prod(-1).sqrt()`, `(gt_size>=16)&(gt_size<32)&mask_gt` → `(b,n_gt,1)` broadcast.
  - **λ: 0.25 fixed 로 첫 probe**(β=6 라 overlap 차이 크게 증폭 → assignment churn 위험). 0.5 는 2차 run 에서. (warmup 0→0.5 는 epoch 접근 번거로우면 skip.)
  - **topk 고정 · candidate selection(STAL) 고정 · inference 불변.**

- **확장 순서 (첫 실험 성공 후에만):** ② norm target score 도 blend(quality label 재정의 검토) · ③ small-GT topk 소폭↑ · ④ quality/IoU score branch·calibration.
- **STAL 중복 걱정 X (Codex):** STAL=후보 공간 확대(주로 극소 GT), NWD blend=그 후보 안에서 순위 재산정. **small(16-32) 한정·tiny 제외**라 중복 작음. (tiny 까지 열면 STAL+NWD 로 positive 과잉관대 위험 → 열지 말 것.)
- **필수 로그 (AP 만 보면 이유 모름):** small GT당 positive 수 · **unmatched small GT 비율** · assigned positive 의 **raw CIoU 평균/상위** · assigned positive 의 **NWD 평균** · small target score 분포 · **P2/P3별 small positive 분포** · small AP50/AP75/AR.
- **실패 해석 분기 (Codex):**
  - positive↑ + AP50/AR↑ + **AP75↓** → NWD 과관대. **λ 낮추거나 CIoU floor 추가.**
  - **positive 변화 없음** → candidate/topk 병목. 다음 = small-GT topk 소폭↑.
  - positive↑ 인데 **AP 전부↓** → false positive/ranking 오염. **NWD gate 축소·P2-only 제한·λ↓.**
  - matched quality↑ 인데 **AP 정체** → score calibration 문제. 다음 = quality score branch.

### Phase 3 — small boundary refine (마지막·조건부, P2Refine 재탕 금지)
- **Phase 2 에서 ranking/positive selection 이 개선된 뒤에만.** P2Refine 은 small AP75/AP50/AR 다 내림(§5)
  → **단순 offset 금지.** 굳이 한다면 assigned-GT-small 만·tiny 제외·γ=0.01·**좌표 이동보다 score-ranking
  동시 학습이 핵심**(위치만 고치고 score 그대로면 PR ranking 망가짐).
- 구현 교훈: AMP dtype 순응(`boxes.to(feat.dtype)`), DDP+AMP autocast 검증. `SOD_MODULE_HEAD_REFINE.md`.

## 3. 판정 (SOD-YOLO 형식, `sod-research-method`)

대조군 = **naive-P2(P2v2)**. 노이즈 바닥 band ±0.007, AP50 ±0.010.
- **헤드라인**: test **mAP50, mAP50-95**
- **기여 근거**: **COCO APs(<32)/APm** + **small(16-32) AP75**(위치 정밀도 직접 지표)
- **성공 조건 (전체 트랙 헤드라인)**: mAP50 ↑ **AND** small AP75 ↑(노이즈 초과) **AND** mAP50-95 안 떨어짐
- **Phase 2(assignment) 1차 최소 성공 (Codex — AP75 단독 아님)**: small **AP50↑ AND AR↑ AND APs↑** ·
  전체 mAP50-95 유지/미세↑ · **small AP75 크게 안 무너짐**. (ranking 실험이라 recall/AP50 이 먼저 움직임)
- eval: `eval_size_ap.py`(by-size AP50/75 확장됨) → `analyze_result.py --base apsize_P2v2 --exp ...`
- **multi-seed 필수**(단일 +0.006 은 노이즈일 수 있음 — P2+WIoU 교훈)

## 4. 학습 세팅 / 구현 지점

- 세팅 = naive-P2 동일: batch32, 300ep, patience50, imgsz640, seed0, MuSGD(auto). **device 3장(assigner OOM 회피).**
- Phase 1(SA-WIoU, 보조): 코드 변경 없음(브랜치 loss + p2 yaml).
- **Phase 2(assignment, 메인)**: `tal.py` `get_box_metrics` 에서 **`assign_overlaps` 분리** → align_metric(line 200)에만
  small-GT gate CIoU/NWD blend(λ=0.25 first, sqrt-area gate, topk 고정). **반환 `overlaps`(CIoU) 불변** → 충돌해결·norm
  자동 유지. NWD 는 `1.1-loss_nwd` 브랜치 `metrics.py:wasserstein_nwd` 포팅. inference 구조 불변.
- Phase 3(refine, 마지막): `nn/modules/block.py`+`head.py`(assigned-small, score-ranking 동시). Phase 2 성공 후.

## 5. 선결 재분석 (Codex — 데이터 재산출)

- **naive-P2 기준 oracle (완료 2026-07-08, small)**: loc 상한 **0.3360**(현재 0.1402→위치완벽, Δ+0.196),
  det 상한 **0.7619**. **det ≫ loc (0.76 vs 0.34)** → **recall/ranking 여지가 위치 여지보다 훨씬 큼.**
  → Codex "ranking/quality 를 refine 보다 먼저(Phase 2)" 근거 **데이터로 확증**. (전체 all: loc 0.365 / det 0.790 동일 패턴)
- **P2Refine small (재산출 완료 2026-07-08)**: AP75 **0.1147→0.1111(−0.0036 하락)**, AP50 0.2953→0.2868
  (−0.0085), AR 0.3221→0.3133(−0.0088). **P2Refine 은 small AP75 조차 못 올리고 전부 내렸다.**
  → **⚠ 단순 offset 재탕 절대 금지.** Phase 2 는 설계 변경 필수(경계 다점 deformable / quality(IoU)-aware
  refine / assigned-small 한정 등). "small 은 feature 있으니 offset 먹힐 것"은 P2Refine 방식으론 반증됨.

## 6. Codex 인수인계 체크리스트

- [ ] **Phase 1 먼저**: `1.3.1-loss_sawiou` SA-WIoU 활성 확인 → `yolo26s-p2.yaml` 학습 → naive-P2 대비 판정.
      **§2 3단계 관문 적용**(체크리스트도 3단계로): small 학습 신호(AP75/AP50/AR/APs) **전무면 보류**,
      **부분 신호면 축소 probe**(AP75 안 올라도 AP50/AR/APs 중 하나↑면 신호 있음 → 무조건 보류 아님).
- [ ] 대조군 = naive-P2(260625/P2v2), 반증 = small AP75 Δ<0.007 또는 mAP50 하락.
- [ ] multi-seed(≥3) 로 확정(단일 seed 노이즈 주의).
- [ ] **Phase 2(메인) = tal.py `assign_overlaps` 분리, align_metric 에만 small-GT gate CIoU/NWD blend
      (λ=0.25 first·sqrt-area gate·topk 고정, 반환 overlaps=CIoU 불변)**. NWD 포팅=`1.1-loss_nwd` metrics.py.
      로그: positive 수·unmatched small 비율·raw CIoU·NWD 평균·target score 분포·P2/P3 분포·AP50/75/AR.
      (자원 하나면 SA-WIoU 보다 이걸 먼저)
- [ ] Phase 3 refine 은 Phase 2 성공 후에만. P2Refine 교훈(AMP dtype·identity init·score-ranking 동시·autocast 검증).
- [ ] negative 여도 정직 기록 → `../04_results/SOD_RESULT_AP_SIZE.md`. 논문엔 방법론만.
- [ ] 지표 관행: SOD-YOLO 는 mAP50/mAP50-95 + COCO APs/APm 표준(AI-TOD tiny 세분은 내부 진단만).

## 7. 참고

- small 진단·oracle: `../04_results/SOD_RESULT_AP_SIZE.md`(P2+WIoU, oracle 상한).
- 교훈: `SOD_MODULE_HEAD_REFINE.md`(P2Refine dtype·닭달걀), `SOD_MODULE_AFFG.md`(identity init·판정 분해).
- 방법론: `/home/hsjeong/workspace/SOD-PAPER/.claude/skills/sod-research-method/`.
