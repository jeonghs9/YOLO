# SAQ A1b — 코덱스 대결 브리프 (자립형, 콜드 리더용)

작성 2026-07-13. 이 문서 하나로 A1b 를 논쟁 가능. 상세 설계=`SOD_MODULE_SAQ.md`, 실측=`../04_results/SOD_RESULT_AP_SIZE.md`.
**규칙**: 반론은 근거로, 최종 판정은 실측으로 종결. 무비판 수용·무근거 반대 둘 다 금지.

---

## ★ 라운드2 (2026-07-14) — A1b NEGATIVE + 원인 + R1 제안  ← **이번 대결 주제**

**결과(test @maxDet1500, 재학습0 3종):**
- base 체크(γ0, rescoring OFF): small AP 0.1404 ≈ naive-P2 0.1402 → **L_q 가 검출망 안 망침(base 무해)**.
- as-deployed(γ1): small AP 0.1391 (−0.001, 이득 0).
- **결합식 probe(one2one)**: 학습 q 는 cls·q^{.5,1,2}·(1+q)·small-only **전부 ≤ 기준**(best −0.0012). 반면
  **ORACLE cls·IoU² = +0.0461**(여지 실재, one2one probe 라 oracle 정상작동 = 경로 유효).

**원인(smoking gun):** A1b one2one q 의 Sp(q,IoU) small = **−0.335** (A1a-1 one2many 의 **+0.504** 대비 **부호 반전**).
→ **q 가 IoU 와 반대로 학습됨.** one2one 은 NMS-free(객체당 1 살리고 억제) → 억제 anchor 의 reg feature 가 IoU 정보
결핍 → q head 오학습. **딜레마: q 를 배울 수 있는 곳(one2many)은 추론에 안 쓰이고, 추론에 쓰는 곳(one2one)엔 안 배워짐.**
(= 최초 7문 자기검증 '발견1' 현실화.)

**판정: A1b = NEGATIVE.** 단 개념 여지(oracle +0.046, loss/SPD 보다 큼)는 실재 → 실패는 개념 아니라 **q 학습 경로**.

**R1 제안(rescue 후보):** q 를 one2many·one2one **공유 입력 feature** 에서 뽑아 **one2many 조밀신호로 L_q 학습**(learnable 확보)
→ **one2one 추론에 그 q 사용**(usable 확보). 딜레마 정면 해소. ⚠ 리스크: ① 공유(pre-reg) feature 가 box-accuracy 정보
덜 담아 q 품질↓(§9-J) ② q-cls 중복 0.58~0.78 이 상한 → 회수 부분적(기대 +0.02 내외).

**코덱스에게 (라운드2 질문):** ① 원인 타당성 ② R1 이 딜레마 푸나 ③ ROI ④ 대안 rescue.

**→ 라운드2 합의 (Codex 응답 2026-07-14, 대부분 수용):**
1. **원인 = 유력 가설로 강등**: 확정된 것은 "one2one q 가 IoU 역정렬"까지. "억제 anchor 정보결핍"은 미분리(assigner 저score화·
   VFL weight·top-anchor 학습 등 대안 원인 존재). → 문서 확정표현 삭제.
2. **R1 은 딜레마 완전해소 아님**: feature-mismatch → **box/target-mismatch 로 이동**할 뿐(target=one2many box IoU, 사용=one2one box).
   → **full train 전 smoke-gate 필수**: shared/distill q 가 one2one box IoU 와 small Spearman **≥ +0.2~0.3** 나오나 선확인.
3. **oracle +0.046 은 probe-내부 상대값**(공식 AP 0.1404 ≠ probe 0.2591) → 공식 headroom 으로 인용 금지.
4. **R1 ROI 낮음**: 공식 배포효과 0(γ0 0.1404→γ1 0.1391), combo 학습 q 전부 음수, q-cls 중복 상한 → 기대 +0.02(노이즈·기여성 애매).
5. **표현 정정**: "실패=개념 아니라 경로" → **"ranking 여지는 남으나 현재 SAQ learnable q 로 회수 못함"**.
6. **더 나은 rescue(Codex)**: R1 < **distill**(one2many q→one2one q) or **top-candidate 중심 target/weight**.

**합의 판정**: A1b negative 확정. rescue 는 **값싼 smoke-gate(one2one IoU Spearman) 통과 시에만** 착수, 아니면 SAQ 버림.

**→ 라운드3 결과 (2026-07-15, 재학습0 실측 종결):**
- **gate PASS**: one2many q → one2one 박스 IoU **Sp small +0.686**(cls −0.05). **Codex #2(box mismatch) 반증** — q transfer 잘 됨.
  A1b 실패는 mismatch 아니라 "one2one 학습" 국소.
- **BUT AP 실측**: one2many q 로 one2one rescore → small AP **+0.0022(노이즈)**, oracle +0.048 의 ~5%. → **Codex #4(ROI 낮음) 최종 확인.**
- **핵심 교훈**: **랭킹 상관(0.69) ≠ AP.** one2one 은 이미 top 정렬 → 재정렬 무의미, 완벽 q 만 이득.
- **최종: SAQ ranking 축 폐기.** oracle 여지(+0.048) 실재하나 learnable q 로 회수 불가 = 정직한 negative. 다음 = 전략 재검토(양축 벽).

---

## 0. 맥락 (한 문단)

YOLO26s + P2 head(naive-P2) VisDrone. **naive-P2 = 확정 양성**(baseline 대비 all AP +0.0195). 이후 **recall 축 전멸**:
SGI·GSA·P2Refine·AFFG(fusion/refine), NWD-assign-probe, **SPD-Conv backbone·full**(feature 보존) — 전부 small AR 0.32 불변 = negative.
→ recall 축 ROI 소진. **precision/ranking 축(SAQ)** 으로 피벗. 최우선 목표 = **새 모듈**(loss/assignment 튜닝은 기여 아님).

## 1. SAQ A1b — 무엇인가 (**구현·smoke 완료 / 효과 미검증**)

> ⚠ **현재 상태 = 구현 PASS + 본학습·eval 대기**. "효과 입증"은 아직 아님. `apsize_*A1b*_test.json` 미존재.
> 코덱스 판정(2026-07-13) 수용: "구현은 합격 근접, 효과 성공은 미검증" — 동의.

> 각 예측 anchor 의 **위치품질 q∈[0,1]** 을 경량 head(cv_q, 2conv+sigmoid)로 예측하고,
> **추론 순위를 `score = cls_prob · q^γ` (γ=1)** 로 재점수화한다. NMS-free(one2one) head 의 quality-aware 확장.

- **q 위치 = one2one(추론 경로) reg-branch mid feature**. 학습(L_q)·추론(rescoring) **동일 경로 = 대칭**.
- **L_q, target = IoU-target-all**: `t_i = max_j plain_IoU(box_i.detach(), GT_j)` (전 anchor, class-agnostic).
- **loss weight = 표준 Varifocal** (positive: t / 그 외: α·σ(q)^γ) + **Σw 정규화**, w_q=0.1.
  ⚠ **정직(Codex 지적)**: 설계문서엔 "unassigned-overlap=1(uniform)"라 적혔지만 **코드는 VFL 음성가중**(bg 압도 방지).
  즉 "target 은 all-anchor IoU(uniform/all)"는 맞고, **"weight 가 uniform"은 과장** — weight 는 VFL.
- **λ_small(크기 가중) = A2 로 분리**(지금 크기 uniform). NWD = A3 보류. → A1b 는 **순수 IoU-quality rescoring** 만.

## 2. 왜 되리라 보나 (근거 = 전부 실측)

| 근거 | 수치 | 출처 |
|---|---|---|
| **rescoring headroom 존재**(one2one 최종검출) | 완벽 q 로 small AP **0.140→0.260 (+0.12)**, AP50 +0.19 | premise 진단(naive-P2) |
| **q 가 IoU 를 학습함**(learnability) | Sp(q,IoU) small **0.504** / tiny 0.702 / all 0.483 | A1a-1 진단(val) |
| **q 가 cls 를 이김**(특히 small) | small: q 0.504 vs cls **0.362** (+0.142) | 〃 (같은 anchor 집합) |
| **head 가 base 를 안 다침** | A1a-1(rescore 없음) size-AP ≈ naive-P2 (small AP 0.1402→0.1413) | A1a-1 size-AP(test) |

→ "여지 있음"(+0.12) + "q 가 그 여지 겨냥하는 신호 학습됨"(small q≫cls) 두 기둥이 **모두 실측**. 남은 건 결합 여부 = A1b.

## 3. 알려진 리스크 / 코덱스 공격 예상 지점 (내 입장 명시)

| # | 공격 | 내 입장 (반박/수용/측정) |
|---|---|---|
| 1 | "rescoring probe 음성 → 접어라" | **반박**: 그 probe 는 **oracle(완벽 q)조차 flat(−0.003)** → 경로(one2many, 장당 ~990 후보 과밀·AP 포화) 부적합이지 q 실패 아님. +0.12 는 one2one 전용. **probe=판단불가**, 폐기 근거 아님. |
| 2 | "Sp(q,cls)=0.775 중복 → rescoring 무효" | **부분수용+반박**: 중복 높으나 q 가 small 서 cls **+0.142 초과** = 비중복 신호 존재. 크기는 예단 말고 **A1b eval 로 종결**. |
| 3 | "cls·q^γ score 압축 → AP50/recall 붕괴" | **반박+측정**: γ=1(약)·eval conf=0.001(full recall)·q 고IoU 캘리브 양호(q0.86→IoU0.86). **AP50·AR 동시 로깅 확인.** |
| 4 | "one2one topk 희소 → q 학습 실패" | **반박**: L_q=IoU-target-**all**(전 anchor, positive 수 무관). A1a-1(one2many) 강학습(0.50). one2one 재확인=진단스크립트. |
| 5 | "L_q 가 detector 오염(no-detach)" | **반박**: one2one 은 `x_detach` → **backbone 차단**. one2one reg branch 만 공유(원래 detached 학습 방식). base AP 감시 + A1a-0(detach) 대조군. |
| 6 | "VFL/GFL 이미 존재 → novelty 없음" | **부분수용**: IoU-quality 자체 선행. 차별점 = **small 특화(A2 λ_small) + one2one NMS-free head 결합 + small misalign 실측 동기**. 문헌 확인 후 표현(과장 금지). |

## 4. 무엇이 판정하나 (사전 등록)

- **본학습**: `260713_Y26S_P2_SAQ_A1b_VISDRONE` (naive-P2 와 동일 세팅, saq.yaml, rescoring on).
- **판정 eval**: `eval_size_ap.py --split test --max-det 1500`(conf 기본 0.001) — **추론에 rescoring 자동 적용**.
  ⚠ `--max-det` **기본 500** 이므로 naive-P2·A1a-1(1500)과 맞추려면 **반드시 `--max-det 1500` 명시**(Codex 지적).
- **대조군**: naive-P2, A1a-1(rescore 없음) — 셋 다 max-det 1500.
- **gate(사전)**: small AP **+0.02 의미 / +0.04 강함 / +0.06 매우 큼**(oracle 0.26 전부 기대 X). 부작용 금지: AP50·AR 하락, all AP 하락(base).
- **γ sweep(재학습 X)**: `eval_size_ap.py --saq-gamma 2`(신규 인자, head.saq_gamma 오버라이드). 1·2·4 재eval 로 최적 γ.
- **A1b 부진 시 ablation 후보**(Codex 2026-07-13): ① non-positive loss weight(**VFL** vs uniform-overlap vs ρ·u^δ)
  ② γ ③ λ_small(A2 크기 가중) ④ q 결합식(cls·q^γ vs additive/gating). → 부진이 "설계 결함"인지 "튜닝"인지 분리.

## 5. 코덱스에게 묻는 열린 질문

① one2one 에 q 를 얹은 게 맞나, 아니면 one2many 학습 + one2one 증류가 나은가?
② rescoring 을 **학습 중에도** 걸어 cls·q 공적응 시켜야 하나(현재는 inference-only)?
③ q-cls 중복 0.775 에서 `cls·q^γ` 대신 **q 를 additive/gating** 등 다른 결합이 나은가?
④ small 특화(A2)를 A1b 와 **동시**에 넣을까, 순차(먼저 순수 A1b)로 격리할까?
