# Small-Aware Quality module (SAQ) — ★ **종료: ranking 축 NEGATIVE (2026-07-15)**

> **최종**: q 학습가능(o2m +0.50)·one2one 학습 붕괴(−0.33)·cross-path 랭킹 우수(+0.69)이나 **실제 AP +0.002(노이즈)**.
> quality-aware rescoring 은 oracle 여지(+0.048) 있으나 **learnable q 로 회수 불가**(랭킹상관 ≠ AP). **폐기.** 상세 §RESULT⑤.


> ⚠ **콜드 리더(코덱스) 시작점**: 아래 "상태" 블록이 **현재 진실**. 본문 §2/§6 은 설계 근거이며 일부 forward-looking
> 표현(“첫 구현 detach” 등)은 이미 실현/갱신됨 — **현재 배포 설계 = A1b(q on one2one + 추론 rescoring)**.
> 대결용 요약은 `SOD_SAQ_A1B_CODEX_BRIEF.md` 참조.

작성 2026-07-09. **v6** = Codex 4·5차 반영: **ignore band 폐기 → IoU-target-all**(t=max plain IoU, §3),
**detach-first → control**(A1a-1 no-detach = main, §6), 전제 실측(oracle rescore headroom +0.12).
최우선=새 모듈(`project_sod-goal-new-module`). 연계 `SOD_MODULE_SMALL_REFINE.md`, `../04_results/SOD_RESULT_AP_SIZE.md`.
상태: **▶ L_q 구현 완료·smoke PASS (2026-07-10)** — `3-saq-quality` 브랜치(미커밋 working tree).
- 구현: `loss.py` `SAQDetectionLoss`(4-vec [box,cls,dfl,q], IoU-target-all VFL + Σw 정규화) + `SAQE2ELoss`
  (w_q=0.1 고정, o2m decay 무관, **rescoring 없음=inference 미변경**) · `tasks.py` init_criterion 분기 · `train.py` loss_names 4번째 `q_loss`.
- smoke 2종 PASS: ① unit(build·quality(2,1,34000)·4-vec·backward·cv_q grad>0) ② 2ep 학습(q_loss 로깅·크래시無, train q_loss 0.816).
- **SPD-Conv(recall 축) backbone·full 모두 negative 확정**(§RESULT, small AR 0.32 불변) → recall ROI 소진, SAQ(ranking) 정당.
- **진단 스크립트 완성**: `eval/saq_quality_diag.py` — best.pt 로 val 후보 anchor(t>0.1)의 **Spearman(q,IoU) vs Spearman(cls,IoU) 크기대별**
  + 캘리브레이션 + q-cls 상관 출력(BN eval 고정·head.training만 True로 q 추출). smoke 가중치로 동작 검증 완료.
- **A1a-1 결과 (2026-07-13): learnability PASS(강).** `260710_..._SAQ_A1a1_VISDRONE3`(RUN_260710_03).
  Sp(q,IoU) small **0.504** > cls 0.362(+0.142), tiny 0.702>0.623, all 0.483>0.379 — 전 밴드 q>cls. base 무해(size-AP≈naive-P2). §RESULT.
  - ⚠ 리스크: **Sp(q,cls)=0.775 고중복**(rescoring 이득 미지) · **q=one2many 학습인데 inference=one2one**(A1b 는 one2one 에 q 필요) · 저-q 캘리브 비단조.
- **rescoring probe(2026-07-13): INCONCLUSIVE.** one2many 후보에선 **oracle(완벽 q)조차 small AP 여지 0**(−0.003) →
  경로 부적합(장당 ~990 후보 과밀, AP 포화). +0.12 여지는 one2one 전용. probe 로 SAQ 판단 불가 → **A1b(one2one) 직행.** §RESULT.
- **▶ A1b 구현 완료·smoke PASS (2026-07-13):** SAQDetect `saq_rescore=True`(기본). **q 를 one2one reg feature 에서 학습(L_q)** +
  **추론 `_inference_rescore`: score=cls·q^γ**(γ=1). 학습·추론 **동일 경로 = 대칭성 확보(A1a-1 발견1 해소).**
  - 코드: head.py SAQDetect.forward/_inference_rescore/_forward_quality(box_head 인자화) · loss.py SAQE2ELoss q_loss 를 one2one 에서.
  - smoke 2종 PASS: unit(one2one q·4-vec·cv_q grad·rescore on≠off) + 2ep 학습.
  - **★ A1b = NEGATIVE (2026-07-14):** `260713_..._SAQ_A1b_VISDRONE`(RUN_260713_01), 246ep. test @maxDet1500 §RESULT.
    ① base(γ0)≈naive-P2(검출망 무해) ② as-deployed(γ1) small AP −0.001 ③ 결합식 probe: **학습 q 전 결합식 ≤ 기준, oracle +0.046**.
  - **원인(규명)**: A1b one2one q 가 **IoU 와 반대로 학습**(Sp small −0.335 vs A1a-1 one2many +0.504). one2one NMS-free 억제 anchor
    reg feature 가 IoU 정보 결핍 → q 오학습. = **one2many/one2one 딜레마(7Q 발견1) 현실화.** 개념 여지(oracle +0.046)는 실재.
  - **★ SAQ 최종 종료 (2026-07-15): ranking 축 negative (완전 규명).** cross-path gate/AP(재학습0) §RESULT⑤.
    gate: one2many q → one2one 박스 IoU **Sp +0.686**(box mismatch 반증, Codex #2 틀림). 그러나 AP: one2many q 로 rescore 해도
    small AP **+0.0022(노이즈)**, oracle +0.048 의 ~5%만. **랭킹상관 ≠ AP**(one2one 이 이미 top 정렬 → 재정렬 무의미).
  - **결론**: quality-aware rescoring 은 **oracle 여지 실재하나 learnable q 로 회수 불가**. Codex #4(ROI) 옳음. **SAQ 폐기.**
    recall(SPD·NWD)·precision(SAQ) 양축 벽. 확정 기여=naive-P2. 신규툴 `--saq-gamma`·`saq_combo_probe.py`·`saq_crosspath_{gate,ap}.py`.
(설계 v6 완성: A1a-1(no-detach)=main, A1a-0(detach)=control, IoU-target-all, NWD/A3 보류.)

---

## 0. 한 줄 정의

> 각 예측의 **위치 품질 q∈[0,1]**(연속) 을 **모든 detection level**의 경량 head 로 예측하고,
> inference 순위를 `score = cls_prob · q^γ` (γ 균일)로 보정한다.
> q 는 **전체 anchor** 학습(**IoU-target-all**: t=max plain IoU, no-overlap=0; assignment 는 **loss weight 로만** 구분, Varifocal + **Σw 정규화**).
> **small 특화는 학습 loss weight 에서만**(NWD 는 2차·보류). branch/inference 는 전 스케일 균일.
> **SAQ 는 raw candidate recall 을 직접 안 늘린다 — 후보 재점수화·FP억제·ranking 개선이 주효과**(§9-K).

## 1. 왜 이 모듈인가 (진단 — headroom 실측)

- **oracle small det 0.76 ≫ loc 0.34** → 여지는 검출/순위/품질정렬. P2Refine(좌표) 반증.
- **가설**: small 은 정확히 위치해도 cls score 가 낮아 PR ranking 에서 FP 에 밀린다(cls≠localization quality).
- **✅ 전제 *일부* 실측 (naive-P2, `saq_premise_diag.py`, §9-K)**:
  - misalignment: cls↔IoU Spearman small **0.501** < medium 0.637.
  - **ranking headroom**: 완벽 q 로 `cls·IoU^γ` → small AP **0.14→0.26(+0.12)**, AP50 **+0.19**.
  - ⚠ **확인된 것 = headroom(상한) 뿐. "q head 가 그 q 를 실제로 학습하는가(learnability)"는 미확인 → A1a/A1b 로 검증.**
- **한계 (Codex1, 실측)**: rescore 로 AR 0.3221 불변 → **SAQ 는 raw candidate recall 을 직접 못 늘림**
  (단 NMS 순서·conf threshold 때문에 **post-NMS AR 은 약간 변동 가능**). recall 주역은 **probe/topk/assignment**.
  → SAQ = det oracle 중 **ranking/FP억제 부분**. unmatched small GT 가 주병목이면 probe 와 **결합**(A4).

## 2. 모듈 구조

```
  각 level(P2~P5) reg branch feature ─►[Quality head: 공유 2×conv+sigmoid]─► q∈[0,1] (연속)
       │  (첫 구현은 input feature.detach() — learnability 먼저, §6 A1a-0)
       ▼
  cls head ─► cls_prob ─────────────────────► score = cls_prob · q^γ   (γ 균일, 전 레벨 동일)
```

- **전 레벨 q head**(P2~P5 공유). P2-only 금지(§9-H). **q 연속값**(이진 금지, §9-K).
- **q 입력 = reg branch feature** (첫 구현 detach; 이후 no-detach 비교 §6). backbone-only feature 는 box 정확도 정보
  부족(§9-J) → reg 계열.
- **γ 첫 실험 = 1** (필수, §9-P): γ=4 는 q noise 시 small TP 도 같이 죽임. score compression 대응 = **val conf 를
  최대한 낮게 고정**(안 그러면 ranking 개선이 아니라 threshold 탈락을 측정).

## 3. 학습 (수식)

기호: 앵커 i(전체), positive P, assigned GT g=A(i), 예측박스 b_i, GT B_g, u_i=maxIoU(anchor i 와 임의 GT).

**(1) quality target = IoU-target-all** (ignore band 폐기 — §9-W 논쟁, Codex 수용):
```
  t_i = max_j plain_IoU(b_i.detach(), GT_j)     # 전 anchor. no-overlap 이면 자연히 0. (τ_ig 없음)
```
- **assignment 여부와 무관하게 target=IoU** → q 는 **class-agnostic localization quality** 를 배움.
- **plain IoU(Codex2)**: `bbox_iou(CIoU=False)`. CIoU 는 음수·penalty 섞여 AP TP 기준과 어긋남. **detach**(§5).
- **ignore band 폐기 이유(내 반박, Codex 수용)**: ignore=무학습 오염(§9-G 재발). all-target = 오염0 + high-IoU
  unassigned 안 죽임 + FP(낮은 t) 자동 억제.
- **VFNet/GFL 은 왜 positive-only?**(Codex): 그쪽 quality 는 **class-specific**(cls score 와 결합) → negative 에
  IoU 주면 class target 과 충돌. SAQ 의 q 는 **class-agnostic**(존재여부는 cls head 별도) → all-anchor 가 자연.

**(2) quality loss** (weight 로 assignment 구분 + 정규화, Codex3):
```
  L_q = ( Σ_i w_i · VFL(q_i, t_i) ) / max( Σ_i w_i , 1 )        # ★ Σw 정규화(bg 압도 방지)
  w_i : assigned positive = λ_s(g)         # λ_small(>1) if small else 1 (size 기준, 클래스무관 §9-N)
        unassigned(겹침)   = 1             # ⚠ 이건 설계값. **실제 코드는 non-positive 전부 α·σ(q)^γ(VFL)** — uniform-1 은 bg 압도로 미채택, overlap-uniform 은 향후 ablation (Codex 2026-07-13 지적)
        clear background   = focal negative (t≈0, down-weight)
  L = L_box + L_cls + L_dfl + w_q·L_q,   w_q **0.1 부터** (§9-Q). maxIoU 는 **chunked 계산**(메모리, Codex).
```
- ⚔ **soft weight 논쟁(내 부분반박)**: Codex 는 unassigned 를 `ρ·u_i^δ` 로 **처음부터 약하게**(중간-IoU
  duplicate FP 방지). **내 반박**: unassigned 는 TAL 이 안 뽑은 것 = **cls 가 낮음** → `cls·q^γ` 도 낮아 FP 안 됨.
  ρ,δ 를 미리 넣으면 **교란변수↑**. → **첫 실험 uniform(w=1)** 으로 깨끗하게, **duplicate FP 로그가 실제 나쁘면
  그때 soft weight(ρ·u^δ) 도입**. (미해결 — NMS/duplicate 로그로 결판)

**(3) NWD — A3, 지금 구현 안 함(Codex6 보류)**:
- A1(IoU-only SAQ)이 oracle 상한 0.26 의 일부라도 회수하는지 **먼저 확인**. 안 되면 NWD novelty 무의미.
- A1·A2 성공 후 택1: **A3-b2(추천)** = 별도 `q_nwd` auxiliary head, **inference 엔 q_iou 만**(AP 정렬 유지, NWD 는
  auxiliary 학습신호로만). / A3-b1(NWD=small low-IoU sample weight) / A3-b3(bounded target, VFL weight 는 IoU 고정).

## 4. 기존 논문 차별점 (표현 완화, 서베이 전)

- IoU-Net/GFL/VFNet: IoU quality target, 전 스케일. **SAQ novelty(잠정)**: small-object IoU 포화를 **NWD-aware
  auxiliary(A3-b2)로 완화** + small loss 가중. "첫 시도" 단정은 문헌 확인 후.

## 5. 기존 실패와 무충돌 (Codex4 정정)

- ✅ box 좌표 target/loss 직접 안 바꿈(≠P2Refine). ✅ fusion 강제 안 함(≠SGI/GSA). ✅ loss 재가중 아님(≠WIoU).
- ⚠ q head 가 reg feature 를 쓰면 L_q 가 공유 feature 통해 reg 에 **간접 영향** 가능 → **첫 구현 input detach 로 차단**,
  A1a-1 에서 no-detach 영향 측정.

## 6. 판정 / ablation (Codex5 — A1 3단 격리)

- 대조군 naive-P2. 지표 mAP50/50-95+COCO APs/APm+small AP50/75/AR. **multi-seed≥3.**
```
  A0     naive-P2
  A1a-1  q head(no detach) + no rescoring   ← ★main learnability: q↔IoU 배우나 + main AP 감시(w_q=0.1)
  A1a-0  q head(detach)    + no rescoring   ← control: reg feature 에 IoU 정보 있나 / 오염 해석용 대조군
  A1b    no-detach + cls·q^γ rescoring(γ=1) ← q 캘리브레이션·γ 문제
  A2     + small loss weight (λ_small)
  A3     + NWD auxiliary (A3-b2)            ← A1·A2 성공 후에만
  A4     + TAL NWD probe 결합 (recall 보완)
```
- **선결 = A1a-1(no-detach)**: quality head 가 실제로 q↔IoU 를 배우는가(learnability). **병행 가능하면 A1a-0/A1a-1
  같이** 돌려 오염 여부까지 한 번에. (detach-first 논쟁 종결 — Codex 가 detach 를 control 로 격하 수용.)
- **no-detach 조건**: w_q=0.1 부터, no-rescoring 상태에서 **main AP 하락 감시**, q-IoU calibration + main AP 동시 기록.
- **중단기준(Codex)**: ① A1a-1 q-IoU Spearman ~0(학습 실패) ② A1b small AP +0.01 이하 ③ FP/duplicate↑로 mAP 하락.
- ⚔ **내 반박(중단기준 ①)**: "q-IoU Spearman 이 cls-IoU(0.501)와 비슷/못하면 접어라"는 **이르다** — q·cls 가 상관 같아도
  **서로 다른 실수를 하면 cls·q^γ 결합에서 이득**(앙상블). 진짜 gate 는 **A1b 실제 rescore AP**, q-IoU 상관 단독 아님.
  → A1a-1 은 "q 가 아예 학습되나(Spearman≫0)"만 gate, 성패는 A1b 로. (Codex 재반론 받을 것.)
- **기대치(Codex)**: 상한 rescore ~0.26(det 0.762 아님). small AP **+0.02 의미 / +0.04 강함 / +0.06 매우 큼**. multi-seed.
- **probe 결과 반영(2026-07-09)**: NWD assignment probe negative(§RESULT). recall 축 이 실험군 ROI↓ → SAQ(ranking)가
  다음 수로 정당. **learnability 약한 경고**(probe 도 학습개입 실패) → A1a-1 q calibration 이 첫 관문.
- **필수 로그**: 예측 q vs 실제 IoU **캘리브레이션 곡선/Spearman**(learnability, 실패 1순위) · bg/ignore q 분포 ·
  **rescore 상한 0.26 대비 회수율** · per-level q 분포 · (A1b) score 분포·val conf 영향.

## 7. 열린 질문

1. A3 방식(b1/b2/b3) — A1·A2 후 확정.
2. q 입력 detach(A1a-0) vs no-detach(A1a-1) 어느 쪽이 learnability·오염 균형 좋은가.
3. γ(1→?) : 진단 K 는 클수록 AP75↑ 이나 q noise 시 TP 손실 — 실측으로.
4. τ_ig(ignore, ≈0.3) · λ_small · w_q(0.1~0.25) 튜닝.

## 8. probe 와의 관계

- **probe/topk/assignment = recall(후보 생성)**, **SAQ = ranking/FP억제(후보 재점수)**. 진단 K 로 분리 확인(AR 불변).
  상보적, A4 결합. probe 학습 중(GPU 6·7).

## 9. 자체검증 로그

**1차(v1→v2)**: A.η(s)폐기 B.rescoring유효 C.상시NWD미미 D.닭달걀오류 E.detach F.novelty.
**2차(v2→v3, Codex 치명3)**: G.fg-only bg오염 H.P2-only=scale-gate I.raw NWD AP충돌 J.q정보부족.
**3차(v3→v4, Codex6+진단)**: K.headroom실측(0.14→0.26)·AR불변 L.plain IoU M.A1세분 N.NWD gradient죽음 O."좌표안밈"정정.
**4차(v4→v5, Codex7)**:
- **P. "전제 확증" 과함 [표현]**: 확인된 것은 headroom(상한)·misalignment 뿐. learnability 는 A1 미검증. §1 낮춤.
- **Q. L_q 정규화 누락 [내 체크리스트7 놓침]**: Σ VFL 은 bg 수 압도로 발산 → `/max(num_pos,1)`, w_q 0.1~0.25. §3-(2).
- **R. ignore band 누락 [내 체크리스트7 놓침]**: high-IoU unassigned 를 q=0 으로 죽이면 "좋은 후보 죽이기" 학습 →
  τ_ig≈0.3 ignore. §3-(1).
- **S. recall 표현 정정 [Codex2]**: "못 올림"→"raw candidate recall 직접 안 늘림, post-NMS AR 미세변동 가능". §1.
- **T. A1 3단 격리 [Codex5]**: A1a-0(detach)/A1a-1(no-detach)/A1b(rescore). §6.
- **U. NWD 보류 + A3-b2 [Codex6]**: A1·A2 성공 후, 별도 q_nwd aux head·inference 엔 q_iou 만. §3-(3).
- **V. γ=1·val conf 낮게 = 필수체크 [Codex7]**: score compression 대응. §2.
- **W. ⚔ 내가 Codex 에 반박(수용 안 함) — 사용자 지시 "무비판 수용 말고 근거로 반박"**:
  - (vs #4) ignore band → **오염 재발(§9-G) 반박**, 대안 **IoU-target-all**(t=maxIoU, λ 로만 pos/neg 구분). §3-(1).
  - (vs #5) detach-first → **learnability 불리·원인 모호 반박**, no-detach 먼저/병행. §6.
  - (vs #2) recall: 표현은 수용하되 **우리 실측 AR 0.3221 불변은 데이터**로 유지. §1.
  - → **결과(5차): Codex 가 반박 2개 둘 다 수용.** ignore band 폐기·IoU-target-all 채택, detach→control 격하.

**5차(v5→v6, 논쟁 수렴 + 내 부분반박)**:
- **W′. ignore band 폐기 확정**: IoU-target-all `t=max_j IoU(b_i.detach(),GT_j)` (§3-1). Codex 수용 + VFNet
  positive-only 이유(class-specific quality) 규명 → SAQ 는 class-agnostic q 라 all-anchor 정당.
- **X. detach-first → control 격하 확정**: A1a-1(no-detach)=main learnability, A1a-0(detach)=control (§6).
- **Y. ⚔ soft weight 도입시점 [내 부분반박, 미해결]**: Codex 는 unassigned=ρ·u^δ 처음부터. 내 반박: unassigned 는
  cls 낮아 FP 안 됨 → 첫 실험 uniform, duplicate FP 로그 나쁘면 그때 soft. (§3-2, NMS 로그로 결판)
- **수용(반박 안 함)**: Σw 정규화, chunked maxIoU, w_q=0.1, VFNet 설명.

## 10. 설계 자기검증 체크리스트 (`verify-sod-math` 등재)

1. 학습-추론 대칭 · 2. 내부 일관성 · 3. 평가 metric 정합 · 4. 정보 충분성 · 5. toy 가정 노출 ·
6. 증거-주장 강도 · **7. imbalance/degenerate/누수(← 이번 Q·R 여기서 놓침 — L_q 정규화·ignore band).**
