# SOD 연구 인수인계 (2026-07-09) — 새 세션 이어가기용

이 문서 하나로 이어서 진행 가능하도록 정리. 상세는 각 링크 md 참조.

## 0. 프로젝트

- **목표**: Small Object Detection 개선 **새 모듈** 논문(MDPI). ⚠ 최우선 = **새 모듈 개발**(loss/assignment 튜닝은 기여 아님).
- **베이스**: YOLO26s (ultralytics 8.4.6 editable fork), 데이터 **VisDrone**(주력). fork=`/home/hsjeong/workspace/SOD-PAPER/ultralytics`.
- **환경**: conda `SOD-PAPER` 전체경로 필수 — `/home/hsjeong/miniconda3/envs/SOD-PAPER/bin/{python,yolo}`.
- **md 위치**: `/home/hsjeong/workspace/md/sod-paper/` (github 미업로드, 포크 git 미추적). 카테고리 폴더 00~06.

## 1. 현재 유효 스토리 (이것만 유효)

> P2 로 고해상 detection head 확보 → small(16-32) 병목을 **recall(feature 보존)** 과 **precision(quality ranking)** 으로
> **분해**. fusion/loss/refine/assignment probe 는 **negative ablation**.

- **① naive-P2** = 확정 양성(핵심 기여). 대조군 `260625_Y26S_P2_VISDRONE`, 예측 `eval/results/pred_P2v2_*`.
- **② small 개선 (진단으로 recall/precision 분리)**:
  - 진단(`saq_premise_diag.py`): oracle q-rescore `cls·IoU^γ` → small AP **0.140→0.260(+0.12)**, AP50 +0.19
    (**precision headroom 큼**). 단 **AR 0.3221 불변 = recall 벽**. cls-IoU misalign small Spearman 0.501<medium 0.637.

## 2. 진행 상태

| 트랙 | 정체 | 상태 | 브랜치 |
|---|---|---|---|
| naive-P2 | 고해상 head | ✅ 확정 양성 | `2-fpn` (baseline) |
| NWD assign probe | tal.py align blend | ❌ negative(AR 불변) | `2.4-probe_nwd_assign` |
| **SPD-Conv (recall)** | space-to-depth 다운샘플 | 🔄 **본학습 중(사용자 CLI)** ★현재 | `4-spdconv` |
| **SAQ (precision)** | quality head | ⏸ 보류(head 구현·smoke OK, L_q 미구현) | `3-saq-quality` |
| marine(BrackishMOT) | 예비 데이터 | ⏸ 보류(클래스 누락) | — |

- **negative ablation**(정직 기여): tiny(정보한계)·fusion 3방향(SGI/GSA/AFFG)·P2Refine·WIoU·NWD-assign-probe.

## 3. ★ 지금 하던 일 & 다음 액션

- **지금**: SPD-Conv **backbone-only** 본학습(사용자가 CLI 로, `260709_Y26S_P2_SPD_VISDRONE`). 코덱스 권고=backbone-only 먼저
  (full 은 neck+용량 교란). ablation: **P2 → +SPD-backbone → +SPD-full**.
- **결과 나오면**: `eval_size_ap.py`(--max-det 1500) 로 test 평가 → naive-P2 대비 **판정 핵심 = small AR 이 움직이는가**
  (AR↑ → recall 축 생존; AR 무변 → SPD negative, SAQ 중심). 결과는 `04_results/SOD_RESULT_AP_SIZE.md` 기록.
- **그 다음**: (AR↑ 시) SAQ `L_q` 구현 재개 → SPD+SAQ 결합. (AR 무변 시) SPD negative 처리, SAQ 단독 or 재진단.
- **recall 로그(코덱스 필수)**: small AR·AP50·matched/unmatched·class별 recall·P2/P3 level 분포.
  ⚠ **우리 end2end(NMS-free)** 라 "NMS 전/후" 대신 **one2many(assigner topk10 후보) recall vs one2one(최종) recall** 로 분리.

## 4. 학습 명령 (긴 학습은 사용자 CLI — 규칙)

```bash
cd /home/hsjeong/workspace/SOD-PAPER/ultralytics   # git checkout 4-spdconv 확인
/home/hsjeong/miniconda3/envs/SOD-PAPER/bin/yolo detect train \
  model=.../ultralytics/cfg/models/26/yolo26s-p2-spd.yaml \
  data=.../dataset/VisDrone/cleaning/data.yaml \
  epochs=300 patience=50 batch=32 imgsz=640 device=1,2,3 workers=4 \
  optimizer=auto seed=0 cos_lr=False close_mosaic=10 amp=True \
  project=.../runs/FPN name=260709_Y26S_P2_SPD_VISDRONE
```
- naive-P2 와 **유일 차이 = backbone SPD-Conv**. OOM 시 batch↓ 또는 device↑(batch total 32 유지).

## 5. 코덱스 열린 질문

① SPD backbone-only vs full — backbone 먼저(결정). ② recall 벽이 feature 한계인가 다른 원인(conf/maxDet/class
confusion/head capacity)인가 — 로그로 분리. ③ SPD(recall)+SAQ(precision) 결합 스토리 타당(조건: SPD AR↑ AND SAQ AP/AP75↑ AND 무충돌).

## 6. 작업 규칙 (메모리 = `~/.claude/.../memory/`)

- **한국어만** 응답. 사용자 **딥러닝 초급자** → 쉬운 비유·큰그림 자주(긴 검증 중 "지금 뭐하나" 반복 질문).
- **모듈 = 기존 연구 차용 + 허점지적 + 간단수정**(밑바닥 복잡설계 금지). 자체설계 모듈 다 실패함.
- **Codex 피드백 무비판 수용 금지** — 근거로 반박·논쟁, 실측으로 종결.
- **설계 제시 전 7문 자기검증**(`verify-sod-math` 스킬 §설계체크리스트): 학습-추론 대칭·평가metric정합·정보충분성·toy가정노출 등.
- **긴 본학습은 사용자 CLI**, 에이전트는 smoke(1~3ep)·eval·build 검증만.
- conda 전체경로, bare pip 금지.

## 7. 핵심 파일

- 설계: `03_modules/SOD_MODULE_SAQ.md`(v6 보류) · `SOD_MODULE_CANDIDATES.md`(SPD 등 대안) · `SOD_MODULE_SMALL_REFINE.md`.
- 결과: `04_results/SOD_RESULT_AP_SIZE.md`(oracle·probe·rescore). 로드맵: `00_roadmap/SOD_ROADMAP_PAPER_PROCESS.md`.
- 데이터: `01_datasets/SOD_DATASET_MARINE_BRACKISHMOT.md`(보류).
- 코드(fork): `SPDConv`=conv.py(Focus 별칭) · `SAQDetect`=head.py · tal.py NWD blend(2.4 브랜치).
- 스크립트(scratchpad, 세션별 휘발 주의): `saq_premise_diag.py`(rescore 진단)·`verify_saq*.py`.
- eval: `eval/eval_size_ap.py`(size AP, --max-det 1500) · `.claude/skills/sod-research-method/oracle_ceiling.py`.
