# SOD 논문 인수인계 문서 — 2026-06-22

> 이 문서는 현재 Claude Code 세션에서 다음 Claude 세션으로 연구 상태를 인계하기 위한 문서입니다.
> 작업 디렉토리: `/home/hsjeong/workspace/SOD-PAPER/`
> ultralytics repo: `/home/hsjeong/workspace/SOD-PAPER/ultralytics/` (editable install, 8.4.6)
> 개인 remote: `jeonghs9/SOD-PAPER`

---

## 1. 연구 전체 맥락

- **목표**: Small Object Detection(SOD) 개선 논문, MDPI 게재
- **베이스 모델**: YOLO26s/n (ultralytics 8.4.6), optimizer=muSGD (YOLO11은 SGD, 비교용)
- **데이터셋**: VisDrone (10클래스) / DOTA v1.5 HBB tiled (16클래스) / AI-TOD (8클래스)
- **확정된 연구 방향 (방향 B)**: **구조 개선(FPN) + 학습 개선(size-adaptive loss)** 복합 기여 논문
- **Ablation 매트릭스 순서**: `Baseline → +Loss → +FPN → +Full`
  - Loss를 먼저 단독으로 확정한 뒤 FPN으로 넘어가는 전략 (결과 분리 목적)

---

## 2. 완료된 작업

### Phase 0 — 환경/데이터셋 구축 (완료)

| 항목 | 경로 |
|---|---|
| VisDrone cleaning | `dataset/VisDrone/cleaning/` (train/val/test, 10cls) |
| DOTA tiled | `dataset/DOTA/cleaning_tiled/` (crop=1024, gap=200, 16cls) |
| AI-TOD cleaning | `dataset/AI-TOD/cleaning/` (8cls) |
| eval 스크립트 | `eval/eval_size_ap.py` (pycocotools 크기별 AP) |

### Phase 1 — Baseline 학습 (완료)

모든 상세 결과: `../01_datasets/SOD_DATASET_VISDRONE.md`, `../01_datasets/SOD_DATASET_DOTA.md`, `../01_datasets/SOD_DATASET_AITOD_RESULTS.md`

**VisDrone val mAP (ultralytics 내장, conf=0.25)**

| 모델 | Optimizer | mAP50 | mAP50-95 |
|---|---|---|---|
| YOLO26s | muSGD | 0.396 | 0.246 |
| YOLO26n | muSGD | 0.377 | 0.232 |
| YOLO11s | SGD | 0.405 | 0.253 |
| YOLO11n | SGD | 0.365 | 0.224 |

**VisDrone pycocotools 크기별 AP (YOLO26s, test split, conf=0.001)**

| band | AP | AR |
|---|---|---|
| vt(2-8px) | 0.0205 | 0.0583 |
| tiny/APT(8-16px) | 0.0564 | 0.1682 |
| small(16-32px) | 0.1191 | 0.2925 |
| medium(32+px) | 0.2558 | 0.4415 |
| AP@[.5:.95] | 0.1646 | — |

**DOTA val mAP**: YOLO26s 0.604 / YOLO26n 0.581 / YOLO11s 0.616 / YOLO11n 0.593  
**AI-TOD val mAP**: YOLO26s 0.570 / YOLO26n 0.493 / YOLO11 미완료

### Phase 2 — Loss 트랙 단독 검증 (완료)

실험 조건: YOLO26s + muSGD + VisDrone, epochs=300, patience=50, batch=32, imgsz=640, seed=0

#### 학습 결과 (ultralytics val split, conf=0.25)

| 실험 ID | Loss | ratio/C | mAP50 | mAP50-95 | P | R | best epoch | 브랜치 |
|---|---|---|---|---|---|---|---|---|
| BASELINE | CIoU | — | 0.396 | 0.246 | 0.517 | 0.283 | 203 | main |
| RUN_260617_03 | InnerCIoU | r=1.0 | 0.403 | 0.251 | 0.525 | 0.285 | — | `1-loss` |
| RUN_260618_01 | InnerSIoU | r=1.0 | 0.406 | 0.253 | 0.528 | 0.285 | 131 | `1-loss` |
| RUN_260619_01 | NWD | C=14.5 | 0.410 | 0.248 | 0.561 | 0.264 | 300 | `1.1-loss_nwd` |

#### pycocotools 크기별 AP (test split, conf=0.001) — 2026-06-22 측정

| size band | Baseline | InnerCIoU r=1.0 | InnerSIoU r=1.0 | NWD C=14.5 |
|---|---|---|---|---|
| vt(2-8) | 0.0205 | 0.0169 | 0.0149 | 0.0137 |
| **tiny/APT(8-16)** | 0.0564 | **0.0632** | 0.0568 | 0.0498 |
| small(16-32) | 0.1191 | 0.1241 | 0.1221 | 0.1117 |
| medium(32+) | 0.2558 | 0.2638 | **0.2660** | 0.2476 |
| **AP@[.5:.95]** | 0.1646 | 0.1697 | **0.1703** | 0.1570 |
| AP50 | 0.2918 | 0.3020 | **0.3039** | 0.2886 |

**Δ vs Baseline**

| size band | InnerCIoU | InnerSIoU | NWD |
|---|---|---|---|
| tiny/APT(8-16) | **+0.0068** | +0.0004 | -0.0066 |
| medium(32+) | +0.0080 | **+0.0102** | -0.0082 |
| AP@[.5:.95] | +0.0051 | **+0.0057** | -0.0076 |

---

## 3. 핵심 발견 및 해석

### 3-1. ratio=1.0의 의미 (중요!)

**InnerCIoU(r=1.0) = CIoU, InnerSIoU(r=1.0) = SIoU 와 수학적으로 동일.**

```
ratio=1.0 → inner box = original box
inner_iou = original_iou (변화 없음)
```

즉 지금까지의 Inner 계열 실험은 **ratio 효과를 전혀 사용하지 않은 상태**입니다.
InnerCIoU가 APtiny +0.0068을 보인 것은 훈련 분산(noise)이지 Inner 효과가 아닙니다.
**실제 Inner 효과는 ratio < 1.0 (예: 0.5, 0.7) 에서 나옵니다.**

### 3-2. NWD 결론

- val mAP50=0.410으로 최고였지만 **pycocotools test 기준으로는 전 구간 Baseline 이하**.
- 원인: NWD는 high-confidence 예측에 집중 → conf=0.25 val에서는 유리, conf=0.001 full-recall test에서는 불리.
- NWD 단독 적용은 SOD 개선에 비효율적. **단독 실험 방향으로는 중단 권장**.
- (향후 보조 loss 결합 가능성은 열어두되 우선순위 낮음)

### 3-3. InnerSIoU 우위

pycocotools 기준 AP@.5:.95(0.1703), APmedium(0.2660) 최고. 전반적 localization 정밀도 최우수.
단, APtiny는 InnerCIoU(0.0632)가 더 높음 — tiny 특화는 InnerCIoU, 전반적 성능은 InnerSIoU.

---

## 4. 미완료 / 다음 실험 후보

### 4-1. 즉시 실험 가능 (코드 이미 준비됨)

**InnerSIoU ratio 튜닝** — `1-loss` 브랜치에서 `loss.py` 한 줄 수정:
```python
# 현재: InnerSIoU=True, ratio=1.0
# 변경: ratio=0.7 또는 ratio=0.5
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, InnerSIoU=True, ratio=0.7)
```
- Inner 원논문 권장: ratio=0.5~0.8
- InnerCIoU도 같은 방식으로 ratio 실험 가능

**InnerCIoU ratio 튜닝** — APtiny가 목적이라면 InnerCIoU ratio=0.7도 병행 실험 가치 있음.

### 4-2. 신규 Loss 후보 (구현 필요)

| Loss | 특징 | SOD 관련성 | 구현 난이도 |
|---|---|---|---|
| **WIoU v3** | 동적 focusing, outlier 억제 | 높음 | 중 |
| **Alpha-IoU** | IoU^α 고차 loss, 저IoU 쌍에 집중 | 중간 | 낮음 (한 줄) |
| Shape-IoU | 형태 유사도 | 중간 | 중 |

### 4-3. 방향 재고가 필요한 지점

현재 실험 모두 ratio=1.0이라 Inner의 진짜 효과를 **한 번도 확인하지 않은 상태**입니다.

**재고할 핵심 질문:**
1. InnerSIoU / InnerCIoU ratio < 1.0 실험을 먼저 해서 효과를 확인해야 할까?
2. WIoU 같은 완전히 다른 접근법이 더 유망할까?
3. Loss 트랙에서 얼마나 더 실험할 것인가? (논문 일정 고려)

---

## 5. 구현 현황 (코드)

### metrics.py (`ultralytics/utils/metrics.py`)

| 브랜치 | 추가 함수/파라미터 |
|---|---|
| `1-loss` | `bbox_iou`에 `InnerCIoU=False`, `InnerSIoU=False`, `ratio=1.0` 파라미터 추가 |
| `1.1-loss_nwd` | `wasserstein_nwd(box1, box2, xywh, C, eps)` 신규 함수 (main 기반, Inner 없음) |

### loss.py (`ultralytics/utils/loss.py`)

| 브랜치 | 변경 내용 |
|---|---|
| `1-loss` | `BboxLoss.forward()` → `bbox_iou(..., InnerSIoU=True, ratio=1.0)` |
| `1.1-loss_nwd` | `BboxLoss.__init__`에 `use_nwd=True, nwd_C=14.5`; forward에서 분기 |

### tal.py — 변경 없음 (TAL assigner는 CIoU 유지, 모든 브랜치 공통)

---

## 6. 브랜치 구조

```
remote: jeonghs9/SOD-PAPER
├── main          — ultralytics 8.4.6 기본 (CIoU default)
├── 0-moe_loss    — MoE loss 실험 (별도 트랙, 현재 세션과 무관)
├── 1-loss        — InnerCIoU, InnerSIoU 구현 (ratio=1.0으로 학습 완료)
└── 1.1-loss_nwd  — NWD 구현 (main 기반, C=14.5 VisDrone 기반)
```

---

## 7. 파일 맵

| 파일 | 내용 |
|---|---|
| `../01_datasets/SOD_DATASET_VISDRONE.md` | VisDrone baseline val/test 전체 결과 |
| `../01_datasets/SOD_DATASET_DOTA.md` | DOTA 전처리 + baseline 결과 |
| `../01_datasets/SOD_DATASET_AITOD_RESULTS.md` | AI-TOD baseline (YOLO11 미완) |
| `../02_loss/SOD_LOSS_EXPERIMENTS.md` | Loss 실험 전체 기록 (val 결과 + 클래스별) |
| `../04_results/SOD_RESULT_AP_SIZE.md` | pycocotools 크기별 AP 전체 비교 (Baseline + 3종 loss) |
| `../00_roadmap/SOD_ROADMAP_PAPER_PROCESS.md` | 전체 연구 흐름 로드맵 |
| `../06_handoff/SOD_HANDOFF_NWD_EXPERIMENT.md` | NWD 실험 설계 문서 (가설 → 기각됨) |
| `../05_eval/SOD_EVAL_PIPELINE.md` | pycocotools 평가 파이프라인 사용법 |
| `eval/eval_size_ap.py` | 크기별 AP 측정 스크립트 |

---

## 8. 학습 명령어 템플릿

```bash
# 브랜치 확인 필수 (어느 브랜치의 loss.py를 쓰느냐가 핵심)
git branch  # 현재 브랜치 확인

# Loss 실험 (1-loss 브랜치 기준)
yolo detect train \
  model="/home/hsjeong/workspace/SOD-PAPER/ultralytics/ultralytics/cfg/models/26/yolo26s.yaml" \
  data="/home/hsjeong/workspace/SOD-PAPER/dataset/VisDrone/cleaning/data.yaml" \
  epochs=300 patience=50 batch=32 imgsz=640 \
  device=6,7 workers=4 seed=0 \
  project="/home/hsjeong/workspace/SOD-PAPER/ultralytics/runs/BASELINE" \
  name="260622_Y26S_<실험명>_VISDRONE"

# 크기별 AP 측정
cd /home/hsjeong/workspace/SOD-PAPER
python eval/eval_size_ap.py \
  --weights ultralytics/runs/BASELINE/<실험폴더>/weights/best.pt \
  --data dataset/VisDrone/cleaning/data.yaml --split test \
  --imgsz 640 --device 0 --max-det 500 --conf 0.001 \
  --tag <태그명>
```

---

## 9. 주의사항

- **평가 기준 혼용 금지**: ultralytics val(conf=0.25) 수치와 pycocotools test(conf=0.001) 수치를 같은 표에 쓰지 말 것
- **muSGD 고정**: YOLO26 실험은 반드시 muSGD (기본값). SGD로 바꾸면 비교 무효
- **seed=0 고정**: 재현성
- **tmp 경로**: `/tmp` 금지, `/home/hsjeong/tmp/` 사용
- **브랜치 확인 필수**: `1-loss`는 InnerSIoU default, `1.1-loss_nwd`는 NWD default, `main`은 CIoU default
