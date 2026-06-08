# YOLO26-MASTER 실험 결과

## 실험 목록

| # | 실험명 | 브랜치 | 날짜 | 설명 |
|---|--------|--------|------|------|
| 1 | 260519_YOLOS26-MASTER_BASELINE-7 | main | 2026-05-19 | YOLO26-MASTER 베이스라인 |
| 2 | 260526_YOLOS26-MASTER_Weight-residual | - | 2026-05-26 | Weight residual 실험 |
| 3 | 260529_YOLOS26-MASTER_BASELINE_moeloss | - | 2026-05-29 | MoE loss 초기 적용 |
| 4 | 260529_YOLOS26-MASTER_BASELINE_moeloss-2 | 0-moe_loss | 2026-05-29 | MoE loss cosine decay + batch_size 스케일링 + broadcasting 수정 (학습 진행 중, ~187 epoch) |
| 5 | 260601_YOLOS26-MASTER_Weight-residual | 1-residual | 2026-06-01 | Weight residual 재실험 (학습 진행 중, ~172 epoch) |

> **공통 학습 설정**: `device=0,1,2,6` (4 GPU), `batch=28`, `imgsz=1024`, `epochs=600`, `seed=0`

---

## All Test 결과 (COCO / AI Hub / KW)

### 260529_YOLOS26-MASTER_BASELINE_moeloss-2
- **브랜치**: `0-moe_loss`
- **설명**: E2ELoss MoE aux loss에 cosine decay 적용 (0.3→0.05) + batch_size 스케일링 버그 수정 + broadcasting 수정
- **모델**: `runs/YOLO26/260529_YOLOS26-MASTER_BASELINE_moeloss-2/weights/best.pt`

#### 3차 측정 — 2026-06-08 (epoch ~430)
- **환경**: conda `YOLO26-MASTER`, 브랜치 `0-moe_loss` / GPU: CPU / imgsz=1024, conf=0.25, batch=1, split=test

**COCO (data_qnsfl_new)** — 6701 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 6701 | 34587 | 0.778 | 0.680 | **0.664** | **0.434** |

**AI Hub (DET_AIHUB_PLATE_LoRA)** — 770 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 770 | 30583 | 0.756 | 0.444 | **0.409** | **0.239** |

**KW (DET_KW)** — 327 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 327 | 1528 | 0.909 | 0.615 | **0.607** | **0.420** |

#### 2차 측정 — 2026-06-05 (epoch ~244)
- **환경**: conda `YOLO26-MASTER`, 브랜치 `0-moe_loss` / GPU: CPU / imgsz=1024, conf=0.25, batch=1, split=test

**COCO (data_qnsfl_new)** — 6701 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 6701 | 34587 | 0.771 | 0.675 | **0.662** | **0.429** |

**AI Hub (DET_AIHUB_PLATE_LoRA)** — 770 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 770 | 30583 | 0.763 | 0.434 | **0.402** | **0.234** |

**KW (DET_KW)** — 327 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 327 | 1528 | 0.948 | 0.571 | **0.568** | **0.423** |

#### 1차 측정 — 2026-06-04 (epoch ~187)
- **환경**: conda `YOLO26-MASTER`, 브랜치 `0-moe_loss` / GPU: CPU / imgsz=1024, conf=0.25, batch=1, split=test

**COCO (data_qnsfl_new)** — 6701 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 6701 | 34587 | 0.775 | 0.666 | 0.651 | 0.425 |
| person | 6219 | 27566 | 0.826 | 0.706 | 0.703 | 0.515 |
| car | 1488 | 5887 | 0.803 | 0.679 | 0.671 | 0.477 |
| motorcycle | 74 | 204 | 0.694 | 0.588 | 0.558 | 0.363 |
| plate_number | 557 | 930 | 0.777 | 0.692 | 0.673 | 0.344 |

**AI Hub (DET_AIHUB_PLATE_LoRA)** — 770 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 770 | 30583 | 0.766 | 0.434 | 0.405 | 0.238 |
| person | 577 | 1671 | 0.737 | 0.351 | 0.332 | 0.210 |
| car | 770 | 22836 | 0.832 | 0.664 | 0.646 | 0.461 |
| motorcycle | 259 | 426 | 0.723 | 0.423 | 0.371 | 0.146 |
| plate_number | 770 | 5650 | 0.772 | 0.299 | 0.272 | 0.136 |

**KW (DET_KW)** — 327 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 327 | 1528 | 0.943 | 0.558 | 0.547 | 0.386 |
| person | 327 | 1528 | 0.943 | 0.558 | 0.547 | 0.386 |

### 260604_YOLOS26-MASTER_noE2E-reg16
- **브랜치**: `0-moe_loss`
- **설명**: yolo26-master-s에서 end2end=False, reg_max=16으로 변경 — end2end 레짐 효과 분리 ablation
- **모델**: `runs/YOLO26/260604_YOLOS26-MASTER_noE2E-reg16/weights/best.pt`

#### 2차 측정 — 2026-06-08 (epoch ~256)
- **환경**: conda `YOLO26-MASTER`, 브랜치 `0-moe_loss` / GPU: CPU / imgsz=1024, conf=0.25, batch=1, split=test

**COCO (data_qnsfl_new)** — 6701 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 6701 | 34587 | 0.785 | 0.690 | **0.674** | **0.438** |

**AI Hub (DET_AIHUB_PLATE_LoRA)** — 770 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 770 | 30583 | 0.710 | 0.461 | **0.424** | **0.239** |

**KW (DET_KW)** — 327 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 327 | 1528 | 0.888 | 0.513 | **0.506** | **0.306** |

#### 1차 측정 — 2026-06-05 (epoch ~50)
- **환경**: conda `YOLO26-MASTER`, 브랜치 `0-moe_loss` / GPU: CPU / imgsz=1024, conf=0.25, batch=1, split=test

**COCO (data_qnsfl_new)** — 6701 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 6701 | 34587 | 0.800 | 0.623 | 0.620 | 0.405 |

**AI Hub (DET_AIHUB_PLATE_LoRA)** — 770 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 770 | 30583 | 0.691 | 0.351 | 0.324 | 0.193 |

**KW (DET_KW)** — 327 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 327 | 1528 | 0.944 | 0.532 | 0.530 | 0.346 |

---

### 260601_YOLOS26-MASTER_Weight-residual
- **브랜치**: `1-residual`
- **설명**: Weight residual 재실험
- **모델**: `runs/YOLO26/260601_YOLOS26-MASTER_Weight-residual/weights/best.pt`
- **실행일**: 2026-06-04
- **GPU**: CPU
- **설정**: imgsz=1024, conf=0.25, batch=1, split=test

**COCO (data_qnsfl_new)** — 6701 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 6701 | 34587 | 0.795 | 0.638 | 0.644 | 0.424 |
| person | 6219 | 27566 | 0.852 | 0.679 | 0.703 | 0.514 |
| car | 1488 | 5887 | 0.830 | 0.658 | 0.671 | 0.479 |
| motorcycle | 74 | 204 | 0.679 | 0.559 | 0.542 | 0.364 |
| plate_number | 557 | 930 | 0.820 | 0.658 | 0.662 | 0.340 |

**AI Hub (DET_AIHUB_PLATE_LoRA)** — 770 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 770 | 30583 | 0.742 | 0.445 | 0.417 | 0.237 |
| person | 577 | 1671 | 0.775 | 0.339 | 0.315 | 0.201 |
| car | 770 | 22836 | 0.809 | 0.671 | 0.656 | 0.461 |
| motorcycle | 259 | 426 | 0.637 | 0.458 | 0.410 | 0.147 |
| plate_number | 770 | 5650 | 0.747 | 0.313 | 0.287 | 0.139 |

**KW (DET_KW)** — 327 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 327 | 1528 | 0.944 | 0.514 | 0.512 | 0.353 |
| person | 327 | 1528 | 0.944 | 0.514 | 0.512 | 0.353 |

---

## YOLO-MASTER All Test 결과

### 260429_YOLOS-MASTER_BASELINE_HEAD
- **브랜치**: `YOLO-MASTER`
- **환경**: conda `YOLO-MASTER` (Ultralytics 8.3.240, Python 3.11)
- **모델**: `runs/BASELINE/260429_YOLOS-MASTER_BASELINE_HEAD/weights/best.pt`
- **실행일**: 2026-06-04
- **GPU**: CPU
- **설정**: imgsz=1024, conf=0.25, batch=1, split=test

**COCO (data_qnsfl_new)** — 6701 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 6701 | 34587 | 0.780 | 0.669 | 0.772 | 0.537 |
| person | 6219 | 27566 | 0.839 | 0.707 | 0.815 | 0.639 |
| car | 1488 | 5887 | 0.802 | 0.693 | 0.794 | 0.599 |
| motorcycle | 74 | 204 | 0.657 | 0.627 | 0.703 | 0.474 |
| plate_number | 557 | 930 | 0.823 | 0.649 | 0.777 | 0.436 |

**AI Hub (DET_AIHUB_PLATE_LoRA)** — 770 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 770 | 30583 | 0.692 | 0.430 | 0.580 | 0.363 |
| person | 577 | 1671 | 0.735 | 0.356 | 0.565 | 0.383 |
| car | 770 | 22836 | 0.743 | 0.713 | 0.794 | 0.592 |
| motorcycle | 259 | 426 | 0.456 | 0.373 | 0.400 | 0.169 |
| plate_number | 770 | 5650 | 0.833 | 0.277 | 0.560 | 0.307 |

**KW (DET_KW)** — 327 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 327 | 1528 | 0.932 | 0.604 | 0.785 | 0.549 |
| person | 327 | 1528 | 0.932 | 0.604 | 0.785 | 0.549 |

### 260428_YOLOS-MASTER_Backbone3 (YOLO-MASTER S, P6 없음)
- **브랜치**: `YOLO-MASTER`  /  **환경**: conda `YOLO-MASTER` (Ultralytics 8.3.240, Python 3.11)
- **모델**: `runs/BASELINE/260428_YOLOS-MASTER_Backbone3/weights/best.pt`  (9.66M params, NMS, reg_max=16)
- **실행일**: 2026-06-04  /  **GPU**: CPU  /  imgsz=1024, conf=0.25, batch=1, split=test

**COCO (data_qnsfl_new)** — 6701 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 6701 | 34587 | 0.774 | 0.675 | 0.767 | 0.530 |
| person | 6219 | 27566 | 0.823 | 0.714 | 0.813 | 0.635 |
| car | 1488 | 5887 | 0.788 | 0.700 | 0.789 | 0.595 |
| motorcycle | 74 | 204 | 0.672 | 0.623 | 0.692 | 0.452 |
| plate_number | 557 | 930 | 0.813 | 0.665 | 0.776 | 0.439 |

**AI Hub (DET_AIHUB_PLATE_LoRA)** — 770 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 770 | 30583 | 0.678 | 0.444 | 0.579 | 0.355 |
| person | 577 | 1671 | 0.683 | 0.378 | 0.555 | 0.362 |
| car | 770 | 22836 | 0.735 | 0.714 | 0.791 | 0.589 |
| motorcycle | 259 | 426 | 0.500 | 0.401 | 0.427 | 0.161 |
| plate_number | 770 | 5650 | 0.794 | 0.281 | 0.544 | 0.306 |

**KW (DET_KW)** — 327 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 327 | 1528 | 0.964 | 0.583 | 0.782 | 0.546 |
| person | 327 | 1528 | 0.964 | 0.583 | 0.782 | 0.546 |

---

## 표준 YOLO26-s 베이스라인 All Test 결과 (비교 기준)

### 260123_YOLOV26S_DET_BASELINE2 (표준 YOLO26-s, MoE 없음)
- **환경**: conda `Yolov26_env` (Ultralytics 8.4.50, Python 3.12)
- **모델**: `Yolo26/ultralytics/runs/BASELINE/260123_YOLOV26S_DET_BASELINE2/weights/best.pt`
- **아키텍처**: `yolo26s.yaml`, end2end=True, reg_max=1, C3k2×8 (ES_MoE/A2C2f 없음), 9.95M params, 500ep
- **실행일**: 2026-06-04  /  **GPU**: CPU  /  imgsz=1024, conf=0.25, batch=1, split=test

**COCO (data_qnsfl_new)** — 6701 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 6701 | 34587 | 0.793 | 0.620 | 0.738 | 0.513 |
| person | 6219 | 27566 | 0.844 | 0.651 | 0.789 | 0.614 |
| car | 1488 | 5887 | 0.816 | 0.631 | 0.762 | 0.575 |
| motorcycle | 74 | 204 | 0.737 | 0.562 | 0.663 | 0.451 |
| plate_number | 557 | 930 | 0.776 | 0.637 | 0.739 | 0.413 |

**AI Hub (DET_AIHUB_PLATE_LoRA)** — 770 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 770 | 30583 | 0.776 | 0.381 | 0.601 | 0.372 |
| person | 577 | 1671 | 0.722 | 0.303 | 0.538 | 0.362 |
| car | 770 | 22836 | 0.773 | 0.660 | 0.768 | 0.575 |
| motorcycle | 259 | 426 | 0.863 | 0.282 | 0.572 | 0.259 |
| plate_number | 770 | 5650 | 0.747 | 0.279 | 0.525 | 0.291 |

**KW (DET_KW)** — 327 images

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 327 | 1528 | 0.938 | 0.427 | 0.689 | 0.458 |
| person | 327 | 1528 | 0.938 | 0.427 | 0.689 | 0.458 |

> ✅ COCO all mAP50 = **0.738** — 사용자 제보값과 정확히 일치, 분석 검증 완료.

---

## 전체 비교 요약 (all mAP50 @ conf=0.25, test split)

| 모델 | 패러다임 | COCO | AI Hub | KW | params | epochs |
|------|---------|------|--------|-----|--------|--------|
| YOLO-MASTER s-p6 (BASELINE_HEAD) | NMS, reg_max=16, P6 | **0.772** | 0.580 | 0.785 | 10.8M | 329 |
| YOLO-MASTER s (Backbone3) | NMS, reg_max=16 | 0.767 | 0.579 | 0.782 | 9.66M | ~317 |
| **표준 YOLO26-s (BASELINE2)** | **end2end, reg_max=1** | **0.738** | **0.601** | 0.689 | 9.95M | 500 |
| YOLO26-MASTER-s (BASELINE-7) | end2end + ES_MoE/A2C2f/MoE | 0.662 | — | — | 12.5M | 472 |
| **YOLO26-MASTER-s (moeloss-2) ← 최신** | 〃 | **0.664** | **0.409** | **0.607** | 12.5M | **~430** |
| YOLO26-MASTER-s (moeloss-2) 2차 | 〃 | 0.662 | 0.402 | 0.568 | 12.5M | ~244 |
| YOLO26-MASTER-s (moeloss-2) 1차 | 〃 | 0.651 | 0.405 | 0.547 | 12.5M | ~187 |
| YOLO26-MASTER-s (weight-resid) | 〃 | 0.644 | 0.417 | 0.512 | 12.5M | ~172 |
| **noE2E-reg16 2차 ← 최신** | NMS + DFL16 + MoE | **0.674** | **0.424** | 0.506 | 12.9M | **~256** |
| noE2E-reg16 1차 (ablation) | NMS + DFL16 + MoE | 0.620 | 0.324 | 0.530 | 12.9M | ~50 |

**핵심**: ~~표준 YOLO26-s(0.738)가 YOLO26-MASTER(0.65~0.66)보다 0.08 높음 → MASTER 모듈이 성능 저하 원인~~

> ❌ **이 결론 폐기 (2026-06-08).** 위 표의 0.738은 Yolov26_env, 0.66~0.67은 YOLO26-MASTER-env에서 측정된 **cross-build 비교라 무효**. 같은 env에서 재측정하니 표준 YOLO26-s=0.621, MASTER=0.664~0.674로 **MASTER가 오히려 +0.04~0.05 높음**. 문서 맨 끝 "★★★★ 동일env 재비교" 섹션 참조. **이 표의 mAP 열은 서로 다른 env 값이 섞여 있어 직접 비교 불가.**

---

## GPU/CPU 성능 결과 & 정확도-속도 통합 분석 (2026-06-04)

측정: imgsz=1024, conf=0.25 (FPS는 사용자 제보값)

| 모델 | end2end | mAP50(COCO) | GPU FPS | CPU FPS | GFLOPs | params |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| **표준 YOLO26-s** (BASELINE2) | ✓ | 0.738 | **75.39** | **7.5** | 22.8 | 9.95M |
| YOLO-MASTER-s-p6 (BASELINE_HEAD) | ✗ | **0.772** | 47.24 | 3.7 | 29.1 | 10.8M |
| YOLO26-MASTER-s (moeloss-2/BASELINE-7) | ✓ | 0.65~0.66 | 40.7 | 3.52 | 30.1 | 12.5M |

### 핵심 결론: YOLO26-MASTER는 현재 Pareto 열등 (모든 축에서 밀림)
- 표준 YOLO26-s 대비: **느림(40 vs 75) + 부정확(0.65 vs 0.738)**
- YOLO-MASTER 대비: **느림(40 vs 47) + 부정확(0.65 vs 0.772)**
- → 이기는 축이 없음. MASTER 모듈(ES_MoE+A2C2f)이 속도·정확도를 **동시에** 깎음

### 속도 손해 원인 = 정확도 손해 원인 (동일)
- NMS 제거로 아끼는 시간은 작음 (클래스 4개, 수 ms)
- MASTER 모듈이 추가한 연산이 훨씬 큼: 22.8 → 30.1 GFLOPs (+32%)
- **ES_MoE는 추론에서도 dense forward** (yaml에 top_k 미지정 → use_top_k=False → expert 3개 전부 계산, backbone 4곳)
- → NMS-free 속도 이점이 모듈 오버헤드에 완전히 잠식, 오히려 더 느림

### YOLO26 이식 동기와의 충돌
- 사용자가 YOLO26으로 이식한 동기 = end2end(NMS-free)로 **추론 속도 향상** 기대
- 그러나 표준 YOLO26-s는 75 FPS인데 YOLO26-MASTER는 40 FPS → **모듈이 이식 동기 자체를 무효화**
- 표준 YOLO26-s(75 FPS, 0.738)가 "NMS-free + 빠름 + 준수한 정확도"로 원래 기대에 가장 부합

### 전략적 함의
- "fixed" YOLO26-MASTER를 만들어도(end2end×MoE 교정) **정확도는 잘해야 0.738 회복**, 그러나 **여전히 느림**(dense MoE/A2C2f 연산 잔존)
- → 표준 YOLO26-s를 정확도로 못 이기면서 속도만 깎는 구조 → MASTER 모듈의 정당성 약함
- 목표별 권장:
  - **속도 + NMS-free 우선** → 표준 YOLO26-s (75 FPS, 0.738) ★ 현재 최선
  - **정확도 우선 (NMS 허용)** → YOLO-MASTER (0.772, 47 FPS)
  - YOLO26-MASTER as-is → 채택 근거 없음
- 그래도 MASTER 모듈을 살리려면: 모듈이 **속도 손해를 정당화할 만큼 정확도를 올린다는 증거**가 선행 필요 (현재는 반대)

### 경량화 시도 옵션 (속도+정확도 동시 회복 목표)
| 방법 | 속도 | 정확도 | 비용 |
|------|:--:|:--:|------|
| ES_MoE top_k=1 sparse 추론 | ▲▲ | ~유지(routing 균등) | config만, 재학습 0 |
| A2C2f 제거/경량화 (최대 무게) | ▲▲ | ablation 필요 | 재학습 |
| ES_MoE 개수 축소 | ▲ | ablation 필요 | 재학습 |
| 표준 YOLO26-s 채택 | ▲▲▲ | 0.738 | 0 (보유) |

---

## 아키텍처 분석 (2026-06-04)

### YOLO26-MASTER vs YOLO-MASTER 구조 비교

| 항목 | YOLO26-MASTER (`yolo26-master-s`) | YOLO-MASTER (`yolo-master-s-p6`) |
|------|----------------------------------|----------------------------------|
| Detection head | **P3, P4, P5 (3개)** | **P3, P4, P5, P6 (4개)** |
| 최대 stride | 32 | **64** |
| end2end | **True** (NMS 없음) | False (NMS 사용) |
| reg_max (DFL bins) | **1** | **16** |
| Backbone 상단 | SPPF + C2PSA 추가 | 없음 |
| 파라미터 수 | 12,494,260 | 10,833,660 |

---

### P6 효과 분석: 유의미한 차이 없음

YOLO-MASTER 내 실험 비교 (동일 데이터, 동일 조건):

| 모델 | 아키텍처 | val best mAP50 | val best mAP50-95 | best epoch |
|------|---------|----------------|-------------------|-----------|
| 260428_Backbone3 | s (P6 없음) | **0.7666** | 0.5018 | 266 |
| 260429_BASELINE_HEAD | s-p6 (P6 있음) | 0.7652 | **0.5036** | 327 |

→ P6 유무가 val mAP50에 미치는 영향: **-0.0014 (사실상 동일)**
→ P6 추가 시 test mAP50 예상 향상: **+0.005~+0.01 이하**

---

### 성능 격차의 실제 원인 (면밀 조사 후 정정 — 2026-06-04)

#### 관찰된 모순
val mAP50은 두 모델이 거의 동일한데 test mAP50만 크게 차이:

| 모델 | val mAP50 (best, conf=0.001) | test mAP50 (COCO, conf=0.25) |
|------|------------------------------|-------------------------------|
| YOLO-MASTER BASELINE_HEAD | 0.765 | **0.772** |
| YOLO26-MASTER BASELINE-7 (600ep) | 0.768 | **0.662** |

→ 같은 모델·같은 forward 경로인데 val은 대등, test만 낮음. end2end가 모델을 "약화"시킨 거라면 val도 낮아야 함 → **단순 아키텍처 약화론은 틀림.**

#### 코드로 확정한 메커니즘
- `engine/validator.py:135`: val 자동 conf = **0.001**, predict 기본 = 0.25 (`default.yaml`)
- 즉 **학습 중 val은 conf=0.001**, **우리 all test는 conf=0.25**로 측정 → 측정 기준이 달랐음
- 실측 모델 확인: YOLO26-MASTER = `end2end=True, reg_max=1, dfl=Identity`, **one2one(NMS-free) 추론**
- one2one 헤드(TAL topk=1)는 객체당 박스 1개만 할당받아 **confidence가 낮게 캘리브레이션됨**
  - NMS(one2many) 헤드는 다수 후보 경쟁 후 max 생존 → confidence 높음
- 결과: conf=0.25에서 one2one의 "정답이지만 낮은 점수" 박스들이 잘려나감 → test mAP50 붕괴

#### 검증으로 배제한 다른 원인
- **loss/assigner 포팅 버그 없음**: TAL 설정 동일 (topk=10/1, alpha=0.5, beta=6.0), one2one tal_topk=1 동일
- **reg_max=1은 버그 아님**: YOLO26의 의도된 직접 회귀 설계
- **P6 효과 미미**: val 차이 0.0014

#### (중간 결론 — 이후 BASELINE2 증거로 폐기됨)
한때 "end2end one2one 헤드의 낮은 confidence 캘리브레이션이 conf=0.25에서 정탐을 잘라먹는 것"으로 결론냈으나, **아래 표준 YOLO26-s 베이스라인 비교로 이 결론은 틀렸음이 확인됨.**

---

### ★ 최종 확정 원인: MASTER 추가 모듈 (2026-06-04)

표준 YOLO26-s 베이스라인을 동일 데이터로 비교한 결과, end2end/reg_max/conf 가설이 모두 기각됨.

| 모델 | 추가 모듈 | params | epochs | test mAP50 (COCO, conf=0.25) |
|------|----------|--------|--------|------------------------------|
| **표준 YOLO26-s** (260123_YOLOV26S_DET_BASELINE2) | 없음 (C3k2×8) | 9.95M | 500 | **0.738** |
| YOLO26-MASTER-s (BASELINE-7) | ES_MoE×4 + A2C2f×2 + MoE loss | 12.5M | 472 | **0.662** |
| YOLO26-MASTER-s (moeloss-2) | 〃 | 12.5M | ~187 | 0.651 |
| YOLO-MASTER-s-p6 (참고, NMS) | reg_max=16, NMS | 10.8M | 329 | 0.772 |

**표준 YOLO26-s도 `end2end=True, reg_max=1`로 완전히 동일한 패러다임인데 conf=0.25에서 0.738.**
→ end2end·reg_max=1·conf=0.25·one2one confidence 가설 **전부 기각**.

#### 진짜 원인
- 표준 YOLO26(0.738) → YOLO26-MASTER(0.662), **동일 데이터·동일 패러다임·비슷한 학습량(500 vs 472ep)**
- 유일한 차이 = **MASTER 고유 추가물: ES_MoE×4 + A2C2f×2 + MoE aux loss**
- 파라미터 **+2.5M 증가**했는데 성능 **-0.076 감소** → **MASTER 모듈이 현재 순수하게 해를 끼침**

#### 비교 기준 정정
- YOLO26-MASTER의 비교 대상은 YOLO-MASTER(0.772)가 아니라 **표준 YOLO26(0.738)**
- 현재 MASTER는 표준 YOLO26보다도 **0.076 낮음** → 이식의 정당성이 아직 미확보

#### 다음 조사: ablation (범인 분리)
0.738 → 0.662로 끌어내린 주범 분리 필요:
1. ES_MoE만 추가 → ?
2. A2C2f만 추가 → ?
3. MoE aux loss on/off → ?

> ⚠️ 공정성: MASTER 모델은 학습 부족 가능성(moeloss-2 ~187/600ep). 단 472ep BASELINE-7도 0.662라 학습량만으로 0.738 도달은 어려워 보임.

---

### ★★ 재반전: 범인은 "모듈"이 아니라 "모듈 × end2end 상호작용" (2026-06-04)

사용자 지적: YOLO-MASTER도 **동일한 ES_MoE(3 experts, k 기본) + A2C2f**를 쓰는데 0.767. 즉 모듈 자체는 문제가 아님.

#### Controlled 진단: ES_MoE routing 분포 비교 (실측, 12장 평균)
| 모델 | layer 3/6/9/12 routing (균등=0.33) | test mAP50 |
|------|-----------------------------------|------------|
| YOLO26-MASTER (moeloss-2) | ≈ [0.33, 0.33, 0.33] (균등) | 0.651 |
| YOLO-MASTER (Backbone3) | ≈ [0.33, 0.33, 0.33] (균등) | 0.767 |

→ **둘 다 routing 균등(전문화 없음).** "routing 붕괴가 범인"이라는 가설도 **기각**. 같은 모듈·같은 routing 상태인데 한쪽만 잘 됨.

#### 2×2 정리 (COCO mAP50 @ conf=0.25)
| | NMS (reg_max=16) | end2end (reg_max=1) |
|--|--|--|
| 모듈 없음 | (미측정) | 표준 YOLO26-s = **0.738** |
| +ES_MoE+A2C2f | YOLO-MASTER = **0.767** | YOLO26-MASTER = **0.65** |

**같은 모듈이 NMS에선 멀쩡(0.767), end2end에선 해(0.738→0.65).**
→ 범인 = **「MASTER 모듈 × end2end 패러다임」 상호작용** (모듈 단독도, routing도 아님)

#### 유력 메커니즘
- end2end 추론 = **one2one(TAL topk=1, 희소 감독)** 헤드 + **MoE aux loss가 E2ELoss에 얽혀 경쟁**
- NMS = one2many(topk=10, 풍부한 감독) → backbone(MoE/attention)이 강하게 학습됨
- 희소한 end2end 감독 + aux loss 경쟁 + 무거운 추가 파라미터 → 모듈이 제대로 학습 안 됨
- 사용자가 고쳐온 **E2ELoss MoE 통합 버그**가 정확히 이 지점

#### 전략적 함의 (중요)
- **end2end를 빼면 = reg_max=16 + NMS = 사실상 YOLO-MASTER와 동일.** YOLO26-MASTER의 정체성(존재 이유)이 곧 **NMS-free end2end**임.
- 따라서 "end2end 제거"는 해법이 아니라 "YOLO-MASTER로 회귀"임 (YOLO-MASTER가 이미 0.767이므로 중복).
- 올바른 목표: **end2end를 유지한 채** 0.65 → 표준 YOLO26 수준(0.738+)으로 회복.
  - 표준 YOLO26-s(end2end, 모듈 없음)가 0.738을 내므로 end2end의 천장은 최소 0.738.
  - 회복 성공 시 → NMS-free를 거의 동등 정확도로 확보 = 진짜 가치. (현재 0.65는 NMS-free 대가로 너무 큼)
- **선결 질문: NMS-free(추론 속도/배포 단순화)가 필수 요구인가?**
  - 아니면 → YOLO-MASTER(0.767)가 정답, end2end와 싸울 이유 없음
  - 맞으면 → 아래 계획으로 end2end×MoE 상호작용 교정

#### 다음 조사 계획 (end2end 필수 가정)
- **Phase 0 (무학습)**: E2ELoss vs v8DetectionLoss의 MoE aux 처리 코드 diff + 학습 후반 moe_loss 크기 모니터링
- **Phase 1 (핵심 단일 실험)**: YOLO26-MASTER `moe=0`(aux off, 모듈 유지) 학습 → 0.738로 회복하면 **aux loss가 범인 확정**
- **Phase 2**: 결과 따라 E2ELoss aux 적용 방식을 YOLO-MASTER 방식으로 교정 / 또는 A2C2f·ES_MoE 분리 ablation (200ep 단축)

---

### 개선 방향

| 개선 방향 | 예상 효과 | end2end 유지 | 비고 |
|-----------|---------|:---:|------|
| P6 추가 | +0.01 이하 | ✅ | 효과 미미 |
| **① 온도 스케일링 (confidence 보정)** | 미검증 | ✅ | 재학습 0, 가장 먼저 시도 |
| **② cls loss gain ↑** | 미검증 | ✅ | 간단한 재학습 |
| **③ one2many→one2one 지식 증류** | 미검증 | ✅ | YOLOv10 방식, 정공법 |
| end2end 비활성 + reg_max=16 | +0.05~+0.10 | ❌ | 사실상 YOLO-MASTER화 |

---

### [향후 작업] 온도 스케일링 실험 (보류 — 2026-06-04)

**목표**: conf=0.25 고정 배포는 그대로 두고, 모델 출력 logit에 `/T` 보정을 넣어 낮게 깔린 정탐 confidence를 0.25 위로 끌어올림.

**원리**: `confidence = sigmoid(logit / T)`. T>1이면 낮은 점수를 0.5 쪽으로 끌어올려 잘리던 정탐 복구. 단 FP도 함께 올라오므로 최적 T 필요. (단일 스칼라 온도 스케일링은 수학적으로 "유효 conf 임계값 변경"과 동일 → conf 스윕이 곧 최적 T 탐색)

**절차**:
1. moeloss-2 모델, **val split**에서 conf 스윕 (0.25 / 0.15 / 0.10 / 0.05) → mAP50 + Precision + Recall 기록
2. mAP50 최대 & Precision 허용 가능한 지점 X\* 선택 → 대응 T\* 계산
3. `Detect._inference`에서 `(scores / T).sigmoid()` 적용 후 **test split conf=0.25 고정** 재측정
4. ⚠️ 튜닝은 반드시 val로, 검증은 test로 (test로 튜닝 시 오버피팅)

**상태**: 보류. Backbone3 결과 확인 후 진행 예정.

---

## ★★★ 대반전: 격차는 모델이 아니라 "버전 × conf=0.25 측정 아티팩트" (2026-06-08)

end2end 제거(noE2E-reg16)로도 성능이 안 바뀐 이유를 끝까지 추적한 결과, **그동안의 모든 격차가 측정 방식 때문이었음**이 실측으로 확인됨.

### 진단 경로 (모두 동일 100장 COCO_CAR_LABELING test, 버전독립 자체 코드)

1. **cls head logit 분포 비교** (`/tmp/logit_probe.py`)
   - noE2E-MASTER sigmoid mean 0.821 vs YOLO-MASTER 0.810 → **동일**. "MASTER가 confidence 낮게 깐다" 가설 기각.
2. **recall 진단** (`/tmp/recall_diag.py`, conf=0.25)
   | 지표 | noE2E-MASTER | YOLO-MASTER |
   |---|---|---|
   | recall@0.25 | 67.0% | 67.5% |
   | conf-cut | 26.7% | 26.1% |
   | hard miss | 6.3% | 6.4% |
   | 평균 IoU | 0.815 | 0.813 |
   → 모든 지표 **통계적으로 동일**.
3. **버전독립 AP50 직접 계산** (`/tmp/ap_diag.py`, full PR curve)
   | 클래스 | noE2E-MASTER (8.4.50) | YOLO-MASTER (8.3.240) |
   |---|---|---|
   | person | 0.757 | 0.761 |
   | car | 0.825 | 0.813 |
   | plate | 0.825 | 0.789 |
   | **mAP50** | **0.803** | **0.788** |
   → **같은 잣대로 재면 동등 (MASTER가 미세하게 앞섬). 0.09 격차 소멸.**

### 진짜 원인

| 측정 방식 | noE2E-MASTER | YOLO-MASTER |
|---|---|---|
| 내 코드 full curve (conf=0.001) | **0.803** | 0.788 |
| 기록된 all-test (conf=0.25) | 0.674 | 0.767 |
| conf=0.25 restriction 하락폭 | **−0.13** ❗ | −0.02 |

**ultralytics 8.4.50은 conf=0.25 평가 시 mAP를 8.3.240보다 훨씬 가혹하게 깎음.** conf=0.25로 예측을 미리 필터링한 뒤 PR curve를 계산하는 방식이 두 버전에서 다름. → 모델 실력차가 아니라 **버전 × conf=0.25 설정의 상호작용**.

### 이것이 설명하는 것들
- "end2end 꺼도 안 바뀜" → 고칠 실제 격차가 없었음
- "val(0.001) 0.76인데 test(0.25) 0.67 폭락" → 8.4.50의 conf=0.25 평가 탓 (모델 약화 아님)
- logit/recall/IoU 진단이 전부 동일 → 두 모델이 실제로 동등
- lr×3 하드코딩(layer 23) 어긋남 → 실재 코드 결함이나 confidence 정상이라 성능 영향 없음

### ⚠️ 기존 결론 재검토 필요
- *"MASTER 모듈이 성능 깎음"*, *"end2end×MoE 상호작용이 범인"*, *"표준 YOLO26보다 0.076 낮음"* — 모두 **버전이 다른 모델 간 conf=0.25 비교**에 기반 → 신뢰 불가 가능성.
- **버전 다른 모델 간 conf=0.25 mAP 비교 금지.** 같은 버전 내 비교(표준 YOLO26 vs MASTER, 둘 다 8.4.50)는 유효 → 별도 재검증 필요.

### 미확정/다음 단계
- 위는 car셋 100장·3클래스(person/car/plate, motorcycle 없음). person/motorcycle 셋·더 많은 이미지로 확장 확인 필요.
- 결정적 확인 실험: **YOLO-MASTER 모델(8.3.240에서 0.77)을 8.4.50 환경에서 conf=0.25로 평가 → ~0.66으로 떨어지면 버전 아티팩트 확정.**

### ✅ 결정적 확인 실험 완료 (2026-06-08)

**YOLO-MASTER 260429_BASELINE_HEAD 모델을 YOLO26-MASTER 환경(ultralytics 8.4.50)에서 직접 평가** (GPU 0, conf=0.25, COCO test, 동일 weight):

| 지표 | 8.3.240 (원래 기록) | 8.4.50 (재평가) | 차이 |
|---|---|---|---|
| Precision | 0.780 | 0.780 | **0.000** |
| Recall | 0.669 | 0.669 | **0.000** |
| **mAP50** | **0.772** | **0.673** | **−0.099** |
| mAP50-95 | 0.537 | 0.438 | −0.099 |

→ **P·R은 소수점까지 동일한데 mAP만 0.099 떨어짐.** 같은 예측인데 mAP 계산식이 버전마다 다르다는 결정적 증거. YOLO-MASTER도 8.4.50에선 0.673 = MASTER 모델들(0.66~0.67)과 동일 수준.

**최종 확정**: 0.77 vs 0.66 격차는 **100% ultralytics 버전 아티팩트**. 모델/모듈/end2end 무관. 이전 결론(MASTER 모듈이 성능 저하 원인, end2end×MoE 상호작용 등)은 **버전 다른 모델 간 비교라 전부 무효**. 공정 비교하려면 모든 모델을 동일 버전(또는 버전독립 AP 코드)으로 재평가해야 함.

## ★★★★ 동일env 재비교 — MASTER 모듈은 성능을 "올린다" (2026-06-08)

cross-version 아티팩트 확정 후, 모든 모델을 **동일 환경/동일 방법**으로 재비교.

### 방법 1: 같은 env(YOLO26-MASTER, ultralytics 8.4.50 MASTER build) native val, conf=0.25, COCO test 전체(6701장)

| 모델 | 모듈 | mAP50 | vs 표준 |
|---|---|---|---|
| 표준 YOLO26-s (BASELINE2, MASTER-env 재평가) | 없음 | **0.621** | 기준 |
| moeloss-2 (end2end) | ES_MoE+A2C2f+MoE | **0.664** | **+0.043** |
| noE2E-reg16 (NMS) | 〃 | **0.674** | **+0.053** |

### 방법 2: 버전독립 자체 AP (full PR curve, COCO car셋 100장)

| 모델 | mAP50 |
|---|---|
| 표준 YOLO26-s | **0.754** |
| YOLO-MASTER (Backbone3) | 0.788 |
| YOLO-MASTER (BASELINE_HEAD) | 0.784 |
| noE2E-MASTER | **0.803** |

→ **두 독립 방법 모두 일치: MASTER 모듈이 표준 YOLO26 대비 +0.03~0.05 향상.**

### 추가 확정: "버전 문자열 동일"도 불충분
표준 YOLO26-s, 동일 weight, conf=0.25, 둘 다 ultralytics **8.4.50**:
- Yolov26_env(stock 8.4.50): P=0.793 R=0.620 **mAP50=0.738**
- YOLO26-MASTER env(8.4.50 MASTER build): P=0.792 R=0.621 **mAP50=0.621**
→ P·R 동일, mAP만 0.117 차이. **env(빌드)가 다르면 같은 버전이라도 mAP 계산이 다름.** 비교는 정확히 같은 env에서만 유효.

### 최종 결론 (이전 모든 "MASTER가 나쁘다" 결론 폐기)
- MASTER 모듈(ES_MoE+A2C2f)은 성능을 **저하시키지 않고 향상**시킴 (+0.03~0.05, 두 방법 일치).
- 0.66~0.67이 낮아 보였던 건 8.4.50 MASTER-env의 conf=0.25 mAP 계산이 가혹했던 것일 뿐, 표준 YOLO26도 같은 env에선 0.621.
- end2end vs NMS(noE2E)는 동일env에서 0.664 vs 0.674로 **거의 차이 없음** (end2end 무해).
- 단, MASTER 모델들은 미완 학습(moeloss-2 ep430/600, noE2E ep256/600)인데도 이미 표준을 앞섬.

## ★★★★★ 근본 원인 규명 + 수정 — `compute_ap` 함수 (2026-06-08)

"버전/env마다 mAP가 다른" 현상의 진짜 원인을 코드에서 찾음. **ES-MoE와 무관.**

### 범인: `ultralytics/utils/metrics.py` 의 `compute_ap` (AP 면적 계산)

MASTER fork가 stock 대비 sentinel(끝점)을 수정해둠:
```python
# stock (표준)
mrec = np.concatenate(([0.0], recall, [1.0]))
mpre = np.concatenate(([1.0], precision, [0.0]))
# MASTER fork (수정본) — (recall[-1], 0.0) 추가
mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
```
효과: 모델이 도달한 **최대 recall 지점에서 precision을 즉시 0으로 끊음** → 표준이 주던 "도달 못한 recall 구간 extrapolation 크레딧"이 사라짐.

### 수치 증명 (동일 PR 곡선, 두 공식)
| 상황 | STOCK AP | MASTER AP | 차이 |
|---|---|---|---|
| conf=0.25 (recall 0.67에서 잘림) | 0.752 | 0.599 | **−0.15** |
| conf=0.001 (recall 0.95까지) | 0.877 | 0.851 | −0.03 |
→ 곡선이 잘릴수록(conf↑) 차이 폭발. full curve면 거의 무차이.

### 실측 확인 (표준 YOLO26-s, 동일 MASTER env)
| conf | mAP50 |
|---|---|
| 0.25 (잘림) | 0.621 |
| 0.001 (full) | 0.715 |
| stock env, conf=0.25 | 0.738 |
→ MASTER env도 full curve면 0.715로 stock 0.738에 근접. 격차는 오직 "conf=0.25 잘림 × MASTER 엄격 공식".

### 저자가 왜 바꿨나 (추정)
같은 fork diff에 **per-image metric 기능(`image_metrics`, DDP gather)**이 추가돼 있음. 이미지 1장은 GT가 적어 PR 곡선이 거의 항상 잘리는데, 표준 extrapolation은 per-image AP를 비정상적으로 부풀림. → **per-image AP를 안정적으로 만들려고 extrapolation을 제거**한 것으로 보임. 부작용으로 글로벌 mAP까지 바뀌어 stock과 비교 불가가 됨.

### 조치
- `compute_ap`를 **표준 공식으로 되돌림** (metrics.py:731, 2026-06-08). 이제 MASTER env mAP도 stock/YOLO-MASTER와 같은 자로 측정됨.
- ⚠️ **수정 후 all test 재측정 필요**: conf=0.25 mAP가 ~0.1 상승할 것으로 예상 (moeloss-2 0.664 → ~0.75+, noE2E 0.674 → ~0.76+). 재측정해야 정확.

### ✅ 수정 검증 완료 (2026-06-08)
표준 YOLO26-s, MASTER env, conf=0.25, COCO test:
| | mAP50 | mAP50-95 |
|---|---|---|
| compute_ap 수정 전 | 0.621 | 0.403 |
| **compute_ap 수정 후** | **0.738** | **0.513** |
| stock Yolov26_env (참고) | 0.738 | 0.513 |
→ 수정 후 stock과 소수점까지 일치. `compute_ap`가 단일 원인 확정. **이제 모든 MASTER 모델 all test 재측정하면 stock/YOLO-MASTER와 직접 비교 가능.**

## ★★★★★★ compute_ap 수정 후 재측정 (2026-06-08) — MASTER가 최상위권

수정된(표준) compute_ap로 all test 재측정. env: YOLO26-MASTER(8.4.50, fixed), conf=0.25, GPU0.

| 모델 | 데이터셋 | 수정 전 | 수정 후 | 상승 |
|---|---|---|---|---|
| moeloss-2 (ep430) | COCO | 0.664 | **0.774** | +0.110 |
| | AIHub | 0.409 | **0.618** | +0.209 |
| | KW | 0.607 | **0.781** | +0.174 |
| noE2E-reg16 (ep256) | COCO | 0.674 | **0.775** | +0.101 |
| | AIHub | 0.424 | **0.614** | +0.190 |
| | KW | 0.506 | **0.720** | +0.214 |

P·R은 불변, mAP만 상승 (AIHub/KW는 recall truncation이 커서 상승폭 더 큼).

### 동일 자(표준 compute_ap) COCO mAP50 최종 줄세우기
| 모델 | COCO mAP50 | 비고 |
|---|---|---|
| 표준 YOLO26-s | 0.738 | MASTER-env(fixed)=stock 일치 |
| YOLO-MASTER (BASELINE_HEAD) | 0.772 | YOLO-MASTER env(표준 compute_ap) |
| **moeloss-2** | **0.774** | ep430/600 (미완) |
| **noE2E-reg16** | **0.775** | ep256/600 (미완) |

→ **MASTER 모델이 표준 YOLO26을 앞서고 YOLO-MASTER와 동급 이상.** 미완 학습 상태인데도 최상위. 기존 "MASTER 모듈이 성능 저하" 결론은 compute_ap 아티팩트에 의한 완전한 착시였음이 최종 확정.

## 📌 compute_ap 코드 비교 + 원인 분석 (2026-06-08)

위치: `ultralytics/utils/metrics.py` → `compute_ap()` → "Append sentinel values".

### 코드 비교

**YOLO26-MASTER 원래 (버그):**
```python
mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
```
**YOLO26-MASTER 수정 후 = 표준 = YOLO12(YOLO-MASTER, 8.3.240) = 순정 YOLO26:**
```python
mrec = np.concatenate(([0.0], recall, [1.0]))
mpre = np.concatenate(([1.0], precision, [0.0]))
```

| 코드베이스 | 베이스 | ultralytics | 표준? |
|---|---|---|---|
| YOLO26-MASTER (원래) | YOLO26 | 8.4.50 | ❌ 비표준 (sentinel 추가) |
| YOLO26-MASTER (수정 후) | YOLO26 | 8.4.50 | ✅ |
| YOLO-MASTER | YOLO12 | 8.3.240 | ✅ |
| 순정 YOLO26 | YOLO26 | 8.4.x | ✅ |
> 출처: 이 비표준 수정은 (git 기록상으론 codebase import에 묻혀 있으나) 이식 과정에서 들어간 것.

### 어떤 현상이 발생하나
`(recall[-1], 0.0)` 점을 끝에 끼워넣어, **모델이 도달한 최대 recall 지점에서 precision을 즉시 0으로 수직 강하**시킴.
- 표준: 최대 recall(R_max) → recall=1.0 까지 precision을 비스듬히 0으로 내림 = "extrapolation 크레딧"(삼각형 면적) 부여
- 수정: R_max에서 바로 0 → 그 크레딧 삭제
- 예) recall=[0.5,0.7], precision=[0.9,0.8] → 표준은 [0.7,1.0] 구간에 ≈0.12 면적 크레딧, 수정은 0.
- recall이 잘릴수록(conf↑, 어려운 데이터) AP 하락폭 커짐. full curve면 미미.

### 어떤 환경일 때 쓰나 (적절한 용도)
- **per-image / 소표본 AP**: 이미지 1장은 GT가 적어 PR곡선이 거의 항상 짧게 잘림. 표준 extrapolation은 per-image AP를 과대·불안정하게 만듦 → 수직 강하 방식이 더 안정적·보수적. (`if len(recall) else 1.0` 가드도 "예측 0개" 같은 per-image 엣지케이스 처리용)
- **도달 못한 recall에 점수 안 주는 엄격 평가**.
- 단 **비표준** → COCO/VOC/ultralytics 관례와 불일치 → 글로벌 mAP를 stock/타 모델과 비교 불가.

### 왜 들어갔나 (추정 경위)
같은 fork에 **per-image metric 기능(`image_metrics`, `_gather_image_metrics`)**이 함께 이식돼 있음. per-image AP 안정화를 위해 compute_ap를 보수적으로 바꾼 것이, **공유 함수라 글로벌 mAP까지 같이 바뀌어** 전체 평가가 깎이고 비교 불가가 됨. 의도(per-image 안정화) 자체는 합리적이나, 글로벌 지표에 새어나간 게 문제.

## ⚠️⚠️ 최종 정정 — compute_ap는 공식 8.4.50 코드였음 (2026-06-08, 순정 pip 검증)

앞 섹션들의 *"fork가/Claude가 compute_ap를 수정했다", "per-image metric 때문"* 설명은 **틀렸음.** 순정 pip `ultralytics==8.4.50`을 받아 직접 대조해 정정함.

### 검증된 사실 (버전별 compute_ap)
| 참조 | 버전 | compute_ap | mAP 성향 |
|---|---|---|---|
| Yolo26 순정 repo / Yolov26_env | **8.4.6** | 표준 `[recall,[1.0]]` | 정상(0.738) |
| YOLO-MASTER env | **8.3.240** | 표준 | 정상(0.772) |
| 순정 pip ultralytics | **8.4.50** | **`recall[-1]` (strict)** | conf=0.25에서 ~0.1 낮음 |
| YOLO26-MASTER (8.4.50 기반) | 8.4.50 | `recall[-1]` = **공식 그대로** | 〃 |

### 결론
- `recall[-1]` strict compute_ap는 **공식 ultralytics가 8.4.6→8.4.50에서 바꾼 것.** 이식/포팅 버그 아님. 전체 diff 감사 결과 인프라는 충실히 이식됨(무해한 상수 1줄 제외).
- 격차의 진짜 원인 = **strict(8.4.50) vs 표준(8.4.6/8.3.240) AP 공식이 모델마다 섞여 비교된 것** = cross-version 측정 아티팩트. (최초 진단이 옳았음)
- **기록 오기 정정**: 표준 YOLO26-s(BASELINE2) 평가 환경은 "Ultralytics 8.4.50"이 아니라 **8.4.6**임 (Yolov26_env 실측 import 8.4.6 확인).

### 결정 (사용자 선택: 1번)
- YOLO26-MASTER의 compute_ap를 **표준 공식으로 유지(revert 유지)** → stock YOLO26(8.4.6)·YOLO-MASTER(8.3.240) baseline과 conf=0.25 직접 비교 가능.
- ⚠️ 이는 **공식 8.4.50과는 의도적으로 다른 상태**임 (metrics.py 주석에 명시). 비교 일관성을 위한 선택.
- 따라서 "compute_ap 수정 후 재측정" 섹션의 MASTER 수치(moeloss-2 COCO 0.774, noE2E 0.775 등)는 **표준 AP 기준으로 baseline과 비교 가능한 유효 수치**임.

## 🔍 인프라 파일 감사 — 이식 충실성 검증 (2026-06-08)

순정 pip `ultralytics==8.4.50`을 받아 YOLO26-MASTER 패키지와 python 라인 단위 대조. 모델 외 인프라가 임의 변경됐는지 확인.

| 파일 | 변경 내용 | 판정 |
|---|---|---|
| `nn/modules/moe/` (신규) | ES_MoE 등 MoE 모듈 | ✅ 모델 |
| `nn/modules/__init__.py` | MoE import / `__all__` 등록만 | ✅ 모델 |
| `nn/tasks.py` | MoE import + 파싱 등록 + `A2C2fMoE` legacy 처리 (31줄 전부) | ✅ 모델 |
| `utils/loss.py` | MoE aux loss 통합 (moe 슬롯/cosine decay/E2ELoss moe, git 커밋과 일치) | ✅ 모델 |
| `models/yolo/detect/train.py` | `moe_loss` 이름 1줄 | ✅ 모델 |
| `cfg/default.yaml` | `moe: 0.15` (MoE gain) | ✅ 모델 |
| `utils/metrics.py` | compute_ap revert | ⚠️ 의도적 변경(주석 명시, 비교용) |
| `utils/torch_utils.py` | `TORCH_2_12` 상수 | ⚠️ 죽은 코드(정의만·미사용, 무해) |

**결론: 이식은 충실함.** 순정 8.4.50 대비 모든 인프라 변경이 (a) 정당한 MoE 통합, (b) 의도적 compute_ap revert, (c) 미사용 상수 1줄뿐. 결과를 오염시키는 숨은 로직 변경 없음. compute_ap는 (재확인) 이식이 아니라 공식 8.4.50 코드였음.
