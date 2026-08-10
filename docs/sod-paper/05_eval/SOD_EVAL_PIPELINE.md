# 크기별 AP (AP_S / APT) 평가 파이프라인

SOD 논문용으로, ultralytics 내장 val과 별개로 **pycocotools 기반 크기별 AP**를 일관되게 뽑는 재사용 스크립트.

## 위치
- `eval/yolo2coco.py` — YOLO detect split → COCO GT json 변환기
- `eval/eval_size_ap.py` — 예측 수집 + COCOeval(custom areaRng) → 크기별 AP 출력
- 결과 저장: `eval/results/` (gt_*.json, pred_*.json, apsize_*.json)
- ultralytics 패키지 밖이라 개인 repo push에는 포함되지 않음.

## 설계 결정 (확정)
- **conf = 0.001** (COCO 표준; PR 곡선 전체 적분). 배포용 conf=0.25 metric과는 **별개 평가**.
- **max_det = 500** (VisDrone 등 고밀도 데이터 기준; COCO 기본 100은 recall 깎음).
- **iou(NMS) = 0.7**, imgsz = 640 (baseline 일치).
- **eval lib = pycocotools** (논문 표준). ultralytics 내장값과는 ~0.01 차이나므로 **논문 표는 pycocotools로 통일**.

## 크기 구간 (√area px 기준, area=px²로 빈닝)
AI-TOD식 세분 + COCO식 동시 출력:
- vt(2-8), **tiny/APT(8-16)**, small(16-32), medium(32+)
- cocoS(<32), cocoM(32-96), cocoL(>96)

APT(tiny) = 변길이 8~16px 객체의 AP@[.5:.95] = SOD 헤드라인 지표.

## 사용법
```bash
# (cwd = SOD-PAPER 루트, conda env SOD-PAPER)
python eval/eval_size_ap.py \
  --weights ultralytics/runs/BASELINE/260610_Y26S_BASELINE_VISDRONE/weights/best.pt \
  --data dataset/VisDrone/cleaning/data.yaml --split test \
  --imgsz 640 --device 5 --max-det 500 --conf 0.001 \
  --tag Y26S_VISDRONE
```
GT json은 split별로 캐시됨(`gt_<dataset>_<split>.json`).

## 검증 (2026-06-12)
같은 test split·설정에서 내장 val과 대조 → 파이프라인 정상 확인.

| 지표 | pycocotools(본 파이프라인) | ultralytics val |
|---|---|---|
| mAP50 | 0.2918 | 0.3068 |
| mAP50-95 | 0.1646 | 0.1748 |

차이 ~0.01 = 알려진 AP 공식 차이(내장 vs COCO 표준). 버그 아님.

## 주의
- cwd가 SOD-PAPER 루트면 로컬 `ultralytics/` 폴더가 패키지를 가려 `import ultralytics` 실패 가능. 스크립트 파일 실행(`python eval/...`)은 sys.path[0]가 eval/라 정상 동작. 인터프리터 인라인 실행 시엔 중립 디렉토리에서 절대경로 사용.
- tiled DOTA는 area가 타일 픽셀 기준 → 크기 구간 해석 시 유의.

## AI-TOD식 초소형 평가 + maxDets (2026-07-03)
- 크기 구간(vt/tiny/small/medium)은 **표준 COCOeval 엔진 + AI-TOD식 areaRng 주입** = AI-TOD eval 과 동일 정의. 별도 도구 불필요. **DOTA eval**(회전박스+VOC mAP)은 초소형 특화가 아니라 **부적합**.
- **maxDets**: `eval_size_ap.py` 기본 500. 초소형 밀집(VisDrone/AI-TOD)은 AI-TOD 표준 **1500** 권장 — 500이면 밀집 프레임 recall 이 잘림(특히 recall 상한 실험에서 중요).

## 상한/통계 도구 (skill: `sod-research-method`)
- `oracle_ceiling.py` — 캐시 예측(`pred_*.json`)으로 loc(위치완벽)/det(recall완벽) 상한을 **학습 없이** 측정. 방향 판정용.
- `analyze_result.py` — `apsize_*.json` 을 **노이즈 바닥 대비 유의성**으로 판정. 대조군은 baseline 이 아니라 **직전 단계**로.
