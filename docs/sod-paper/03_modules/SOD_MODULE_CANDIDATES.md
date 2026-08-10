# SOD 기존기법 차용 후보 조사 (2026-07-09) — SAQ 대안·상보

목적: SAQ 실패 대비 + `feedback_module-borrow-simple`(밑바닥 말고 기존 차용+허점+간단수정). 우리 제약:
P2 있음, small(16-32) 집중, **fusion/refine/loss(WIoU)/assignment(NWD) 다 negative**, quality-rescore +0.12(precision 여지),
**recall 벽(AR 0.32, probe로 candidate 늘려도 안 됨)**, tiling/해상도 거부.

## 축 지도 (우리가 판 곳 vs 안 판 곳)

| 축 | 대표기법 | 우리 이력 |
|---|---|---|
| feature fusion/neck | FPN·PANet·SGI·GSA·AFFG | ❌ negative |
| box refine/offset | Cascade·P2Refine | ❌ negative |
| loss reweight | WIoU·SA-WIoU | ❌ small 약함 |
| assignment | NWD-align(probe)·ATSS | ❌ probe negative(AR 불변) |
| **quality/ranking** | GFL·VFNet·IoU-Net | 🔄 SAQ 진행중(rescore +0.12) |
| **downsample 정보보존** | **SPD-Conv** | ⬜ **안 해봄 ★** |
| context/RF | RFB·LKA·dilated | ⬜ 거의 안 봄 |
| NMS/post | Soft-NMS·score-voting | ⬜ 안 해봄 |

---

## 1. SPD-Conv ★★★ (최우선 대안 — 안 해본 축, recall 공략)

- **논문**: "No More Strided Convolutions or Pooling" (ECML-PKDD 2022, arXiv:2208.03641).
- **허점 지적**: strided conv/pooling 이 **fine-grained 정보를 잃음** → 저해상·small object 에서 치명(AlexNet 1/8 축소 시 정확도 30%↓).
- **기법(수식)**: downsample 을 stride 대신 **space-to-depth**로.
  ```
  X (S,S,C) --SPD scale2--> 4개 서브맵 f00,f10,f01,f11 concat --> (S/2, S/2, 4C) --non-strided conv--> (S/2,S/2,C2)
  ```
  공간정보를 버리지 않고 채널로 접어 보존 → 같은 downsample 배율, 정보손실 0.
- **성능**: YOLOv5 의 stride-2 conv 7곳(backbone 5+neck 2) 교체 → **COCO AP_S +11~13%**(s: 21.1→23.5).
- **장점**: 레이어 교체(간단)·검증됨·**우리 안 해본 축**·small 직접. **단점**: 채널4배로 연산/메모리↑, P2 고해상과 일부 중복 가능.
- **★ 인사이트**: 우리 **recall 벽(AR 0.32)**이 "small feature 정보부족"이면, probe(candidate 늘리기)·SAQ(순위)로는 못 뚫는다.
  SPD-Conv 는 **정보손실이라는 근본 원인**을 downsample 단계에서 막음 → recall 벽 공략 가능. SAQ(precision)와 **상보**.
- **수정안(차용+간단수정)**: YOLO26 backbone/neck 의 **stride-2 `Conv` 를 `SPDConv` 로 교체**(특히 P2 로 내려가는 초기 downsample).
  yaml 레이어 1종 교체. "허점=downsample 정보손실, 수정=레이어 교체, 결과=small feature 질↑→recall↑".

## 2. GFL / VarifocalNet (VFNet) ★★ (= SAQ 계열, 진행중)

- **기법**: 예측 **품질(IoU)을 분류 score 에 융합**해 순위 정렬. VFL: `cls target = IoU`(positive), focal(negative).
  ```
  VFL(p,q) = -q(q log p + (1-q)log(1-p))   if q>0 ;   -α p^γ log(1-p)   if q=0
  ```
- **장점**: 검증된 강baseline, `VarifocalLoss` 우리 코드에 이미 있음(재사용). **단점**: 전 스케일 균일 → small IoU 포화 허점.
- **인사이트/수정**: = SAQ. 차용(VFL) + "small IoU 포화" 허점을 small loss 가중/NWD 로 간단수정. rescore +0.12 = precision 여지 실측.

## 3. RFLA (Gaussian Receptive Field Label Assignment) ★ (assignment — 리스크)

- **논문**: ECCV 2022 (arXiv:2208.08738). AI-TOD +4.0 AP.
- **기법**: 앵커의 **effective receptive field 를 Gaussian** 으로, GT 도 Gaussian → **RFD(KLD 기반) 로 label assign**
  (IoU 임계·center sampling 이 large 편향이라 tiny 홀대하는 허점 지적). Hierarchical Label Assignment.
- **NWD 와 차이**: NWD=box Gaussian 의 W2 거리(우리 probe 가 씀). RFLA=**receptive field** Gaussian + KLD + 계층할당.
- **장점**: tiny 특화 검증. **단점**: **assignment 축 = 우리 probe(NWD-align) negative(AR 불변)와 같은 계열** → ROI 낮을 위험.
- **인사이트/수정**: probe 실패로 assignment 축 회의적이나, RFLA 는 receptive-field prior 가 달라 **재시도 여지는 있음**. 우선순위 낮음(SPD 후).

## 4. RFB (Receptive Field Block) / LKA ★ (context)

- **기법**: multi-branch **dilated conv** 로 다양한 receptive field → small 주변 맥락 보강. (RFB: Inception+dilation)
- **장점**: 경량 context. **단점**: 우리 fusion(SGI/GSA) negative 와 유사 계열 리스크. neck 삽입.
- **인사이트/수정**: small 은 맥락 의존 큼. neck 특정 위치(P3→P2)에 RFB 1개 삽입해 context만. fusion 강제 아닌 "receptive field 확장"으로 차별.

## 5. Soft-NMS / Score-voting ★ (post, 매우 간단)

- **기법**: NMS 에서 겹치는 박스를 삭제 대신 **score 감쇠**(`s_i ← s_i·f(IoU)`). dense small 에 유리.
- **장점**: **학습 불필요·inference 한 줄**·즉시. **단점**: recall 벽이면 효과 제한, SAQ(rescore)와 목적 중복.
- **인사이트/수정**: dense small(VisDrone)에서 NMS 과억제 완화. SAQ 실패 시 **공짜 시도**.

---

## 종합 인사이트 & 추천

- 우리는 **quality/ranking(precision) 축**(SAQ)을 파는 중 — rescore +0.12 로 여지 확실하나 **recall 벽**은 못 넘음.
- **recall 벽의 근본이 feature 정보부족이면, 안 해본 `SPD-Conv`(downsample 정보보존)가 가장 상보적·유망**. 간단(레이어 교체)·검증(+11~13%).
- **추천 우선순위**: ① SPD-Conv(recall/feature, 안 해본 축) ② SAQ/VFL(precision, 진행중) — **둘은 상보** → 최종 SPD+SAQ 조합 가능.
  ③ Soft-NMS(공짜) ④ RFB(context) ⑤ RFLA(assignment, 리스크).
- 다음 액션 후보: SAQ smoke/A1 병행 중, **SPD-Conv 를 YOLO26 stride-2 Conv 교체로 별도 브랜치 실험**(naive-P2 대비 small AP·AR).

## SPD-Conv 구현·실험 상태 (2026-07-09) — 코덱스 인수인계

- **구현**: `SPDConv(Focus)` 별칭 클래스(conv.py). Focus 가 이미 SPD-Conv 와 동일 연산
  (`conv(cat([x[::2,::2],x[1::2,::2],x[::2,1::2],x[1::2,1::2]],1))` = space-to-depth + stride-1 conv) → **새 로직 0줄**.
  등록: tasks.py(base_modules), nn/modules/__init__. 브랜치 **`4-spdconv`** 커밋(0fcdc869, 2d43069d).
- **yaml 2종** (naive-P2 와 stride-2 교체만 차이):
  - `yolo26-p2-spd.yaml` — **backbone stride-2 Conv 5개**만 SPDConv. build OK(SPDConv 5, anchor 34000, **15.6M**).
  - `yolo26-p2-spd-full.yaml` — **backbone5 + neck3 = 8개**(논문 충실). build OK(SPDConv 8, anchor 34000, **17.9M**).
- **논문 대비**: 논문(YOLOv5-SPD)=backbone5+neck2. 우리 backbone-only 는 neck 미교체(불완전), full 은 neck3 까지(우리 P2 neck 은
  down-path 3개). **완전 동일은 full 이 근접**(단 YOLOv5→YOLO26·P2 추가라 1:1 아님).
- **본학습 (코덱스: backbone-only 먼저)**: 사용자 CLI. ablation **① P2 → ② +SPD-backbone(`yolo26-p2-spd`) → ③ +SPD-full**.
  **첫 실행 = backbone-only**(가설=downsample 정보손실 직접검증, 해석 순수). full 은 backbone 이 AR/AP 올렸을 때만 확장
  (full 먼저면 "SPD 좋다"vs"neck+용량 증가" 반박). 대조군 naive-P2(P2v2) @maxDets1500.
- **판정 핵심 = small AR(recall)이 움직이는가** (AP 아님). AR↑ → recall 축 생존 → SPD+SAQ 결합 스토리 성립. AR 무변 → SPD negative.
  보조: small AP50/AP75 · mAP50-95 유지 · SPD 논문 COCO +11~13% 회수율.
- **필수 로그 (코덱스)**: small AR·AP50·matched/unmatched count·class별 recall·P2/P3 level 분포.
  ⚠ **우리 end2end(NMS-free)** 라 "NMS 전/후 recall" 대신 **one2many(assigner topk10 후보) recall vs one2one(최종) recall** 로 분리.
- **negative 여도 정직 기록** → `../04_results/SOD_RESULT_AP_SIZE.md`.

## 출처
- SPD-Conv arXiv:2208.03641 · RFLA arXiv:2208.08738 · GFL/VFNet(기지식) · SOD survey arXiv:2503.20516
