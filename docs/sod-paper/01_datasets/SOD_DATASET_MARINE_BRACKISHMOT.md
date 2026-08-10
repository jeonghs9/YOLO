# BrackishMOT (Marine) 데이터셋 — split 과정·클래스 문제 (⏸ 보류)

작성 2026-07-09. 상태: **⏸ 보류** — 클래스 누락 문제로 새 marine 데이터셋을 찾을 때까지 중단.
경로: `/home/hsjeong/workspace/SOD-PAPER/dataset/Marine/BrackishMOT/`.

## 1. 원본 데이터

- **BrackishMOT** (수중 실촬영, MOT 추적 포맷). `BrackishMOT/BrackishMOT/` 하위에 공식 분할:
  - **train: 78 sequences**, **test: 20 sequences**. (synthetic `brackishMOTSynth`·`background_videos` 는 미사용)
  - 각 sequence = `gt.txt`(frame,id,x,y,w,h,conf,class,vis) + `seqinfo.ini` + `img1/`.
- 6 classes: `0 fish · 1 small_fish · 2 crab · 3 shrimp · 4 jellyfish · 5 starfish`.

## 2. split 과정 (내가 한 것)

1. **MOT → YOLO 변환**: 각 sequence `gt.txt` 를 프레임별 YOLO 라벨(`class cx cy w h` 정규화)로.
2. **train/val**: 원본 **train 78 seq 를 sequence 단위 8:2** → train 62 / val 16 (프레임 아닌 **sequence-level**,
   같은 영상이 train·val 에 섞이지 않게).
3. **test**: 원본 **test 20 seq 그대로** 유지.
- 파일명: `brackishMOT-<seq>_<frame>.jpg`. data.yaml: `split/data.yaml`.

## 3. split 데이터 수

| split | images(frames) | sequences |
|---|---|---|
| train | 10,243 | 62 (원본 train 80%) |
| val | 2,888 | 16 (원본 train 20%) |
| test | 3,487 | 20 (원본 test 전체) |
| 계 | 16,618 | 98 |

## 4. ★ 클래스 분포 & 누락 문제 (보류 사유)

| 클래스 | train | val | test |
|---|---|---|---|
| fish(0) | 584 | **0 ❌** | 251 |
| small_fish(1) | 5,757 | 1,100 | 2,417 |
| **crab(2)** | 326 | **0 ❌** | **0 ❌** |
| shrimp(3) | 2,827 | 325 | 823 |
| jellyfish(4) | 8,204 | 4,281 | 8,161 |
| starfish(5) | 206 | 64 | **0 ❌** |

- **crab**: train 만 존재, val·test 0 → 학습되나 **평가 불가**. (노션 결과의 crab per-class AP 는 GT=0 이라 0/nan = 무의미)
- **원인 2갈래**:
  - **test 의 crab·starfish 없음 = 원본 공식 test(20 seq)에 애초에 없음** → test 유지했으니 불가피.
  - **val 의 fish·crab 없음 = 내 8:2 sequence 분할 불균형** → 소수 클래스가 소수 seq 에만 몰려 random split 이 train 에
    전부 배정. **재split(stratified)으로 개선 가능**(단 test 는 못 살림).
- **근본**: 극심한 클래스 불균형(jellyfish 12,485 vs crab 326, starfish 270) + sequence-level split.

## 5. 실험 기록

- baseline 학습 3개(Y26S Baseline / P2 Baseline 등) 완료 → **노션 업로드**. (crab/starfish AP 는 test GT 없어 무의미)

## 6. 재개 시 할 일 (보류 해제하면)

1. **stratified 재split**: 소수 클래스(crab·fish·starfish)가 train/val 에 모두 들어가게.
2. **test 한계 인정**: crab·starfish 는 원본 test 에 없어 이 데이터셋으론 평가 불가 → 소수 클래스 **병합/제외** 검토.
3. 또는 **다른 marine/수중 데이터셋** 물색(클래스 균형·공식 test 확인). ← 사용자가 이 방향으로 탐색 예정.
