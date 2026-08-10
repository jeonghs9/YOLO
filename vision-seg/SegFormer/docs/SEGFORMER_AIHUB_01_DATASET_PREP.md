# SEGFORMER_AIHUB_01_DATASET_PREP

- 최종 갱신: 2026-08-06
- 대응 Python 파일: `UTIL/aihub_make_masks.py`, `UTIL/aihub_make_split.py`
- 원본 데이터: `/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE/` (AI Hub 차선 데이터, 미변경)
- 산출물: `/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE_SEG/`
- 선행 작업: SkyScapes 항공영상 (`SEGFORMER_SKYSCAPES_01~03`) — 환경·백본·학습 스크립트를 그대로 재사용한다
- 상태: **데이터 준비 완료**, config 작성 및 학습 미착수

---

## 1. 왜 SegFormer(A안)인가

AI Hub 라벨은 차선의 **중심선 폴리라인**이다. 이 형식을 다루는 길은 두 갈래였다.

| 방향 | 대표 모델 | 장점 | 단점 |
|---|---|---|---|
| **A. SegFormer + 띠 라벨** | 기존 코드 | 환경·백본·학습/평가/시각화 전부 재사용. 즉시 시작 | 차선 개별 구분 불가(후처리 필요) |
| B. 차선 검출 전용 | UFLD, LaneATT, CLRNet, PolyLaneNet | 데이터 형식에 정확히 부합, 차선별 분리, 고속 | 새 코드베이스, `lane_type` 속성 헤드를 직접 추가해야 함 |

**A 로 진행한다.** 검증된 파이프라인이 이미 있어 몇 시간이면 기준선을 얻는다.
차선별 분리가 실제로 필요하다고 판단되면 그때 B 로 넘어간다.

---

## 2. 원본 데이터 실측

### 2.1 규모

| 서브셋 | 장수 | 조건 | 해상도 | 클립 |
|---|---:|---|---|---:|
| `c_1920_1200_daylight_validation_2` | 15,000 (52%) | 주간 | 1920×1200 | 1,195 |
| `c_1920_1200_daylight_validation_3` | 5,246 (18%) | 주간 | 1920×1200 | 428 |
| `c_1920_1200_night_validation_1` | 7,555 (26%) | 야간 | 1920×1200 | 576 |
| `1920_1080_night_validation_d_1` | 1,177 (4%) | 야간 | **1920×1080** | 166 |
| **합계** | **28,978** | | | **2,365** |

이미지 15.4GB. 이미지/라벨 stem 불일치 **0**, JSON 파싱 실패 **0**.

SkyScapes 는 라벨 있는 원본이 10장뿐이었다. 이 데이터는 **2,900 배** 규모다.

### 2.2 라벨 형식

```json
{"image": {"file_name": "11971063.jpg", "image_size": [1200, 1920]},
 "annotations": [
   {"class": "traffic_lane", "category": "polyline",
    "attributes": [{"code":"lane_color","value":"white"},
                   {"code":"lane_type","value":"dotted"}],
    "data": [{"x":114,"y":865}, {"x":234,"y":850}]}]}
```

`image_size` 는 `[height, width]` 순서다.

| class | 개수 | category |
|---|---:|---|
| `traffic_lane` | 32,515 (daylight_3 기준) | polyline |
| `crosswalk` | 1,791 | polygon |
| `stop_line` | 1,172 | polyline |

| 속성 | 값 분포 (4개 서브셋 전수) |
|---|---|
| `lane_type` | **solid / dotted** — dotted 비율 40.8~47.6% (서브셋 간 일관) |
| `lane_color` | white / yellow / blue |

**`lane_type` 이 우리가 필요한 실선/점선 구분을 그대로 담고 있다.** SkyScapes 와 클래스
체계가 그대로 호환된다.

> 4개 서브셋을 **전수** 스캔한 결과 형식이 완전히 일관됐다. 초기에 800장 표본으로 봤을
> 때 야간 서브셋의 dotted 비율이 4.5% 로 이상해 보였으나, 전수로는 40.8% 였다.
> **표본 편향이었다.** 파일명 정렬 순서가 촬영 순서라 앞부분만 보면 특정 구간에 몰린다.

---

## 3. 핵심 쟁점 — 폴리라인에는 두께가 없다

semantic segmentation 은 픽셀마다 정답이 필요한데 폴리라인은 선이라 면적이 없다.
일정 폭으로 칠해 면으로 만들어야 한다(rasterize). CULane 등이 쓰는 표준 방식이다.

### 3.1 점선의 빈칸도 라벨에 포함된다 (중요)

**폴리라인이 점선의 끊긴 부분을 관통한다.** 실측으로 확인했다. 폴리라인을 따라가며
밝기를 샘플링해 변동계수를 재면:

| lane_type | 표본 | 선을 따라간 밝기 변동계수 (중앙값) |
|---|---:|---:|
| solid | 200 | **0.165** (균일 = 계속 페인트 위) |
| dotted | 159 | **0.320** (요동 = 페인트/아스팔트 교대) |

점선이 실선의 **약 2 배**다. 확대 시각화에서도 빈 아스팔트 구간에 밴드가 그대로
이어지는 것이 보인다.

**따라서 두 데이터셋은 라벨의 의미가 다르다.**

| | SkyScapes | AI Hub |
|---|---|---|
| 라벨이 가리키는 것 | **도색된 페인트 픽셀** | **차선이 지나가는 경로(띠)** |
| 점선의 빈칸 | background | **dashed** |
| 예측 결과물 | 페인트 모양 | 연속된 띠 |

귀결 3가지:

1. **두 데이터의 IoU 를 직접 비교하면 안 된다.** 정답의 정의가 다르다.
2. **모델을 서로 재활용할 수 없다.** 시점도 라벨 의미도 다르다.
3. 실선/점선 판정 목적에는 **AI Hub 방식이 오히려 유리하다.** SkyScapes 에서 dashed
   정밀도가 41~43% 에 머문 원인이 "얇은 점선 조각을 픽셀 단위로 맞히기 어렵다"였는데,
   띠 라벨은 그 어려움이 없다. 모델은 "띠 안쪽 무늬가 끊겼는가"로 클래스를 판단한다.

### 3.2 원근 보정이 필수다 (실측 보정)

차량 전방 시점이라 같은 차선도 화면 위(멀리)는 가늘고 아래(가까이)는 굵다.
실선 구간에서 선에 **수직으로 밝기 프로파일**을 떠 실제 도색 폭을 측정했다.
(중심에서 이어지는 밝은 구간만 세어 옆 차선 오염을 막았다.)

| y | daylight_1200 | night_1200 | night_1080 |
|---:|---:|---:|---:|
| 575 | 9.25px | — | 3.25px |
| 625 | 5.75px | 4.25px | 3.50px |
| 725 | 6.00px | 6.25px | 7.50px |
| 825 | 8.00px | 11.50px | 15.25px |
| 925 | 13.00px | 19.50px | 24.62px |
| 1025 | 26.25px | 26.50px | 29.25px |
| 1175 | 32.25px | 33.25px | — |

**차선 라벨의 y 중앙값이 690 인데 그 지점 실제 폭은 6px 이다.** CULane 식 고정 폭
(12px 등)으로 칠했다면 대다수 라벨이 실제의 2 배로 부풀려졌을 것이다.

카메라 장착 위치가 달라 서브셋마다 곡선이 다르므로 **보정표도 서브셋별**로 둔다
(`aihub_make_masks.py` 의 `WIDTH_TABLE`). 표 사이는 선형보간, 밖은 [3, 36]px 로 클램프.

측정 스크립트는 `--measure` 로 재실행할 수 있다.

---

## 4. 1단계 — 마스크 변환 (`aihub_make_masks.py`)

### 4.1 클래스

| ID | 클래스 | 원본 |
|---:|---|---|
| 0 | `background` | 그 외 전부 (**crosswalk, stop_line 포함**) |
| 1 | `solid` | `traffic_lane` + `lane_type=solid` |
| 2 | `dashed` | `traffic_lane` + `lane_type=dotted` |

crosswalk/stop_line 을 ignore 가 아니라 **background 로 흡수**한다. ignore 로 두면 그
자리에 차선이 무작위로 예측된다 (SkyScapes 01단계 3절과 같은 판단).

`lane_color` 는 이번엔 쓰지 않는다. 다만 **한국 도로에서 황색선은 중앙선**이라 의미가
크므로 클래스를 늘릴 여지로 남겨둔다.

### 4.2 구현 요점

- 폴리라인을 4px 간격으로 잘게 나눠 구간마다 그 지점의 y 에 맞는 폭으로 그린다
- **dotted 를 먼저, solid 를 나중에** 그린다. 교차 지점에서 solid 가 살아남는다
  (실선 침범 판정이 하류 목적이라 solid 를 보수적으로 지킨다)
- 모르는 값을 만나면 조용히 넘어가지 않고 기록한다

### 4.3 결과 (실측)

| 클래스 | 픽셀 비율 |
|---|---:|
| background | 98.383% |
| solid | **0.972%** |
| dashed | **0.646%** |

- 전경 **1.62%** — SkyScapes(0.45%)보다 **3.6 배** 많다. 클래스 불균형이 크게 완화된다
- 전경 없는 장: 75개 (28,978 중)
- 마스크 28,978개, 0.31GB

### 4.4 실행

```bash
PYTHONNOUSERSITE=1 python3 \
  /home/hsjeong/workspace/vision-seg/SegFormer/scripts/aihub_make_masks.py \
  --workers 24

# 시험 (서브셋당 40장만)
  ... aihub_make_masks.py --out /home/hsjeong/tmp/aihub/seg_test --limit 40
```

산출: `CAR_LANE_SEG/masks/{서브셋}/{stem}.png`, `CAR_LANE_SEG/manifest.csv`

`manifest.csv` 컬럼: `subset, stem, clip, H, W, solid_px, dashed_px, img, mask`

---

## 5. 2단계 — 분할 (`aihub_make_split.py`)

### 5.1 클립 — 무엇이고 무엇이 아닌가

**클립은 파일도 폴더도 아니다.** `manifest.csv` 의 `clip` 컬럼 하나다.
파일명이 촬영 순서 정수라, **정렬 후 앞뒤 번호 차이가 1 이 아닌 지점**이 경계다.

```text
clip 2: 3014631 ~ 3014647   (17프레임)   <- 번호가 이어짐
clip 3: 3014649 ~ 3014652   ( 4프레임)   <- 3014648 이 없어 새 클립
clip 4: 3014658 ~ 3014658   ( 1프레임)
```

클립이 한 일은 **"분할할 때 이 프레임들을 쪼개지 말고 한 봉지째 보내라"** 뿐이다.

| 오해 | 실제 |
|---|---|
| 클립 단위로 학습한다 | 학습은 **프레임 한 장씩** 한다 |
| 클립마다 대표 1장을 뽑았다 | **한 장도 버리지 않았다** (28,978장 그대로) |
| 중복 제거를 위해 도입했다 | **누수 차단**을 위해 도입했다 |

### 5.2 왜 필요했나 — 연속 프레임 누수

이 데이터는 주행 영상에서 뽑은 연속 프레임이다. 간격 1 인 쌍이 전체의 92% 다.

| 비교 | 화소 평균 절대차 |
|---|---:|
| 인접 프레임 쌍 | **20.8** |
| 무작위 쌍 | 62.2 |

**인접 프레임이 무작위 쌍보다 3.0 배 유사**하다. 0.03초 차이 나는 사실상 같은 장면을
train 과 test 에 나눠 넣으면 실력이 아니라 암기를 측정하게 된다.

### 5.3 클립만으로는 부족하다 — 서브셋 홀드아웃

클립 분할은 "완전히 같은 프레임"은 막지만 그 이상은 못 막는다.

| 비교 | ID 거리 중앙값 | 300 이내 |
|---|---:|---:|
| **서브셋 내부** 클립 간 | **118 ~ 453** | **47~60%** |
| **서브셋 간** | **7,768 ~ 43,260** | 1.7~11.4% |

**같은 폴더 안의 클립들은 서로 몇 초 거리다.** clip 71 과 clip 72 는 같은 도로를 몇 초
간격으로 지난 장면일 수 있다. 클립 단위로 나눠도 "같은 도로·같은 조명·같은 날씨"가
train 과 test 에 나뉜다. 반면 다른 폴더끼리는 20~350 배 멀어 사실상 다른 주행이다.

서브셋 간 **동일 ID 0개, 인접 ID 0~3개**로 누수 없이 나눌 수 있음을 확인했다.

→ **test 는 폴더째 홀드아웃**한다. val 이 클립 단위(다소 낙관적)인 것은 괜찮다.
val 은 체크포인트 선택용이고 정직한 숫자는 test 가 낸다.

### 5.4 확정된 분할

```text
test        = c_1920_1200_daylight_validation_3 + 1920_1080_night_validation_d_1  (폴더째)
train / val = c_1920_1200_daylight_validation_2 + c_1920_1200_night_validation_1  (클립 8:2)
```

| split | 클립 | 장수 | solid% | dashed% | 전경% |
|---|---:|---:|---:|---:|---:|
| train | 1,419 | **18,094** | 0.992 | 0.638 | 1.630 |
| val | 352 | **4,461** | 0.909 | 0.684 | 1.593 |
| test | 594 | **6,423** | 0.959 | 0.640 | 1.599 |

서브셋 구성:

| split | 구성 |
|---|---|
| train | daylight_2 12,020 (66%) + night_1 6,074 (34%) |
| val | daylight_2 2,980 (67%) + night_1 1,481 (33%) |
| test | daylight_3 5,246 (82%) + night_d_1 1,177 (18%) |

**검증 통과**: 클립 누수 0, stem 불일치 0, 라벨 고유값 `{0,1,2}`, dtype uint8,
전경 비율 편차 0.04%p 이내.

**트레이드오프**: test 에 `night_d_1`(1920×1080, 다른 카메라)이 포함되는데 학습에는
1080 데이터가 없다. 이 부분 점수는 "차선 학습 실패"가 아니라 "처음 보는 카메라"
때문일 수 있다. 해석 시 감안한다.

### 5.5 층화 배정 방식

클립을 전경 픽셀량 내림차순 정렬한 뒤 비율 패턴으로 라운드로빈 배정한다.

```python
pat = ['train']*8 + ['val']*2
lst.sort(key=lambda t: (-t[1], t[0]))   # 전경량 내림차순, 동점은 키순 -> 결정론적
for i, (key, _) in enumerate(lst):
    assign[key] = pat[i % len(pat)]
```

무작위 배정과 비교 (daylight_3, 클립당 평균 전경량):

| 방식 | train | test | 차이 |
|---|---:|---:|---:|
| 무작위 seed 0 | 488,151 | 453,665 | 7.1% |
| 무작위 seed 1 | 470,372 | 483,578 | 2.8% |
| 무작위 seed 2 | 457,017 | 526,076 | 15.1% |
| 무작위 seed 4 | 395,766 | 585,358 | **47.9%** |
| **층화** | 476,035 | 445,771 | **6.4%** |

층화가 항상 무작위보다 낫지는 않다(seed 1 이 더 좋았다). 가치는 평균이 아니라
**최악을 없애는 것**이다. seed 4 같은 사고가 원천적으로 없고 재현 가능하다.

### 5.6 출력 레이아웃

```text
CAR_LANE_SEG/split_holdout/
├── images/{train,val,test}/{서브셋}__{stem}.jpg
└── labels/{train,val,test}/{서브셋}__{stem}.png
```

mmseg `CustomDataset` 이 `img.replace(img_suffix, seg_map_suffix)` 로 단순 문자열
치환을 하므로 이미지와 라벨의 파일명이 확장자만 빼고 같아야 한다
(`mmseg/datasets/custom.py:146-151`). 서브셋이 달라도 stem 이 겹칠 수 있어
`{서브셋}__{stem}` 으로 접두어를 붙였다.

**심볼릭 링크가 아니라 실제 복사**다 (`--copy`).

### 5.7 실행

```bash
# 확정 분할 (서브셋 홀드아웃)
PYTHONNOUSERSITE=1 python3 \
  /home/hsjeong/workspace/vision-seg/SegFormer/scripts/aihub_make_split.py \
  --test-subsets c_1920_1200_daylight_validation_3 1920_1080_night_validation_d_1 \
  --train-val-ratio 8 2 \
  --out /home/hsjeong/workspace/dataset/AIHUB/CAR_LANE_SEG/split_holdout \
  --copy --overwrite

# 계획만 확인
  ... --dry-run

# 구 방식(전 서브셋을 클립 단위 6:2:2) — 참고용, 권장하지 않음
  ... aihub_make_split.py --ratio 6 2 2 --out .../split --copy --overwrite
```

---

## 6. 남은 한계

### 6.1 프레임 중복 — 데이터가 겉보기보다 12배 작다

한 장도 버리지 않았으므로 중복은 그대로다.

| split | 장수 | 실제 장면(클립) | 중복 배수 |
|---|---:|---:|---:|
| train | 18,094 | 1,419 | **12.8배** |
| val | 4,461 | 352 | 12.7배 |
| test | 6,423 | 594 | 10.8배 |

**학습에는 큰 해가 없다.** 증강(RandomCrop, Flip, PhotoMetricDistortion)이 걸려 같은
장면도 매번 다르게 보이므로 약한 증강 효과가 된다. 다만 1 에폭의 실질 가치가 1/12 이라
iter 수 산정 시 감안한다.

**평가는 왜곡된다.** 클립 길이가 1~77 프레임으로 제각각인데 mmseg 는 전체 픽셀을
누적해 IoU 를 내므로, 긴 클립이 짧은 클립보다 훨씬 큰 발언권을 갖는다.

| test 클립 상위 | 전경 픽셀 점유 |
|---|---:|
| 1.1% | 6.1% |
| 4.9% | 23.2% |
| 10.0% | **40.0%** |
| 25.0% | **69.3%** |

**대응안(미적용)**: `--eval-per-clip N` 을 추가해 val/test 만 클립당 N장(예: 3장)씩
뽑으면 각 장면이 동등한 발언권을 갖고 평가 시간도 4배 이상 단축된다. train 은 그대로 둔다.

### 6.2 test 의 1080 서브셋

5.4 절 참조. 학습에 1920×1080 데이터가 없어 해당 부분 점수는 별개 요인이 섞인다.

### 6.3 폴리라인 근사

2점짜리 폴리라인이 53%(17,797/33,687)다. 직선 구간은 문제없지만 곡선 차선은 근사가
거칠어진다.

---

## 7. 진행 상태

| 단계 | 상태 |
|---|---|
| 데이터 구조/형식 분석 | **완료** (2절) |
| 폴리라인 → 마스크 변환 | **완료** (`aihub_make_masks.py`, 4절) |
| 원근 보정 폭 실측 | **완료** (3.2절) |
| 서브셋 홀드아웃 분할 | **완료** (`aihub_make_split.py`, 5절) |
| 클래스 가중치 계산 | 미착수 |
| 데이터셋 클래스 등록 | 미착수 — `img_suffix='.jpg'` 라 SkyScapes 클래스를 그대로 못 쓴다 |
| config 작성 | 미착수 |
| 학습 | 미착수 |

---

## 8. 다음 단계에서 결정할 것

### 8.1 데이터셋 클래스

`SkyScapesLaneDataset` 은 `img_suffix='.png'` 라 그대로 쓸 수 없다. 클래스명과
팔레트는 동일하므로 `AIHubLaneDataset` 을 새로 만들되 `img_suffix='.jpg'` 만 바꾼다.

```python
@DATASETS.register_module()
class AIHubLaneDataset(CustomDataset):
    CLASSES = ('background', 'solid', 'dashed')
    PALETTE = [[0, 0, 0], [0, 0, 255], [255, 0, 0]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                         reduce_zero_label=False, **kwargs)
```

### 8.2 입력 처리 (미정)

차선 라벨은 **y=526 아래에만** 존재한다(1200 기준). 상단 44% 는 하늘·건물이라
그대로 넣으면 연산의 절반을 버린다.

| 방식 | 장단점 |
|---|---|
| **상단 크롭 + RandomCrop** (권장) | 낭비 없고 다양성 확보. y<500 을 잘라 1920×700 으로 만든 뒤 crop |
| 전체 리사이즈 | 구현 단순, 추론 시 전처리 불필요 |
| RandomCrop 만 | 하늘만 담긴 crop 이 다수 발생 |

해상도가 두 종류(1200/1080)라 크롭 라인을 해상도별로 둬야 한다.

### 8.3 재사용 가능한 자산

| 항목 | 재사용 |
|---|---|
| `segformer_cu111` 환경 | **그대로** |
| `mit_b0~b5.pth` 사전학습 | **그대로** |
| 학습/평가 명령 (`dist_train.sh`, `dist_test.sh`) | **그대로** |
| `skyscapes_visualize_test.py` | 경로 인자만 바꿔 사용 가능 |
| 데이터셋 클래스 | 신규 (8.1절) |
| config | B0/B2 복사 후 수정 |

---

## 9. B0 학습 결과 (2026-08-06)

### 9.1 설정

| 항목 | 값 |
|---|---|
| config | `local_configs/segformer/B0/segformer.b0.1024x640.aihub.40k.py` |
| 데이터셋 클래스 | `AIHubLaneDataset` (`mmseg/datasets/aihub_lane.py`, `img_suffix='.jpg'`) |
| 입력 | 1920x1200 -> 1024x640 (0.533배). 1920x1080 은 1024x576 후 하단 Pad(255=ignore) |
| 하늘 크롭 | **하지 않음** (추론 전처리를 단순하게 유지) |
| 유효 배치 | 4 x GPU 4 = **16** |
| 학습량 | **40,000 iter** ≈ 35 에폭 (장면 기준 451회) |
| lr | 1.2e-4 AdamW poly, warmup 3,000 |
| 클래스 가중치 | `[1.0, 9.9596, 12.4159]` (sqrt-inverse, train 전수 실측) |
| 소요 | 약 3.5시간 (0.31 s/iter, GPU당 7.4GB) |
| work_dir | `work_dirs/b0_aihub_40k` (체크포인트 20개, 0.90GB) |

클래스 가중치 근거 — train 18,094장 전수:

| 클래스 | 픽셀 | 비율 | 손실 기여도 (배경:전경) |
|---|---:|---:|---|
| background | 41,009,124,016 | 98.3702% | 가중치 없음 60.4:1 |
| solid | 413,427,210 | 0.9917% | **sqrt-inverse 5.5:1 (채택)** |
| dashed | 266,024,774 | 0.6381% | median-freq 0.5:1 |

SkyScapes 는 같은 방식에서 10.2:1 이었다. 전경이 3.6배 많아 불균형이 절반으로 완화됐다.

### 9.2 val 추이 (4,461장, 2,000 iter 간격 20회)

| iter | solid IoU | dashed IoU | 전경평균 |
|---:|---:|---:|---:|
| 2,000 | 39.56 | 29.62 | 34.59 |
| 8,000 | 46.50 | 38.97 | 42.73 |
| 12,000 | 47.96 | 41.40 | 44.68 |
| 20,000 | 49.03 | 43.74 | 46.39 |
| 28,000 | 49.24 | 45.15 | 47.20 |
| 34,000 | 51.39 | 44.87 | 48.13 |
| **40,000** | **51.08** | **45.70** | **48.39** |

**끝까지 우상향했고 최종 지점이 최고점**이다 (최고 solid 51.57@32k, 최고 dashed 45.90@38k).
SkyScapes B0/B2 가 각각 70%/45% 지점에서 최고를 찍고 횡보한 것과 대조적이다.
**아직 수렴하지 않았다는 신호**로, iter 를 늘리면 더 오를 여지가 있다.

### 9.3 test 결과 (6,423장 전수)

`dist_test` 와 별도 계산한 혼동행렬이 정확히 일치했다 (solid 52.64 / dashed 43.09).

| | solid IoU | dashed IoU | **전경평균** |
|---|---:|---:|---:|
| **전체** | 52.64 | 43.09 | **47.86** |
| 주간 (daylight_3) | 53.52 | 44.53 | **49.03** |
| 야간 (night_d_1, 1920x1080 미학습 카메라) | 46.77 | 35.91 | **41.34** |

**val 48.39 -> test 47.86 으로 차이가 0.5 뿐이다.** 서브셋 홀드아웃(다른 주행)으로 나눴는데도
성능이 유지된 것이라 일반화가 잘 됐다는 강한 근거다. 시각화 표본 100장 기준 train 52.59 /
test 47.23 으로 과적합도 크지 않다.

야간이 7.7점 낮은 것은 야간이라서가 아니라 **1920x1080 미학습 카메라**이기 때문으로 보인다
(학습에는 1200 해상도만 있었다).

### 9.4 혼동행렬 (test 전체)

| 정답 \ 예측 | background | solid | dashed | GT 합계 |
|---|---:|---:|---:|---:|
| background | 14,108,028,980 | 94,762,895 | 92,340,395 | 14,295,132,270 |
| solid | 10,355,177 | 125,382,705 | 3,576,445 | 139,314,327 |
| dashed | 7,462,794 | 4,116,238 | 81,385,571 | 92,964,603 |

| | 재현율 | 정밀도 |
|---|---|---|
| solid | bg 7.43% / **solid 90.00%** / dashed 2.57% | bg 42.26% / **solid 55.91%** / dashed 1.84% |
| dashed | bg 8.03% / solid 4.43% / **dashed 87.54%** | bg 52.08% / solid 2.02% / **dashed 45.90%** |

**SkyScapes 의 최대 병목이었던 solid 미검출이 완전히 해소됐다.**

| | solid 재현율 | dashed 재현율 |
|---|---:|---:|
| SkyScapes B0 (완주) | 52.3% | 86.1% |
| SkyScapes B2 (완주) | 65.3% | 88.2% |
| **AIHub B0 (완주)** | **90.0%** | **87.5%** |

원인은 (a) 데이터 74배, (b) 띠 라벨(빈칸 포함 연속 띠라 맞히기 쉬운 형태) 두 가지로 보인다.
**단, SkyScapes 와 IoU 를 직접 비교하면 안 된다** — 정답의 정의가 다르다(페인트 vs 경로).
위 재현율 비교는 "무엇이 병목이었나"를 보는 용도로만 유효하다.

**남은 병목은 정밀도다.** 예측한 solid 의 42.3%, dashed 의 52.1% 가 실제로는 background 다.
클래스 간 오분류(solid<->dashed)는 2~4% 로 미미하므로, 문제는 여전히 **전경 대 배경**이다.

### 9.5 시각화

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 \
/home/hsjeong/miniconda3/envs/segformer_cu111/bin/python \
  UTIL/aihub_visualize.py \
  --checkpoint work_dirs/b0_aihub_40k/iter_40000.pth \
  --out /home/hsjeong/tmp/aihub/vis_iter40000 --device cuda:0
```

출력: `vis_iter40000/{train,val,test}/` 각 100장, `원본 | GT | 예측` 3분할(2312x516).
`vis_iter2000/` 과 **같은 seed 라 파일명이 동일**해 학습 전후를 직접 비교할 수 있다.

| split | iter_2000 | iter_40000 |
|---|---:|---:|
| train | 36.99 | 52.59 |
| val | 33.89 | 47.64 |
| test | 33.26 | 47.23 |

**주의**: 시각화의 GT 패널은 원본 폴리라인이 아니라 **4절에서 변환한 마스크**다.
폴리라인은 면적이 0 이라 그대로는 학습도 시각화도 되지 않는다.

### 9.6 CCTV 영상 적용 — 실패

`/home/hsjeong/workspace/dataset/TEST_VIDEO/19700101_000157_T.mp4` (4K, 5분)에 적용.
출력 `b0_aihub_predonly.mp4` (2,960프레임, 0.204 s/frame).

**결과가 나쁘다.** 프레임당 평균 solid 73,523px / **dashed 2,847px** 로 점선을 거의 못 찾고,
실제 차선이 아닌 곳에 얼룩이 흩어진다 (150초·250초 지점 dashed 0).

원인은 **시점 불일치**다.

| | 학습 (AI Hub) | 이 영상 |
|---|---|---|
| 시점 | 차량 전방 (지상 1.5m) | 비스듬한 고정 CCTV (고가 위) |
| 차선 방향 | 아래에서 위로 수렴 | 좌우 대각선 |

흥미롭게도 SkyScapes(항공) 모델이 이 영상에서 더 나았다 — 위에서 내려다보는 구도가
CCTV 와 더 가까웠기 때문이다. **이 관찰이 BEV 변환 학습(02단계)의 직접적 동기가 됐다.**

### 9.7 사고 기록 — `/tmp` 고갈로 루트 파일시스템 100%

test 평가 3회 실행으로 **`/tmp` 에 354.5GB 가 쌓여 `/`(819G) 가 100% 로 찼다.**
발견 즉시 정리해 58% 로 복구했다 (추가로 gather 임시파일 116.2GB, 총 470.7GB 회수).

**원인 2가지 — 둘 다 `tools/test.py` 경로에만 있다.**

| # | 원인 | 크기 | 정리 |
|---|---|---|---|
| 1 | `tools/test.py:  efficient_test = True #False` (포크가 기본값을 바꿈) | 18MB x 6,423 = **118GB/회** | **안 됨 (영구 누적)** |
| 2 | `multi_gpu_test` 의 gather `tmpdir` 이 `None` -> `tempfile.mkdtemp()` -> `/tmp` | **113GB/회** | 정상 종료 시 자동 삭제 |

**학습 중 평가는 안전하다.** `DistEvalHook` 이 같은 `multi_gpu_test` 를 다른 인자로 부른다.

```python
# mmseg/core/evaluation/eval_hooks.py
multi_gpu_test(runner.model, self.dataloader,
               tmpdir=osp.join(runner.work_dir, '.eval_hook'),  # /home (2.9TB)
               gpu_collect=self.gpu_collect)                     # efficient_test 미전달 -> False
```

실증: 40,000 iter 학습(평가 20회) 후 `work_dir/.eval_hook` 이 존재하지 않는다 = 매번 자동 정리.

**대용량 test 실행 시 필수 플래그** (24분간 145회 샘플링으로 실증):

```bash
./tools/dist_test.sh <config> <ckpt> 4 --eval mIoU \
  --eval-options efficient_test=False \
  --tmpdir /home/hsjeong/tmp/mmseg_gather
```

| 항목 | 측정값 |
|---|---:|
| **`/` 증가 최대** | **0.00 GB** |
| `/home` 증가 최대 | 113.52 GB (일시적) |
| gather 디렉토리 | 최대 113.51 GB -> **종료 시 0.00 GB** |
| `/tmp` 의 `.npy` | **0개** |

**추가 안전망**: `envs/segformer_cu111/lib/python3.8/site-packages/sitecustomize.py` 를 만들어
이 환경의 python 이 `conda activate` 없이 직접 호출돼도 임시 파일이 `/home/hsjeong/tmp` 로
가게 했다 (conda env config vars 는 activate 할 때만 걸린다). 사용자가 `TMPDIR` 을 명시하면
그쪽을 존중한다.

### 9.8 다음 수순

| 우선순위 | 방법 | 근거 |
|---:|---|---|
| 1 | **BEV(IPM) 변환 후 학습** | 9.6절. 시점 정규화로 CCTV 등 다른 시점에도 대응 가능. -> `SEGFORMER_AIHUB_02_BEV_PLAN.md` |
| 2 | iter 를 60,000 이상으로 늘려 재학습 | 9.2절. 마지막 평가가 최고점이라 수렴 전이다 |
| 3 | 정밀도 개선 | 9.4절. 예측 전경의 42~52% 가 배경. 경계 가중 손실 등 |
| 4 | B2 로 교체 | SkyScapes 에서 +6.6 을 얻었다 |
