# SEGFORMER_SKYSCAPES_02_TILING

- 최종 갱신: 2026-08-05
- 대응 Python 파일: `UTIL/skyscapes_make_tiles.py`
- 사용 모델: NVlabs SegFormer (mmsegmentation 0.11.0 포크)
- 입력: `/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split/{train,val,test}/`
- 출력: `/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split/tiles/{train,val,test}/`
- 선행 문서: `SEGFORMER_SKYSCAPES_01_DATASET_ANALYSIS.md`

---

## 1. SegFormer(mmseg)가 데이터를 읽는 규칙

SegFormer 자체에는 데이터 로더가 없다. mmsegmentation 0.11.0의 `CustomDataset`이 전부 처리한다.
근거는 `mmseg/datasets/custom.py:146-151`이다.

```python
for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
    img_info = dict(filename=img)
    if ann_dir is not None:
        seg_map = img.replace(img_suffix, seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
```

여기서 도출되는 **필수 조건 4가지**다.

| # | 조건 | 설명 |
|---|---|---|
| 1 | `img_dir`와 `ann_dir`가 **별개의 디렉토리** | 이미지 폴더를 재귀 스캔해 목록을 만들고, 라벨은 같은 상대경로로 `ann_dir` 아래에서 찾는다 |
| 2 | 파일명이 **접미사만 빼고 동일** | `img.replace(img_suffix, seg_map_suffix)` 단순 문자열 치환이다. 우리는 양쪽 다 `.png`라 파일명이 완전히 같으면 된다 |
| 3 | 라벨은 **uint8 단일 채널 인덱스맵** | 픽셀값이 곧 클래스 ID. RGB 컬러맵을 넣으면 안 된다 |
| 4 | 클래스 ID가 **0부터 연속** | `num_classes`와 개수가 맞아야 한다. 우리는 `0,1,2` |

추가로 알아둘 것:

- `ignore_index`는 기본 **255**다. 라벨의 255 픽셀은 손실 계산과 IoU 집계에서 자동 제외된다.
- `reduce_zero_label=False`로 둬야 한다. `True`면 모든 라벨값을 1씩 빼서 0을 ignore로 만드는데,
  우리는 0이 실제 background 클래스라 그러면 안 된다.
- 이미지 크기가 제각각이어도 되지만, 우리는 전부 1024×1024로 통일했다.

---

## 2. 출력 디렉토리 구조

```text
/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split/tiles/
├── train/
│   ├── images/    222 × .png   (1024×1024 RGB)
│   └── labels/    222 × .png   (1024×1024 uint8, 값 0/1/2)
├── val/
│   ├── images/     48 × .png
│   └── labels/     48 × .png   (값 0/1/2/255)
└── test/
    ├── images/     48 × .png
    └── labels/     48 × .png   (값 0/1/2/255)
```

`images/`와 `labels/`의 파일명이 확장자까지 완전히 동일하다. 타일 파일명 규칙은
`{원본stem}_x{좌상단X:05d}_y{좌상단Y:05d}.png`이며, 좌표가 들어 있어 나중에
타일 예측을 원본 좌표계로 되돌릴 수 있다.

```text
2012-04-26-Muenchen-Tunnel_4K0G0010_x00000_y00512.png
```

---

## 3. 타일 규칙

원본은 5616×3744(약 2100만 화소)라 그대로 넣을 수 없다. 1024×1024로 자른다.

### 3.1 train — 겹침 있음 (stride 512)

- 소스가 6장뿐이라 50% 오버랩으로 샘플 수를 늘린다. 중복은 증강으로 본다.
- 가장자리 타일은 이미지 안쪽으로 당겨 붙인다(패딩 없음).
- 가로 시작좌표 10개 × 세로 7개 = 장당 70타일 → 6장 × 70 = **420타일**

### 3.2 val/test — 겹침 없음 (격자 + ignore 패딩)

**평가에서는 각 픽셀이 정확히 한 번만 집계되어야 한다.** 처음엔 train과 같이
"마지막 타일을 안쪽으로 당기는" 방식을 썼는데, 5616은 1024로 나누어떨어지지 않아
가로 528px·세로 352px이 두 타일에 중복으로 들어갔다. 실측 피해는 다음과 같았다.

```text
test solid   원본 88,856 -> 타일 합계 175,017  (×1.97 이중 집계)
test dashed  원본 43,900 -> 타일 합계  65,937  (×1.50 이중 집계)
```

이 상태로 mIoU를 재면 겹친 영역의 픽셀이 두 번 반영되어 지표가 왜곡된다.

**수정**: 0부터 tile 간격의 격자로 자르고, 이미지 밖으로 나가는 오른쪽·아래를 패딩한다.

- 이미지 패딩값 = `0` (검은색)
- 라벨 패딩값 = `255` (= `ignore_index`) → 손실·지표에서 자동 제외
- 가로 6개 × 세로 4개 = 장당 24타일 → 2장 × 24 = **48타일** (val, test 각각)
- 패딩 픽셀 수 = `6144×4096 − 5616×3744 = 4,139,520` /장 → 2장 = **8,279,040** (실측 일치)

### 3.3 빈 타일 솎아내기 (train만)

전경(solid/dashed)이 한 픽셀도 없는 타일이 249개나 나온다. 전부 쓰면 학습이
"전부 background"로 수렴하기 쉽다. 전경 타일 수의 `--train-empty-ratio`배(기본 0.3)까지만
남기고, 정렬된 목록에서 균등 간격으로 결정론적으로 고른다.

val/test는 평가 커버리지를 위해 **전부 사용**한다. 빈 타일을 빼면 오탐(false positive)을
측정할 수 없기 때문이다.

---

## 4. 생성 결과 및 검증 (실측)

### 4.1 타일 수

| split | 전체 타일 | 전경 타일 | 빈 타일 | 선택 | 비고 |
|---|---:|---:|---:|---:|---|
| train | 420 | 171 | 249 | **222** | 전경 171 + 빈 51 |
| val | 48 | 25 | 23 | **48** | 전부 사용 |
| test | 48 | 14 | 34 | **48** | 전부 사용 |

총 318타일, 484MB.

### 4.2 픽셀 보존 검증 — 핵심 항목

| split | solid (타일) | solid (원본) | 배율 | dashed (타일) | dashed (원본) | 배율 | ignore |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 787,720 | 212,476 | ×3.707 | 365,576 | 104,390 | ×3.502 | 0 |
| val | 60,993 | 60,993 | **×1.000** | 27,284 | 27,284 | **×1.000** | 8,279,040 |
| test | 88,856 | 88,856 | **×1.000** | 43,900 | 43,900 | **×1.000** | 8,279,040 |

- **val/test는 원본 픽셀 수와 정확히 일치**한다. 이중 집계가 완전히 제거됐다.
- train의 ×3.5~3.7은 stride 512 오버랩에 의한 의도된 중복이다.
- ignore 픽셀 수가 이론값 `8,279,040`과 정확히 일치한다.

### 4.3 그 밖의 검증

- `images/`와 `labels/`의 매수 및 파일명 완전 일치 (assert 통과)
- 모든 타일이 정확히 1024×1024
- 이미지 mode = `RGB`, 라벨 dtype = `uint8`
- 라벨 고유값: train `{0,1,2}`, val/test `{0,1,2,255}` — 그 외 값 없음

### 4.4 남은 한계

- **test 전경 타일이 14개뿐**이다. 01단계 4.3절에 적었듯 test의 `4K0G0080`이 전경
  560픽셀짜리 사실상 빈 이미지라, 실질 평가는 `4K0G0130` 한 장이 담당한다.
- train 전경 타일 171개는 적은 편이다. 사전학습 백본과 강한 증강,
  `RepeatDataset`이 필수다.
- 타일을 512로 줄이면 샘플 수가 크게 늘지만(장당 294개) 문맥이 좁아진다.
  실선/점선 구분은 점선의 끊김 주기를 봐야 하므로 문맥이 중요하다. 1024를 기본으로 둔다.

---

## 5. 실행 명령

```bash
# 계획만 확인 (파일 생성 없음)
python3 /home/hsjeong/workspace/vision-seg/SegFormer/scripts/skyscapes_make_tiles.py --dry-run

# 실제 생성 (기존 tiles/ 삭제 후 재생성)
python3 /home/hsjeong/workspace/vision-seg/SegFormer/scripts/skyscapes_make_tiles.py --overwrite

# 타일 크기/스트라이드 변경
python3 /home/hsjeong/workspace/vision-seg/SegFormer/scripts/skyscapes_make_tiles.py \
    --tile 512 --stride-train 256 --overwrite

# 빈 타일을 더 많이/적게 남기기 (전경 타일 수 대비 배수)
python3 /home/hsjeong/workspace/vision-seg/SegFormer/scripts/skyscapes_make_tiles.py \
    --train-empty-ratio 0.1 --overwrite
```

---

## 6. 다음 단계에서 만들 것 (03단계)

타일은 준비됐고, mmseg에 인식시키는 작업이 남았다. 01단계 6.2~6.3절과 동일하되
경로가 확정된 형태다.

**① `mmseg/datasets/skyscapes.py` 신규 작성**

```python
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SkyScapesLaneDataset(CustomDataset):
    CLASSES = ('background', 'solid', 'dashed')
    PALETTE = [[0, 0, 0], [0, 0, 255], [255, 0, 0]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         reduce_zero_label=False, **kwargs)
```

**② `mmseg/datasets/__init__.py`에 import + `__all__` 등록**

**③ config의 data 섹션** — 디렉토리 구조가 위와 같으므로 다음이 그대로 들어간다.

```python
dataset_type = 'SkyScapesLaneDataset'
data_root = '/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split/tiles'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,                      # 222타일뿐이라 반복으로 에폭당 스텝을 확보
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/images',
            ann_dir='train/labels',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type, data_root=data_root,
        img_dir='val/images', ann_dir='val/labels', pipeline=test_pipeline),
    test=dict(
        type=dataset_type, data_root=data_root,
        img_dir='test/images', ann_dir='test/labels', pipeline=test_pipeline))
```

**주의**: 학습 실행은 `python tools/train.py`가 아니라 분산 런처를 써야 한다.
`SegFormerHead.linear_fuse`가 SyncBN을 하드코딩하기 때문이다 (01단계 5.5절).

```bash
cd /home/hsjeong/workspace/vision-seg/SegFormer/source
./tools/dist_train.sh <config> 1
```
