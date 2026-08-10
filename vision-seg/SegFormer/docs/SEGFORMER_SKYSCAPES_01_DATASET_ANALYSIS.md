# SEGFORMER_SKYSCAPES_01_DATASET_ANALYSIS

- 최종 갱신: 2026-08-05
- 대응 Python 파일: `UTIL/skyscapes_make_split.py`, `UTIL/setup_segformer_env.sh`, `UTIL/restore_user_site_py38.sh`
- 사용 모델: NVlabs SegFormer (MiT-B0 ~ B5), mmsegmentation 0.11.0 포크
- 저장소: `PROJECT/OCR-CAR-CATEGORY/git/SegFormer` (https://github.com/NVlabs/SegFormer.git, `--depth 1` clone 완료)
- 원본 데이터: `/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/원본/SS_Multi_Lane/SS_Multi_Lane/`
- 학습용 데이터: `/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split/` (3클래스 병합 + 재분할 완료)

---

## 1. 데이터셋 실측 구조 (원본)

```text
SS_Multi_Lane/
├── LICENSE.md
├── skyscapes18_class_table.md
├── train/{images(8 × .jpg), labels/{grayscale, rgb}(각 8 × .png)}
├── val/{images(2), labels/{grayscale, rgb}(각 2)}
└── test/images(6)            ← labels 디렉토리 없음
```

- 이미지 총 16장. 한 장이 **5616 × 3744** = 약 2100만 화소, RGB `.jpg`.
- `labels/grayscale/*.png`는 이미 클래스 인덱스 맵(uint8, 값 0~12). RGB→인덱스 변환 불필요.
- 이미지와 라벨의 stem이 동일 (`..._4K0G0010.jpg` ↔ `..._4K0G0010.png`).
- 원본 test 6장은 GT가 없어 정량 평가 불가.

---

## 2. 원본 클래스 (SkyScapes-Lane 13클래스)

`skyscapes18_class_table.md`의 SkyScapes-Lane 표 기준. grayscale 픽셀값 = 클래스 ID.
전수 확인 결과 0~12만 존재하고 13 이상은 없다.

| ID | 클래스 | RGB | 설명 | 픽셀 비율(전체 10장) |
|---:|---|---|---|---:|
| 0 | Background | `[0,0,0]` | 차선 표시가 아닌 영역 | 99.6898% |
| 1 | Dash Line | `[255,0,0]` | 긴 점선 (차로 구분선) | 0.0695% |
| 2 | Long Line | `[0,0,255]` | 얇은 실선 (추월금지선, 도로 가장자리선) | 0.1723% |
| 3 | Small dash line | `[255,255,0]` | 짧은 점선 (횡단보도 둘레 등) | 0.0140% |
| 4 | Turn signs | `[0,255,0]` | 노면 화살표 | 0.0071% |
| 5 | Other signs | `[255,128,0]` | 그 외 노면 표시 (속도 숫자 등) | 0.0065% |
| 6 | Plus sign on crossroads | `[128,0,0]` | 교차로 십자선 | 0.0007% |
| 7 | Crosswalk | `[0,255,255]` | 횡단보도 | 0.0030% |
| 8 | Stop line | `[0,128,0]` | 정지선 | 0.0068% |
| 9 | Zebra zone | `[255,0,255]` | 사선 빗금 안전지대 | 0.0174% |
| 10 | No parking zone | `[0,150,150]` | 주정차 금지 지그재그선 | 0.0019% |
| 11 | Parking space | `[200,200,0]` | 주차구획선 | 0.0037% |
| 12 | Other lane-markings | `[100,0,200]` | 그 외 차선 표시 | 0.0074% |

> 같은 파일에 SkyScapes-Dense(20클래스)도 있으나 이 디렉토리는 Lane 13클래스가 맞다.

---

## 3. 3클래스 병합 (확정)

실선/점선 판정만 필요하므로 다음과 같이 병합한다.

| 새 ID | 새 클래스 | 원본 ID | 시각화 색 |
|---:|---|---|---|
| 0 | `background` | 0, 4, 5, 6, 7, 8, 9, 10, 11, 12 | `[0,0,0]` |
| 1 | `solid` | 2 (Long Line) | `[0,0,255]` (원본 Long Line 색 계승) |
| 2 | `dashed` | 1 (Dash Line), 3 (Small dash line) | `[255,0,0]` (원본 Dash Line 색 계승) |

실선/점선이 아닌 노면표시 9종(화살표·횡단보도·정지선·안전지대·주차구획선 등)은
`ignore`가 아니라 **background로 흡수**한다. ignore로 두면 그 영역에 지도(supervision)가
없어 추론 시 정지선 위에 실선·점선이 무작위로 나타날 수 있다. background로 명시하면
"이건 차선이 아니다"를 학습하므로 배포 출력이 깨끗해진다.

### 3.1 background 클래스를 없앨 수 있는가 → 없앨 수 없다

semantic segmentation은 모든 픽셀에 클래스를 하나씩 배정한다. softmax가 N개 채널 중
하나를 고르는 구조라 "아무것도 아님"이라는 선택지가 없다. 차선이 아닌 99.7%의 픽셀도
어딘가로 가야 하고 그 자리가 background다. 즉 우리가 추가한 클래스가 아니라 구조상
반드시 존재하는 자리다.

| 방법 | 가능 | 결과 |
|---|---|---|
| 클래스 목록에서 그냥 삭제 | 불가 | 나머지 픽셀이 갈 곳이 없어 모델 구성 불가 |
| background를 `ignore_index`로 제외 | 불가 | "차선이 아닌 것"을 못 배워 화면 전체를 실선/점선으로 칠함 |
| 시그모이드 멀티라벨(2채널 독립 확률) | 가능하나 보류 | 개념상 background 소멸. mmseg 평가 코드가 argmax 전제라 개조 필요 → 원본 코드 유지 원칙과 충돌 |
| **모델엔 두고 클래스 목록·지표에서 제외** | **채택** | 실무 표준. mIoU는 solid/dashed 2개만 평균 |

**운영 규칙: 성능 보고 시 mIoU는 solid·dashed 2개만 평균낸다.** background IoU는 99%를
넘어서 함께 평균내면 지표가 무의미해진다.

---

## 4. 데이터 재분할 (완료)

- 스크립트: `UTIL/skyscapes_make_split.py`
- 출력: `/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split/` (원본 미변경, 복사본 132MB)

### 4.1 분할 기준

라벨이 있는 10장을 다시 묶어, **병합 후 solid/dashed 픽셀량이 train:val:test = 6:2:2
비율에 가장 가깝게** 배분되는 조합을 전수 탐색으로 고른다.

- 탐색 공간: C(10,2) × C(8,2) = 1,260조합
- 목적함수: 각 (split, 전경클래스)의 픽셀 점유율과 목표 점유율의 **최대 편차**를 최소화
- 제약: 모든 split이 solid·dashed를 하나 이상 보유
- 동점 처리: train 전경 픽셀이 많은 쪽 → 사전순. 결정론적이라 재실행해도 결과가 같다.

원본 split을 그대로 쓰지 않는 이유는 (a) val과 test가 분리되어 있지 않고,
(b) 원본 test 6장은 GT가 없어 정량 평가에 못 쓰기 때문이다.

### 4.2 이미지별 병합 후 픽셀 수 (실측)

| 원본 split | stem | solid | dashed |
|---|---|---:|---:|
| train | 4K0G0010 | 9,514 | 4,958 |
| train | 4K0G0020 | 44,414 | 22,131 |
| train | 4K0G0030 | 105,257 | 23,857 |
| train | 4K0G0051 | 53,867 | 17,470 |
| train | 4K0G0070 | 16,579 | 5,153 |
| train | 4K0G0080 | 487 | 73 |
| train | 4K0G0090 | 862 | 7,825 |
| train | 4K0G0100 | 6,237 | 3,558 |
| val | 4K0G0110 | 36,739 | 46,722 |
| val | 4K0G0130 | 88,369 | 43,827 |
| **합계** | | **362,325** | **175,574** |

### 4.3 확정된 split (최대 점유율 편차 5.00%p)

```text
/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split/
├── train/{images, labels/grayscale, labels/rgb}/   6장
├── val/{images, labels/grayscale, labels/rgb}/     2장
├── test/{images, labels/grayscale, labels/rgb}/    2장
└── unlabeled/images/                               6장 (GT 없음, 정성 추론 전용)
```

| split | 매수 | solid | 점유 | dashed | 점유 | 구성 (원본split/stem) |
|---|---:|---:|---:|---:|---:|---|
| train | 6 | 212,476 | 58.64% | 104,390 | 59.46% | train/0010, 0030, 0051, 0090, 0100, val/0110 |
| val | 2 | 60,993 | 16.83% | 27,284 | 15.54% | train/0020, train/0070 |
| test | 2 | 88,856 | 24.52% | 43,900 | 25.00% | train/0080, val/0130 |
| unlabeled | 6 | — | — | — | — | test/0040, 0060, 0120, 0140, 0150, 0160 |

`labels/grayscale`은 **3클래스로 재매핑된 uint8 인덱스맵**이고, `labels/rgb`는 그 시각화다.
원본 13클래스 라벨이 필요하면 원본 디렉토리에서 다시 만들면 된다.

**주의 — test의 `4K0G0080`은 전경이 560픽셀뿐인 사실상 빈 이미지다.** 목적함수가 총
픽셀 점유율만 보기 때문에 선택되었고, 실질적으로 test는 `4K0G0130` 한 장이 담당한다.
빈 이미지는 오탐(false positive) 검증용으로는 의미가 있으나, 다양성 관점에서는 손해다.
필요하면 목적함수에 "split별 최소 전경 픽셀" 제약을 추가해 재탐색할 수 있다.

### 4.4 검증 결과 (실측)

| split | 이미지 | grayscale | rgb | background | solid | dashed | 라벨 값 범위 |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 6 | 6 | 6 | 125,840,958 | 212,476 | 104,390 | `[0,1,2]` |
| val | 2 | 2 | 2 | 41,964,331 | 60,993 | 27,284 | `[0,1,2]` |
| test | 2 | 2 | 2 | 41,919,852 | 88,856 | 43,900 | `[0,1,2]` |
| unlabeled | 6 | — | — | — | — | — | — |

- 이미지/grayscale/rgb 매수 일치 및 stem 일치 확인 (assert 통과)
- 라벨 dtype uint8, 값이 `{0,1,2}` 외에 없음 확인
- rgb 고유색이 `[[0,0,0],[0,0,255],[255,0,0]]` 3종뿐임 확인
- **전경 비율은 여전히 0.25% 수준**이다. 클래스 불균형 대응은 필수 (6.4절)

### 4.5 실행 명령

```bash
# 계획과 통계만 확인 (파일 생성 없음)
python3 /home/hsjeong/workspace/vision-seg/SegFormer/scripts/skyscapes_make_split.py --dry-run

# 실제 생성 (기존 출력 삭제 후 재생성)
python3 /home/hsjeong/workspace/vision-seg/SegFormer/scripts/skyscapes_make_split.py --overwrite
```

---

## 5. 실행 환경

### 5.1 왜 현재 서버 환경으로는 불가능한가

현재 기본 환경은 **Python 3.12.13 + torch 2.6.0+cu126**이다. SegFormer는
`mmseg/__init__.py`에 하드 assert가 있다.

```python
MMCV_MIN = '1.1.4'
MMCV_MAX = '1.3.0'
assert (mmcv_min_version <= mmcv_version <= mmcv_max_version)
```

mmcv-full 1.3.0의 CUDA 확장은 torch 1.10에서 삭제된 `THC/THC.h`를 쓴다. 따라서
torch 2.6 / py3.12에서는 빌드 자체가 실패한다. 버전만 낮춰 끼우는 우회는 불가능하다.

### 5.2 사전 검증 결과 (`segformer_cu111` 실현 가능성)

| 항목 | 확인 결과 |
|---|---|
| `mmcv_full-1.3.0-cp38-cp38-manylinux1_x86_64.whl` (cu111/torch1.8.0) | **실재 확인** → 소스 컴파일 불필요. 최대 실패 요인 제거 |
| `torch-1.8.1+cu111-cp38-cp38-linux_x86_64.whl` | 실재 확인 |
| NVIDIA 드라이버 | `535.230.02` ≥ cu111 최소 요구 `450.80.02`. 드라이버는 하위 호환 |
| GPU | Tesla V100-SXM2-16GB × 8, sm_70 — cu111 빌드 타깃 포함 |
| conda | `/home/hsjeong/miniconda3/bin/conda` 25.7.0 사용. `~/.local/bin/conda`는 pip 설치본이라 깨져 있으니 쓰지 말 것 |
| 디스크 | 1.1T 여유 |

**mmcv 휠 인덱스 주소 주의**: pytorch.org가 아니라
`https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html` 이다.

### 5.3 코드에서 발견한 함정 3개 (선제 처리 완료)

1. `mmseg/core/evaluation/metrics.py:89-92`가 `np.float` 사용 → numpy 1.24부터 삭제된
   별칭이라 **평가 단계에서 AttributeError**. `numpy==1.23.5` 고정.
2. `mmseg/models/decode_heads/segformer_head.py:18` 등 **7개 파일이 모듈 최상단에서**
   `from IPython import embed` → IPython 없으면 import 자체 실패. IPython은 다시
   `pexpect`를 요구하므로 함께 설치.
3. `timm==0.3.2`는 `--no-deps`로 설치. 의존성 해석이 torch를 최신으로 끌어올리는 것을
   막는다. timm 0.3.2의 `timm.models.registry` / `timm.models.layers` 경로는 이 코드가
   쓰는 API와 일치하고, torch 1.8에는 `torch._six.container_abcs`가 아직 있어 별도 패치 불필요.

### 5.4 사고 기록 — 공유 user-site 오염 (발생 후 원복)

**증상**: `setup_segformer_env.sh` 초판이 `conda run -n segformer_cu111 pip install`을
사용했다. 그 결과 env의 site-packages는 텅 빈 채, 공유 user-site만 다운그레이드되었다.

**원인**: 이 서버에는 `/home/hsjeong/.local/lib/python3.8/site-packages`에 670개 패키지를
가진 공유 user-site가 있다. python3.8 conda env의 `sys.path`에서 이 경로가
**env 자신의 site-packages보다 앞에 온다.**

```text
/home/hsjeong/miniconda3/envs/segformer_cu111/lib/python3.8/lib-dynload
/home/hsjeong/.local/lib/python3.8/site-packages          ← user-site 가 먼저
/home/hsjeong/workspace/MSYolo/mmyolo
/home/hsjeong/miniconda3/envs/segformer_cu111/lib/python3.8/site-packages
```

pip이 user-site의 기존 설치를 "기존 버전"으로 인식해 **제자리에서 교체**했다.

**피해 및 원복**:

| 패키지 | 원래 | 덮어쓴 값 | 원복 |
|---|---|---|---|
| torch | 1.10.0 | 1.8.1+cu111 | 완료 |
| torchvision | 0.11.1 | 0.9.1+cu111 | 완료 |
| mmcv-full | 1.7.1 | 1.3.0 | 완료 |
| timm | 0.6.7 | 0.3.2 | 완료 |
| numpy | 1.24.4 | 1.23.5 | 완료 |
| opencv-python | 4.9.0.80 | 4.5.1.48 | 완료 |
| mmsegmentation | (없음) | editable egg-link 신규 | 제거 완료 |

원복 스크립트: `UTIL/restore_user_site_py38.sh`.
torch 1.10.0은 로그가 로컬 버전 접미사 없이 기록되어 PyPI 기본 빌드(cu102)로 판단했고,
그에 맞춰 mmcv-full 1.7.1도 `cu102/torch1.10.0` 사전 빌드 휠을 사용했다.
모든 복구 설치는 `--no-deps`로 해 나머지 669개 패키지를 건드리지 않았다.

**재발 방지**: `setup_segformer_env.sh`를 다음과 같이 고쳤다.

- `export PYTHONNOUSERSITE=1` — user-site를 sys.path에서 완전히 제거
- `conda env config vars set PYTHONNOUSERSITE=1 -n segformer_cu111` — activate 시에도 유지
- `conda run` 대신 env의 python을 직접 호출: `$ENV_PREFIX/bin/python -m pip install ...`
- 설치 전 격리 확인 단계 추가 (sys.path에 user-site가 있으면 assert 로 중단)
- 스모크 테스트에서 `numpy/torch/mmcv/timm`의 `__file__`이 env 내부인지 검증

### 5.5 SegFormerHead가 SyncBN을 하드코딩한다 (단일 GPU 학습 시 필수 주의)

`mmseg/models/decode_heads/segformer_head.py:59`

```python
self.linear_fuse = ConvModule(
    in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1,
    norm_cfg=dict(type='SyncBN', requires_grad=True))   # config 의 norm_cfg 를 무시한다
```

config에서 `norm_cfg=dict(type='BN')`을 줘도 디코더의 `linear_fuse`는 **항상 SyncBN**이다.
SyncBN은 train 모드에서 `torch.distributed.get_world_size()`를 호출하므로 프로세스 그룹이
초기화돼 있지 않으면 다음 에러가 난다. 추론(eval)은 영향이 없어 forward만 테스트하면 못 잡는다.

```text
RuntimeError: Default process group has not been initialized,
please make sure to call init_process_group.
```

**대응**: GPU가 1장이어도 학습은 분산 런처로 돌린다. 원본 코드를 고치지 않는 정석 경로다.

```bash
cd /home/hsjeong/workspace/vision-seg/SegFormer/source
./tools/dist_train.sh <config> 1        # GPU 1장
./tools/dist_train.sh <config> 4        # GPU 4장
```

`python tools/train.py <config>` 단독 실행은 이 이유로 실패한다.

### 5.6 환경 구축 완료 및 검증 결과 (실측)

```text
python     : /home/hsjeong/miniconda3/envs/segformer_cu111/bin/python
numpy      : 1.23.5   @ .../envs/segformer_cu111/lib/python3.8/site-packages/numpy
torch      : 1.8.1+cu111 @ .../envs/segformer_cu111/lib/python3.8/site-packages/torch
torch cuda : 11.1        cuda avail: True     gpu count: 8
gpu0       : Tesla V100-SXM2-16GB
mmcv       : 1.3.0       timm: 0.3.2          mmseg: 0.11.0
설치 위치  : numpy/torch/mmcv/timm 전부 env 내부 확인 (user-site 아님)
np.float ok: float64
mmcv ops   : import OK (CUDA 확장 정상 로드)
```

분산 초기화 후 실제 학습 스텝 검증:

```text
SegFormer-B0 파라미터: 3.72M
train loss : 1.1306
gradient   : 189 / 191 파라미터에 grad 생성
peak VRAM  : 0.99 GB (batch 2, 512x512)
eval out   : (1, 3, 512, 512)
```

- grad가 안 생긴 2개는 `linear_fuse`의 SyncBN running_mean/var 계열 buffer로, 학습 대상이 아니다.
- B0 기준 batch 2 / 512×512에서 1GB 미만. V100 16GB면 배치를 크게 잡을 여유가 충분하다.

### 5.7 실행 명령

```bash
# 환경 구축 (기존 segformer_cu111 이 있으면 지우고 재생성)
bash /home/hsjeong/workspace/vision-seg/SegFormer/scripts/setup_segformer_env.sh

# 활성화
conda activate segformer_cu111
```

스모크 테스트는 패키지 버전과 **설치 위치(`__file__`)**, `np.float` 동작, mmcv CUDA 확장 로드,
mmseg 전체 임포트, SegFormer 3클래스 구성, 학습 스텝(손실+역전파+grad 생성 수+peak VRAM),
추론 출력 shape까지 검증한다.

### 5.6 학습 후 모델 실행 환경

| 방법 | py3.12 + torch2.6에서 실행 | 설명 |
|---|---|---|
| 학습 conda 환경 그대로 | 불가 | 동일 환경 필요 |
| **ONNX 내보내기** | **가능** | `tools/pytorch2onnx.py`로 1회 변환 후 onnxruntime/TensorRT로 어디서든 실행. 배포 권장 경로 |
| HF transformers로 가중치 이식 | 가능 | state_dict 키 매핑 1회 필요 |

`.pth` 체크포인트 자체는 텐서 묶음이라 환경 종속이 없다. 종속되는 것은 `mit_b*` 백본과
`SegFormerHead` 정의가 이 mmseg 포크 안에만 있다는 점이다. **학습은 conda 환경,
배포는 ONNX**가 무난하다.

---

## 6. 남은 작업 (02단계 이후)

### 6.1 타일링

5616×3744를 그대로 넣을 수 없다. mmseg CustomDataset 레이아웃으로 타일을 만든다.

```text
data/skyscapes_lane/
├── img_dir/{train, val, test}/   *.png (1024×1024 타일)
└── ann_dir/{train, val, test}/   *.png (동일 stem, uint8 0/1/2)
```

- 1024×1024, stride 512(50% 오버랩) 기준 장당 약 70~80타일
  → train 약 420~480, val 약 140~160, test 약 140~160
- 라벨은 반드시 **nearest neighbor**로 처리 (보간하면 없는 클래스 ID가 생김)
- 전경 픽셀이 0인 타일이 대량 발생한다. 남길 비율을 옵션으로 제어

### 6.2 데이터셋 클래스 등록 (3단계)

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

**③ `local_configs/segformer/B*/segformer.b*.1024x1024.skyscapes.160k.py` 작성**

### 6.3 config 핵심 변경점

`local_configs/segformer/B0/segformer.b0.512x1024.city.160k.py` 복사 후 수정.

| 항목 | Cityscapes 원본 | SkyScapes-Lane |
|---|---|---|
| `decode_head.num_classes` | 19 | **3** |
| `dataset_type` | `CityscapesDataset` | `SkyScapesLaneDataset` |
| `data_root` | `data/cityscapes/` | `data/skyscapes_lane/` |
| `img_dir` / `ann_dir` | `leftImg8bit/*` / `gtFine/*` | `img_dir/*` / `ann_dir/*` |
| `crop_size` | `(512, 1024)` | `(1024, 1024)` 또는 `(512, 512)` |
| `samples_per_gpu` | 1 | V100 16GB 기준 B0=4~8, B2=2, B5=1 |
| `norm_cfg` | `SyncBN` | **`SyncBN` 유지**. 어차피 `linear_fuse`가 하드코딩이라 학습은 항상 분산 런처 필요 (5.5절) |
| `test_cfg` | `mode='whole'` | 원본 크기 추론 시 `mode='slide'` + crop/stride 지정 |
| `evaluation.interval` | 4000 | 데이터가 작으므로 1000~2000 |
| `reduce_zero_label` | — | **`False`** (0번이 실제 background 클래스) |

`PhotoMetricDistortion` 유지. 항공 뷰는 방향 불변성이 있어 `RandomRotate`(90도 배수)
추가 검토 가치가 있다.

### 6.4 클래스 불균형 대응

병합 후에도 전경은 0.25% 수준이다. 기본 CrossEntropy만 쓰면 전부 background로 예측하고도
pixel accuracy 99.7%가 나온다.

- 1순위: `loss_decode=dict(type='CrossEntropyLoss', class_weight=[...])`
  — 4.2절 픽셀 수 기반 역빈도 가중치
- 2순위: OHEM (`train_cfg=dict(sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000))`)
- 3순위: 타일링 시 전경 없는 타일 비율 제한
- **평가는 solid/dashed의 클래스별 IoU를 반드시 확인.** Overall Accuracy는 무의미하다.

### 6.5 사전학습 가중치

- config의 `pretrained='pretrained/mit_b0.pth'`가 없으면 실패한다.
- train 6장 규모상 ImageNet 사전학습 백본은 선택이 아니라 필수다.
- **해결 완료 (03단계 1절 참조)**: README의 google drive 링크가 동작하지 않아,
  HuggingFace `nvidia/mit-b{0..5}`를 받아 `UTIL/mit_hf_to_nvlabs.py`로 키를 변환했다.
  b0~b5 6개 모두 `SegFormer/pretrained/mit_b*.pth`에 배치 완료.
- **주의**: mmseg 공식(openmmlab) 체크포인트는 이름이 비슷해도 **쓸 수 없다.** 구현체가
  달라 키가 전혀 안 맞는데, `init_weights()`가 `strict=False`라 조용히 무시되고
  랜덤 초기화로 학습된다.

---

## 7. 진행 상태

| 단계 | 상태 |
|---|---|
| SegFormer clone | **완료** (`git/SegFormer`) |
| 데이터 구조/클래스 분석 | **완료** (1~2절) |
| 3클래스 병합 정책 확정 | **완료** (3절) |
| 데이터 재분할 + 라벨 재매핑 | **완료** (`UTIL/skyscapes_make_split.py`, 4절) |
| 공유 user-site 오염 원복 | **완료** (`UTIL/restore_user_site_py38.sh`, 5.4절) |
| `segformer_cu111` 환경 구축 | **완료** (`UTIL/setup_segformer_env.sh`, 학습 스텝까지 검증, 5.6절) |
| 타일링 (02단계) | **완료** → `SEGFORMER_SKYSCAPES_02_TILING.md` |
| 데이터셋 클래스 + config (03단계) | **완료** → `SEGFORMER_SKYSCAPES_03_TRAINING_SETUP.md` |
| 사전학습 가중치 확보 | **완료** — google drive 대신 HuggingFace `nvidia/mit-b*` 를 변환 (03단계 1절) |
| 학습 실행 | 준비 완료 (4-GPU 스모크 검증 통과), 본 학습은 사용자가 직접 실행 |
