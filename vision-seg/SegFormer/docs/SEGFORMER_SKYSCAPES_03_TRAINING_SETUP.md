# SEGFORMER_SKYSCAPES_03_TRAINING_SETUP

- 최종 갱신: 2026-08-05
- 대응 파일: `UTIL/mit_hf_to_nvlabs.py`, `git/SegFormer/mmseg/datasets/skyscapes.py`,
  `git/SegFormer/local_configs/segformer/B0/segformer.b0.1024x1024.skyscapes.20k.py`
- 선행 문서: `SEGFORMER_SKYSCAPES_01_DATASET_ANALYSIS.md`, `SEGFORMER_SKYSCAPES_02_TILING.md`
- 상태: **B0 1차 학습 완료** (8절에 결과)

---

## 1. 사전학습 백본 — HuggingFace 에서 조달

01단계 6.5절의 계획은 NVlabs README 의 google drive 링크에서 `mit_b*.pth` 를 받는 것이었으나
다운로드가 되지 않았다. 대신 **HuggingFace `nvidia/mit-b{0..5}`** 에서 받아 변환했다.
같은 ImageNet-1k 사전학습 MiT 백본이고 구현체만 다르다.

### 1.1 먼저 받아 뒀던 파일은 쓸 수 없다

`pretrained/segformer_mit-b*_8x1_1024x1024_160k_cityscapes_*.pth` 3개는 **openmmlab 공식
mmseg 구현체**의 체크포인트다. NVlabs 포크와 키 이름이 전혀 다르다.

| | 키 예시 | 텐서 수 |
|---|---|---:|
| openmmlab 공식 (받아 둔 파일) | `backbone.layers.0.0.projection.weight`, `decode_head.fusion_conv.*` | 192 |
| NVlabs 포크 (이 코드베이스) | `patch_embed1.proj.weight`, `block1.0.norm1.*` | 176 |

게다가 Cityscapes 19클래스로 **학습이 끝난** 모델이라 ImageNet 백본이 아니다.
`MixVisionTransformer.init_weights()` 가 `load_checkpoint(..., strict=False)` 라
**에러 없이 조용히 전부 무시되고 랜덤 초기화로 학습**된다. 가장 위험한 실패 유형이다.

### 1.2 키 변환 규칙

`UTIL/mit_hf_to_nvlabs.py` 가 처리한다.

| HuggingFace | NVlabs |
|---|---|
| `segformer.encoder.patch_embeddings.{i}.proj.*` | `patch_embed{i+1}.proj.*` |
| `segformer.encoder.patch_embeddings.{i}.layer_norm.*` | `patch_embed{i+1}.norm.*` |
| `segformer.encoder.block.{i}.{j}.layer_norm_1.*` | `block{i+1}.{j}.norm1.*` |
| `segformer.encoder.block.{i}.{j}.layer_norm_2.*` | `block{i+1}.{j}.norm2.*` |
| `...attention.self.query.*` | `block{i+1}.{j}.attn.q.*` |
| `...attention.self.key.*` + `...attention.self.value.*` | `block{i+1}.{j}.attn.kv.*` (**concat**) |
| `...attention.self.sr.*` | `block{i+1}.{j}.attn.sr.*` |
| `...attention.self.layer_norm.*` | `block{i+1}.{j}.attn.norm.*` |
| `...attention.output.dense.*` | `block{i+1}.{j}.attn.proj.*` |
| `...mlp.dense1.*` / `...mlp.dense2.*` | `block{i+1}.{j}.mlp.fc1.*` / `mlp.fc2.*` |
| `...mlp.dwconv.dwconv.*` | `block{i+1}.{j}.mlp.dwconv.dwconv.*` |
| `segformer.encoder.layer_norm.{i}.*` | `norm{i+1}.*` |
| `classifier.*` | 버림 (ImageNet 1000클래스 분류기) |

**핵심은 k/v 결합**이다. HF 는 key/value 를 별도 Linear 로 두지만 NVlabs 는
`kv = nn.Linear(dim, dim*2)` 하나로 합쳐 뒀다. `mix_transformer.py:104-107` 의
`kv(x).reshape(B, -1, 2, heads, hd)` 는 채널을 row-major 로 펼치므로 앞쪽 `dim` 채널이 k,
뒤쪽 `dim` 채널이 v 다. 따라서 `kv.weight = cat([key.weight, value.weight], dim=0)`.
순서를 뒤집으면 에러 없이 조용히 성능만 무너진다.

스크립트는 매핑 규칙에 없는 키를 만나면 **조용히 버리지 않고 예외를 던진다.**
가중치가 일부만 실리는 사고는 학습이 다 끝난 뒤에야 성능으로 드러나기 때문이다.

### 1.3 변환 검증 결과 (실측)

세 겹으로 검증했다.

1. **어텐션 수치 검증** — 표준 MHA(`softmax(q@kᵀ·scale)@v`)를 HF 가중치로 직접 계산한 값과,
   변환된 가중치를 실은 NVlabs `Attention` 모듈의 출력을 비교. k/v concat 순서가 뒤집혔다면
   여기서 걸린다. → **6개 모델 전부 `max|diff| = 0.00e+00`**
2. **`load_state_dict(strict=True)`** — 키와 shape 이 하나라도 안 맞으면 예외.
3. **파라미터 수가 SegFormer 논문 Table 과 일치** — 외부 기준과의 대조.

| 모델 | 텐서 | 파라미터 (논문값) | 파일 | 크기 |
|---|---:|---:|---|---:|
| mit_b0 | 176 | 3.32M (3.4M) | `pretrained/mit_b0.pth` | 13.3MB |
| mit_b1 | 176 | 13.15M (13.1M) | `pretrained/mit_b1.pth` | 52.7MB |
| mit_b2 | 332 | 24.20M (24.2M) | `pretrained/mit_b2.pth` | 96.9MB |
| mit_b3 | 572 | 44.07M (44.1M) | `pretrained/mit_b3.pth` | 176.5MB |
| mit_b4 | 832 | 60.84M (60.8M) | `pretrained/mit_b4.pth` | 243.6MB |
| mit_b5 | 1052 | 81.44M (81.4M) | `pretrained/mit_b5.pth` | 326.1MB |

### 1.4 실행 명령

```bash
# 전체 (b0~b5) 변환. 기존 파일이 있으면 건너뛴다
cd /home/hsjeong/workspace/vision-seg/SegFormer/source && \
PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/segformer_cu111/bin/python \
  /home/hsjeong/workspace/vision-seg/SegFormer/scripts/mit_hf_to_nvlabs.py

# 특정 모델만 / 검증만 (저장 안 함) / 덮어쓰기
  ... mit_hf_to_nvlabs.py --models b0 b2
  ... mit_hf_to_nvlabs.py --models b0 --dry-run
  ... mit_hf_to_nvlabs.py --overwrite
```

---

## 2. 데이터셋 클래스 등록

**① `mmseg/datasets/skyscapes.py` (신규)** — `SkyScapesLaneDataset`,
`CLASSES=('background','solid','dashed')`, `PALETTE=[[0,0,0],[0,0,255],[255,0,0]]`,
`img_suffix='.png'`, `seg_map_suffix='.png'`, `reduce_zero_label=False`.

**② `mmseg/datasets/__init__.py`** — import 추가 + `__all__` 에 등록. (이 2줄이 원본 대비 유일한 수정)

---

## 3. config

`local_configs/segformer/B0/segformer.b0.1024x1024.skyscapes.20k.py` (신규).
B2 는 이 파일을 `_base_` 로 상속받아 백본/디코더 폭과 배치만 바꾼다.

### 3.1 Cityscapes 원본 대비 변경점

| 항목 | Cityscapes 원본 | SkyScapes-Lane | 이유 |
|---|---|---|---|
| `num_classes` | 19 | **3** | background/solid/dashed |
| `dataset_type` | `CityscapesDataset` | `SkyScapesLaneDataset` | |
| `data_root` | `data/cityscapes/` | `.../DLR-SkyScapes/split/tiles` | 02단계 타일 |
| `img_dir`/`ann_dir` | `leftImg8bit/*`/`gtFine/*` | `{split}/images`/`{split}/labels` | |
| `crop_size` | (512,1024) | **(1024,1024)** | 타일 크기와 동일 |
| `RandomCrop.cat_max_ratio` | 0.75 | **1.0 (비활성)** | 아래 3.2 |
| `loss_decode.class_weight` | 없음 | **[1.0, 17.1479, 25.1715]** | 아래 3.3 |
| `samples_per_gpu` | 1 | **2** | V100 16GB 실측, 아래 3.4 |
| `max_iters` | 160000 | **20000** | 타일 222장 규모 |
| `evaluation.interval` | 4000 | **1000** | |
| `checkpoint_config` | interval 4000 | **interval 1000, 전부 보관** | 아래 3.5 |
| `test_cfg` | `mode='whole'` | `mode='whole'` 유지 | 타일이 정확히 crop_size |
| `RepeatDataset.times` | 500 | **50** | |

### 3.2 `cat_max_ratio` 는 반드시 꺼야 한다

`mmseg/datasets/pipelines/transforms.py` 의 `RandomCrop` 은 `cat_max_ratio < 1.0` 이면
"한 클래스 점유율이 그 값 미만이 될 때까지 최대 10회 재크롭"을 시도한다.
우리 데이터는 **배경이 99.5%** 라 0.75 조건을 절대 만족할 수 없다. 결국 매 샘플마다
10번 헛돌고 마지막 crop 을 그냥 쓴다 — 효과는 0인데 데이터로딩만 10배 느려진다.

### 3.3 클래스 가중치

train 타일 222장(232,783,872 픽셀) 실측:

| 클래스 | 픽셀 | 비율 |
|---|---:|---:|
| background | 231,630,576 | 99.5046% |
| solid | 787,720 | 0.3384% |
| dashed | 365,576 | 0.1570% |

가중치별 손실 기여도(= 빈도 × 가중치):

| 방식 | 가중치 | 배경 : 전경 |
|---|---|---:|
| 없음 | [1, 1, 1] | 200.8 : 1 |
| **sqrt-inverse (채택)** | **[1.0, 17.1479, 25.1715]** | **10.2 : 1** |
| median-frequency | [0.00323, 0.94993, 2.04684] | 0.5 : 1 |

가중치 없이 학습하면 전부 background 로 예측하고도 픽셀 정확도 99.5% 가 나온다.
median-frequency 는 완전 균형이지만 오탐이 늘 수 있어, 중간값인 sqrt-inverse 를 기본으로 뒀다.
**4000 iter 를 넘겨도 solid/dashed IoU 가 0 근처면 median-frequency 로 교체**한다
(config 주석에 값이 들어 있다).

### 3.4 VRAM 실측 (V100 16GB, 1024×1024, GPU 1장)

| samples_per_gpu | peak VRAM |
|---:|---:|
| 2 | 6.80 GB |
| 3 | 10.19 GB |
| 4 | 13.58 GB (총 15.77GB 중 86%) |

샘플당 약 3.4GB. 4 는 평가 단계 오버헤드까지 겹치면 위험해서 **2** 로 뒀다.
유효 배치는 GPU 장수로 키운다 (`samples_per_gpu` 는 GPU 1장당 값이다).

B2 는 `samples_per_gpu=2` 가 **OOM** 이라 1 로 뒀다 (batch 1 에서 8.52GB).

### 3.5 체크포인트를 전부 보관하는 이유

mmseg 0.11.0 의 `EvalHook` 은 **best 체크포인트를 따로 저장하지 않는다.**
`max_keep_ckpts` 를 걸면 최신 N개만 남으므로 중간에 나온 최고 성능 체크포인트가 지워진다.
B0 는 1개 15MB 라 20개 다 보관해도 300MB다. 학습 후 로그의 mIoU 추이를 보고 직접 고른다.

---

## 4. 환경 수정 — yapf 다운그레이드

스모크 실행에서 학습이 **첫 iter 도 못 돌고** 죽었다.

```text
File ".../mmcv/utils/config.py", line 413, in pretty_text
    text, _ = FormatCode(text, style_config=yapf_style, verify=True)
TypeError: FormatCode() got an unexpected keyword argument 'verify'
```

`tools/train.py:102` 가 시작 직후 `cfg.dump()` 를 호출하고, 그게 mmcv 1.3.0 의
`Config.pretty_text` → `yapf.FormatCode(..., verify=True)` 를 탄다. **yapf 0.40 에서
`verify` 인자가 삭제**되어 env 에 깔려 있던 0.43.0 과 충돌했다.

**조치**: env 내부에만 `yapf==0.32.0` 설치. `setup_segformer_env.sh` 에도 고정을 추가해
환경을 재구축해도 재발하지 않게 했다.

```bash
PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/segformer_cu111/bin/python \
  -m pip install --no-deps "yapf==0.32.0"
```

---

## 5. 검증 결과 (실측)

### 5.1 구성 요소별

```text
[1] config 파싱 OK
[2] 데이터셋 등록: True
[3] train   222장  img(3,1024,1024) seg(1,1024,1024)
    val      48장  CLASSES=('background','solid','dashed')
    test     48장
[4] 모델 빌드 OK  파라미터 3.72M   num_classes = 3
[5] 사전학습 적재 검증: 체크포인트 176개 중 176개가 백본과 값 일치
[6] batch=2 학습스텝 OK  loss=1.6766  grad생성 189/191   peak VRAM 6.80GB
```

`[5]` 가 핵심이다. `init_weights()` 가 `strict=False` 라 로그만 봐서는 가중치가 실렸는지
알 수 없으므로, 체크포인트 텐서와 백본 파라미터 값을 직접 대조했다.
grad 가 안 생긴 2개는 `linear_fuse` SyncBN 의 running_mean/var buffer 로 학습 대상이 아니다.

### 5.2 4-GPU 실제 학습 (60 iter 스모크)

```text
Iter [20/60]  time: 0.472  memory: 7369  decode.loss_seg: 1.0635  decode.acc_seg: 36.96
Iter [40/60]  time: 0.272  memory: 7369  decode.loss_seg: 1.0267  decode.acc_seg: 40.17
Saving checkpoint at 60 iterations
+------------+-------+-------+
| Class      | IoU   | Acc   |
+------------+-------+-------+
| background | 51.84 | 51.89 |
| solid      |  0.19 | 13.69 |
| dashed     |  0.08 | 46.54 |
+------------+-------+-------+
```

학습·체크포인트 저장·클래스별 IoU 평가가 전부 동작한다. 60 iter 시점은 아직 warmup
구간(lr 5.5e-7)이라 IoU 는 의미 없다. 다만 **Acc 가 13.7/46.5 로 이미 전경을 예측하고 있어**
클래스 가중치가 의도대로 먹고 있음은 확인된다.

- 정상 상태 **0.272 s/iter** → 20000 iter ≈ **1시간 32분** (+ 평가 20회 약 2분)
- GPU 1장당 7.4GB

---

## 6. 진행 상태

| 단계 | 상태 |
|---|---|
| 01 데이터 분석 / 3클래스 병합 / 재분할 | 완료 |
| 01 `segformer_cu111` 환경 구축 | 완료 (+ yapf 0.32.0 고정 추가) |
| 02 타일링 | 완료 |
| **03 사전학습 백본 확보 (HF 변환)** | **완료** — b0~b5 6개 |
| **03 데이터셋 클래스 등록** | **완료** |
| **03 config 작성 (B0, B2)** | **완료** |
| **03 4-GPU 스모크 실행 검증** | **완료** |
| **B0 본 학습 (10,000 iter)** | **완료** — 8절, test 전경평균 IoU **39.81** |
| **B2 본 학습 (30,000 iter)** | **완료** — 9절, test 전경평균 IoU **46.39** (`iter_14000.pth`) |
| **test 평가 / 혼동행렬 / 시각화** | **완료** — 두 모델 모두 |
| dashed 가중치 조정 재학습 | 미착수 — 9.7절 1순위 |
| 원본 해상도 추론 / ONNX 내보내기 | 미착수 (04단계) |

---

## 7. 학습 실행 명령

```bash
cd /home/hsjeong/workspace/vision-seg/SegFormer/source

PATH=/home/hsjeong/miniconda3/envs/segformer_cu111/bin:$PATH \
PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./tools/dist_train.sh \
  local_configs/segformer/B0/segformer.b0.1024x1024.skyscapes.20k.py 4 \
  --work-dir work_dirs/b0_skyscapes_20k \
  --seed 0 --deterministic
```

`PATH` 앞에 env 의 bin 을 붙이는 이유는 `dist_train.sh` 가 `python -m torch.distributed.launch`
를 그대로 호출하기 때문이다. `conda activate segformer_cu111` 을 먼저 했다면 `PATH` 와
`PYTHONNOUSERSITE` 는 생략해도 된다.

**`python tools/train.py` 단독 실행은 실패한다.** `SegFormerHead.linear_fuse` 가 SyncBN 을
하드코딩(`segformer_head.py:59`)해서 프로세스 그룹 초기화가 필요하다. GPU 1장이어도
`./tools/dist_train.sh <config> 1` 로 돌려야 한다 (01단계 5.5절).

### 7.1 학습 중 볼 것

`work_dirs/b0_skyscapes_20k/*.log` 에 1000 iter 마다 클래스별 IoU 표가 찍힌다.
**`background` IoU 는 무시하고 `solid`/`dashed` 만 본다.** aAcc(전체 정확도)도 무의미하다
— 전부 background 로 찍어도 99.5% 가 나오기 때문이다.

### 7.2 이어서 학습

```bash
... ./tools/dist_train.sh <config> 4 --work-dir work_dirs/b0_skyscapes_20k \
  --resume-from work_dirs/b0_skyscapes_20k/iter_10000.pth
```

### 7.3 test 평가

```bash
PATH=/home/hsjeong/miniconda3/envs/segformer_cu111/bin:$PATH PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh \
  local_configs/segformer/B0/segformer.b0.1024x1024.skyscapes.20k.py \
  work_dirs/b0_skyscapes_20k/iter_10000.pth 4 --eval mIoU
```

---

## 8. B0 1차 학습 결과 (2026-08-05)

### 8.1 실행 설정

7절 기본 명령에 다음 `--options` 를 붙여 실행했다.

```text
data.samples_per_gpu=3  runner.max_iters=10000  optimizer.lr=0.00009
lr_config.warmup_iters=750  evaluation.interval=500  checkpoint_config.interval=500
```

| 항목 | 값 |
|---|---|
| 유효 배치 | 3 × GPU 4장 = **12** |
| 학습량 | 10,000 iter ≈ 540 에폭 (타일 222장 기준) |
| lr | 9e-5 (poly, warmup 750) |
| 소요 시간 | 약 70분 (0.41 s/iter) |
| work_dir | `work_dirs/b0_skyscapes_20k` |

### 8.2 val 추이 (타일 48장, 500 iter 간격 20회)

| iter | solid IoU | solid Acc | dashed IoU | dashed Acc | 전경평균 |
|---:|---:|---:|---:|---:|---:|
| 500 | 16.68 | 32.2 | 26.10 | 80.2 | 21.39 |
| 1000 | 28.29 | 53.5 | 24.31 | 87.5 | 26.30 |
| 1500 | 28.53 | 51.6 | 24.52 | 88.9 | 26.52 |
| 2000 | 27.59 | 49.2 | 29.22 | 85.7 | 28.41 |
| 2500 | 28.72 | 56.4 | 29.17 | 84.7 | 28.95 |
| 3000 | 29.85 | 48.0 | 30.72 | 84.0 | 30.29 |
| 3500 | 27.24 | 53.9 | 28.39 | 87.8 | 27.81 |
| 4000 | 30.53 | 47.5 | 32.88 | 84.4 | 31.71 |
| 4500 | 30.29 | 46.5 | 34.48 | 82.5 | 32.38 |
| 5000 | 31.74 | 54.5 | 31.36 | 85.4 | 31.55 |
| 5500 | **32.59** | 48.7 | 31.76 | 85.5 | 32.18 |
| 6000 | 31.18 | 46.9 | 33.69 | 82.4 | 32.44 |
| 6500 | 30.63 | 44.4 | 35.09 | 83.1 | 32.86 |
| **7000** | 31.88 | 48.1 | 35.41 | 82.3 | **33.64** |
| 7500 | 32.17 | 55.8 | 32.77 | 85.0 | 32.47 |
| 8000 | 31.17 | 48.2 | 35.22 | 81.7 | 33.20 |
| 8500 | 30.31 | 44.2 | 35.12 | 81.5 | 32.71 |
| 9000 | 30.06 | 43.2 | **35.84** | 80.8 | 32.95 |
| 9500 | 30.74 | 45.2 | 34.98 | 82.6 | 32.86 |
| 10000 | 31.01 | 46.0 | 35.25 | 82.1 | 33.13 |

500 iter 단위 변동폭이 ±3 정도로 크다. val 이 타일 48장(전경 25장)뿐이라 평가 노이즈가
크므로 **단일 평가로 추세를 판단하면 안 된다.** 1000 iter 단위로 보면 21 → 28 → 32 → 33 으로
상승 후 6,500 부근부터 33 대 횡보다.

### 8.3 test 결과 (타일 48장, 학습·검증에 한 번도 쓰이지 않음)

| 체크포인트 | solid IoU | solid Acc | dashed IoU | dashed Acc | 전경평균 |
|---|---:|---:|---:|---:|---:|
| `iter_7000.pth` | 40.90 | 53.41 | 36.88 | 87.32 | 38.89 |
| `iter_10000.pth` | **41.11** | 52.29 | **38.50** | 86.09 | **39.81** |

background IoU 는 99.7, aAcc 는 99.7 로 무의미하다 (01단계 3.1절 운영 규칙).

### 8.4 결과 해석 — 주의할 점 3가지

**① test 가 val 보다 6점 높다 (38.9~39.8 vs 33.1~33.6).**
보통은 반대여야 한다. 원인은 split 구성에 있다. test 는 `4K0G0080`(전경 560픽셀,
사실상 빈 이미지)과 `4K0G0130` 2장인데, mmseg 가 데이터셋 전체 혼동행렬을 누적해 IoU 를
계산하므로 **실질적으로 `4K0G0130` 한 장의 점수**다 (02단계 4.4절). 표본이 1장이라
이 6점 차이를 "일반화 성능이 더 좋다"로 읽으면 안 된다.

**② solid/dashed 의 우열이 val 과 test 에서 뒤집힌다.**

| | solid IoU | dashed IoU |
|---|---:|---:|
| val | 31.0 | 35.3 (dashed 우세) |
| test | 41.1 | 38.5 (**solid 우세**) |

학습 중에는 val 만 보고 "solid 가 구조적으로 약하다"고 판단했으나, test 에서 뒤집혔다.
**어느 클래스가 더 어려운지는 확정되지 않았다.** val 2장 / test 실질 1장이라 두 추정 모두
신뢰구간이 매우 넓다. 원본 이미지가 10장뿐인 데이터셋의 근본적 한계다.

**③ 다만 Acc(재현율) 격차는 val·test 양쪽에서 일관된다.**

| | solid Acc | dashed Acc |
|---|---:|---:|
| val (20회 전체) | 32~56% | 80~89% |
| test | 52~53% | 86~87% |

**밴드가 한 번도 겹치지 않는다.** 이건 노이즈로 설명되지 않는 실제 차이다.
solid 는 절반 가까이 놓치고(재현율 낮음) 대신 찾은 것은 정확하며(IoU 41 > Acc 53 대비 양호),
dashed 는 대부분 찾지만(재현율 86%) 실제보다 두껍게 칠해 오탐이 많다(IoU 38.5 ≪ Acc 86).
→ **solid 는 미검출, dashed 는 과검출**이 각각의 병목이다.

### 8.5 검증하지 않은 위험 요소

원본 파일명이 `2012-04-26-Muenchen-Tunnel_4K0G0010 ~ 0160` 으로, **한 번의 비행에서 연속
촬영된 프레임**으로 보인다. 프레임 번호가 가까운 이미지끼리 지상 영역이 겹칠 가능성이 있다.
겹친다면 train 의 `4K0G0110` 과 test 의 `4K0G0130` 이 같은 지면을 일부 공유해 test 점수가
부풀려졌을 수 있다. **아직 확인하지 않았다.** 다음 작업 전에 원본 이미지의 지리적 중첩을
확인해 볼 가치가 있다.

### 8.6 시각화 및 혼동행렬 (`UTIL/skyscapes_visualize_test.py`)

`원본 | 정답 GT | 예측` 3분할 이미지를 만든다. 출력은
`work_dirs/b0_skyscapes_20k/vis_test/` 아래 `tiles/`(48장, 등배율)와
`full/`(2장, 타일을 원본 좌표로 되붙여 5616×3744 복원 후 축소).

```bash
cd /home/hsjeong/workspace/vision-seg/SegFormer/source
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
/home/hsjeong/miniconda3/envs/segformer_cu111/bin/python \
  /home/hsjeong/workspace/vision-seg/SegFormer/scripts/skyscapes_visualize_test.py

# 다른 체크포인트/split, 크기 조절
  ... skyscapes_visualize_test.py --checkpoint work_dirs/.../iter_7000.pth --split val
  ... skyscapes_visualize_test.py --full-width 2400 --no-tiles
```

구현상 주의 2가지:

1. **한글 제목에는 CJK 폰트가 필요하다.** matplotlib 동봉 DejaVuSans 로는 □□ 로 깨진다.
   `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` 의 **index=1** 이 한국어다.
2. **전체보기(`full/`)는 선을 굵게 그린다.** 5616×3744 를 화면 폭에 맞춰 축소하면 폭 2~4px
   차선이 사라진다. 축소 배율의 역수만큼 팽창시킨다(현재 9px). 타일 보기는 등배율이라
   팽창 없이 실제 예측 두께 그대로다. **전체보기의 선 굵기로 정밀도를 판단하면 안 된다.**

스크립트가 재계산한 IoU 가 `dist_test` 결과와 정확히 일치했다
(solid 41.11 / dashed 38.50) → 추론 경로가 올바름을 교차 확인.

#### 혼동행렬 (test, ignore 제외, 단위 픽셀)

| 정답 \ 예측 | background | solid | dashed | GT 합계 |
|---|---:|---:|---:|---:|
| background | 41,845,085 | 23,602 | 51,165 | 41,919,852 |
| solid | **39,286** | 46,461 | 3,109 | 88,856 |
| dashed | 5,535 | 570 | 37,795 | 43,900 |

| | 재현율 (행 정규화) | 정밀도 (열 정규화) |
|---|---|---|
| solid | bg 44.21% / **solid 52.29%** / dashed 3.50% | bg 33.41% / **solid 65.78%** / dashed 0.81% |
| dashed | bg 12.61% / solid 1.30% / **dashed 86.09%** | bg 55.57% / solid 3.38% / **dashed 41.05%** |

**핵심 — 실선/점선을 헷갈리는 게 아니다.** 시각화 이미지만 보면 GT 의 긴 파란 실선이
예측에서 빨간 점선으로 조각나 보이지만, 수치는 다르다. solid → dashed 오분류는 **3.50%**,
dashed → solid 는 **1.30%** 로 둘 다 미미하다. 두 클래스의 구분 자체는 잘 하고 있다.

실제 병목은 **전경 대 배경**이다.

- **solid 는 44.21% 를 background 로 놓친다** (미검출). 대신 검출한 것의 정밀도는 65.78% 로 준수.
- **dashed 는 재현율 86.09% 로 잘 찾지만 정밀도가 41.05% 뿐이다.** 예측한 dashed 픽셀의
  **55.57% 가 실제로는 background** — 실제보다 두껍게/넓게 칠한다.

→ 개선 방향이 클래스별로 다르다. solid 는 **놓치는 것**을, dashed 는 **번지는 것**을 잡아야 한다.
클래스 가중치를 양쪽 다 올리는 식의 대응은 dashed 과검출을 악화시킨다.

### 8.7 다음 개선 방향

8.6 의 혼동행렬에 따라 클래스별로 처방이 다르다.

| 방법 | 겨냥하는 문제 | 근거 | 비용 |
|---|---|---|---|
| **B2 로 교체** | 둘 다 | 백본 3.7M → 27M. config 준비 완료(`B2/segformer.b2.1024x1024.skyscapes.20k.py`, batch 1 이므로 lr 을 6e-5 부근으로 낮출 것) | 재학습 |
| dashed 가중치 **하향** (25.17 → 15 부근) | dashed 과검출 (정밀도 41%) | 예측 dashed 의 55.57% 가 배경. 가중치가 과했다 | 재학습 |
| solid 가중치 상향 (17.15 → 25 부근) | solid 미검출 (배경으로 44% 유실) | 단, dashed 를 함께 올리면 역효과 | 재학습 |
| 타일 1024 → 1536 | solid 미검출 | 실선은 긴 맥락이 필요 | 재학습 + 타일 재생성 |
| **평가 신뢰도 확보** | 전부 | **val 2장 / test 실질 1장이 가장 큰 문제. 교차검증(leave-one-out)이 수치 안정화에 가장 효과적** | 10회 학습 |

가장 하지 말아야 할 것은 **두 클래스 가중치를 함께 올리는 것**이다. solid 미검출만 보고
전경 가중치를 일괄 상향하면 이미 정밀도 41% 인 dashed 가 더 번진다.

---

## 9. B2 학습 결과 (2026-08-05)

### 9.1 실행 설정

B0 대비 바꾼 것은 **백본과 그에 따라 강제되는 배치·lr·iter 뿐**이다. 클래스 가중치,
데이터 파이프라인, 타일은 전부 동일하게 두었다. 백본 용량이 원인인지를 분리해서 보려면
한 번에 한 가지만 바꿔야 하기 때문이다.

```bash
cd /home/hsjeong/workspace/vision-seg/SegFormer/source

PATH=/home/hsjeong/miniconda3/envs/segformer_cu111/bin:$PATH \
PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./tools/dist_train.sh \
  local_configs/segformer/B2/segformer.b2.1024x1024.skyscapes.20k.py 4 \
  --work-dir work_dirs/b2_skyscapes_30k \
  --seed 0 --deterministic \
  --options runner.max_iters=30000 optimizer.lr=0.00003 \
            lr_config.warmup_iters=2250 \
            evaluation.interval=1000 checkpoint_config.interval=1000
```

| 항목 | B0 | **B2** |
|---|---|---|
| 파라미터 | 3.72M | **27.35M** (7.3배) |
| `samples_per_gpu` | 3 | **1** (2 는 OOM) |
| 유효 배치 | 12 | **4** |
| lr | 9e-5 | **3e-5** (배치 1/3 에 맞춰 선형 축소) |
| warmup | 750 | 2,250 (동일 비율 7.5%) |
| max_iters | 10,000 | **30,000** (동일 샘플 수 120,000 맞춤) |
| 에폭 환산 | 약 540 | 약 541 |
| 속도 | 0.41 s/iter | 0.39 s/iter |
| 소요 시간 | 70분 | **약 195분** |
| GPU 당 VRAM | 7.4GB | 10.0GB |

### 9.2 val 추이 (30회)

| iter | solid IoU | solid Acc | dashed IoU | dashed Acc | 전경평균 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 21.34 | 37.4 | 25.80 | 83.8 | 23.57 |
| 2000 | 24.23 | 34.0 | 34.02 | 81.1 | 29.12 |
| 3000 | 27.53 | 35.8 | 33.67 | 82.5 | 30.60 |
| 4000 | 34.55 | 50.0 | 34.69 | 84.9 | 34.62 |
| 5000 | 30.74 | 44.1 | 34.15 | 85.3 | 32.45 |
| 6000 | 32.50 | 46.5 | 35.01 | 82.7 | 33.75 |
| 7000 | 32.76 | 48.4 | 34.97 | 82.0 | 33.86 |
| 8000 | 30.80 | 43.3 | 34.75 | 83.9 | 32.77 |
| 9000 | 32.20 | 44.0 | 38.59 | 78.1 | 35.40 |
| 10000 | 31.55 | 44.1 | 36.64 | 83.2 | 34.09 |
| 11000 | 26.89 | 33.9 | 37.91 | 83.0 | 32.40 |
| 12000 | 28.75 | 36.8 | 40.94 | 77.4 | 34.84 |
| 13000 | 33.90 | 47.1 | 40.74 | 82.4 | 37.32 |
| **14000** | **36.63** | 53.8 | 38.72 | 85.0 | **37.67** |
| 15000 | 33.16 | 45.6 | 37.64 | 85.2 | 35.40 |
| 16000 | 30.89 | 38.6 | 40.13 | 83.5 | 35.51 |
| 17000 | 32.48 | 42.2 | **41.38** | 78.4 | 36.93 |
| 18000 | 33.09 | 44.7 | 39.48 | 82.5 | 36.28 |
| 19000 | 33.35 | 44.0 | 41.17 | 82.6 | 37.26 |
| 20000 | 34.56 | 47.9 | 39.20 | 83.9 | 36.88 |
| 21000 | 35.72 | 48.9 | 39.60 | 84.3 | 37.66 |
| 22000 | 31.01 | 39.8 | 40.78 | 83.0 | 35.90 |
| 23000 | 32.52 | 44.7 | 39.22 | 83.2 | 35.87 |
| 24000 | 30.34 | 38.9 | 40.55 | 81.2 | 35.45 |
| 25000 | 30.88 | 40.1 | 40.47 | 81.4 | 35.67 |
| 26000 | 32.33 | 42.5 | 40.17 | 83.0 | 36.25 |
| 27000 | 33.06 | 43.7 | 40.42 | 82.8 | 36.74 |
| 28000 | 32.80 | 43.6 | 40.46 | 82.7 | 36.63 |
| 29000 | 32.50 | 41.9 | 41.04 | 82.5 | 36.77 |
| 30000 | 31.06 | 39.4 | 40.65 | 82.5 | 35.85 |

13,000 부근까지 상승한 뒤 **35~37 대에서 수렴**했다. 후반 15,000 iter 는 사실상 성능 향상이
없었다. 다음 학습에서는 15,000~20,000 iter 로 줄여도 무방하다.

### 9.3 test 결과 — B0 대비 +6.6

| 모델 | 체크포인트 | solid IoU | solid Acc | dashed IoU | dashed Acc | **전경평균** |
|---|---|---:|---:|---:|---:|---:|
| B0 | iter_10000 | 41.11 | 52.29 | 38.50 | 86.09 | 39.81 |
| **B2** | **iter_14000** | **51.66** | 65.34 | 41.12 | 88.24 | **46.39** |
| B2 | iter_17000 | 49.27 | 60.45 | 43.52 | 84.92 | **46.40** |
| B2 | iter_30000 | 48.56 | 57.39 | 44.01 | 86.44 | 46.29 |

B2 세 체크포인트가 46.3~46.4 로 **사실상 동률**이다. 선택 기준은 성능이 아니라 특성이다.

- `iter_14000` — solid 51.66 (최고). solid 재현율 65.3% 로 실선 미검출이 가장 적다. **권장**
- `iter_30000` — dashed 44.01 (최고). 두 클래스가 가장 균형적

B0 의 val 최고점(7,000)보다 최종(10,000)이 test 에서 나았던 것과 달리, B2 는 val 최고점
(14,000)이 test 에서도 최상위였다. 다만 차이가 0.1 이하라 어느 쪽이든 무방하다.

### 9.4 혼동행렬 (B2 iter_14000, test)

| 정답 \ 예측 | background | solid | dashed | GT 합계 |
|---|---:|---:|---:|---:|
| background | 41,848,302 | 23,167 | 48,383 | 41,919,852 |
| solid | 28,874 | 58,057 | 1,925 | 88,856 |
| dashed | 4,801 | 360 | 38,739 | 43,900 |

**B0 대비 변화 — solid 미검출이 크게 줄었다.**

| 지표 | B0 | B2 | 변화 |
|---|---:|---:|---|
| solid 재현율 | 52.29% | **65.34%** | **+13.1** |
| solid 를 background 로 놓침 | 44.21% | **32.50%** | **-11.7** |
| solid 정밀도 | 65.78% | **71.16%** | +5.4 |
| dashed 재현율 | 86.09% | 88.24% | +2.2 |
| dashed 정밀도 | 41.05% | **43.50%** | +2.5 |
| dashed 를 background 로 오검출 | 55.57% | 54.33% | -1.2 |

**8.7 절의 예상과 결과가 달랐다.** 개선의 대부분은 **solid 미검출 완화**에서 나왔고
(재현율 +13.1), dashed 과검출은 거의 그대로다 (정밀도 41.05 → 43.50, +2.5 뿐).

즉 **백본 용량은 solid 미검출 문제를 푸는 수단이었고, dashed 과검출은 별개의 문제다.**
dashed 정밀도가 여전히 43.5% — 예측한 dashed 픽셀의 절반 이상이 배경 — 이므로,
8.7 절의 "dashed 가중치 하향(25.17 → 15 부근)"은 B2 위에서 여전히 유효한 다음 수순이다.

### 9.5 시각화

```bash
cd /home/hsjeong/workspace/vision-seg/SegFormer/source
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
/home/hsjeong/miniconda3/envs/segformer_cu111/bin/python \
  /home/hsjeong/workspace/vision-seg/SegFormer/scripts/skyscapes_visualize_test.py \
  --config local_configs/segformer/B2/segformer.b2.1024x1024.skyscapes.20k.py \
  --checkpoint work_dirs/b2_skyscapes_30k/iter_14000.pth \
  --out-dir work_dirs/b2_skyscapes_30k/vis_test
```

출력: `work_dirs/b2_skyscapes_30k/vis_test/{tiles,full}/`.
스크립트가 재계산한 IoU 가 `dist_test` 와 일치(51.66 / 41.12)해 추론 경로를 교차 확인했다.

### 9.6 체크포인트 정리

B2 체크포인트는 1개 **329MB**(B0 의 7.3배, AdamW 옵티마이저 상태 포함)라 30개면 9.9GB 다.
`iter_14000` / `iter_17000` / `iter_30000` 3개만 남기고 27개(8.9GB)를 삭제했다.
`latest.pth` 심볼릭링크는 `iter_30000.pth` 를 가리키며 유효하다.

### 9.7 다음 수순

| 우선순위 | 방법 | 근거 |
|---:|---|---|
| 1 | **dashed 가중치 하향** (25.17 → 15 부근), B2 기반 | 정밀도가 43.5% 로 여전히 최대 병목. B2 로도 해결되지 않았다 |
| 2 | 평가 신뢰도 확보 (leave-one-out 교차검증) | val 2장 / test 실질 1장. 모든 수치의 신뢰구간이 넓다 |
| 3 | max_iters 15,000~20,000 으로 단축 | 후반 15,000 iter 는 성능 향상이 없었다 (9.2절) |
| 4 | B4/B5 시도 | B0→B2 에서 +6.6 을 얻었으나 수확체감 여부는 미확인 |
