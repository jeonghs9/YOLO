#!/usr/bin/env bash
# NVlabs SegFormer 전용 conda 환경 구축 (segformer_cu111)
#
# ── 사전 검증 완료 사항 ────────────────────────────────────────────────────
#   - mmcv_full-1.3.0-cp38-cp38-manylinux1_x86_64.whl (cu111/torch1.8.0 인덱스) 사전 빌드 휠 존재
#     -> CUDA 확장 소스 컴파일 불필요. 이 저장소 설치의 최대 실패 요인이 제거된다.
#   - torch-1.8.1+cu111-cp38-cp38-linux_x86_64.whl 존재
#   - NVIDIA 드라이버 535.230.02 >= 450.80.02 (cu111 최소 요구) -> V100(sm_70) 구동 가능
#
# ── 반드시 필요한 격리 처리 (초판에서 사고가 났던 지점) ────────────────────
#   `conda run -n <env> pip install` 을 쓰면 안 된다.
#   이 서버에는 /home/hsjeong/.local/lib/python3.8/site-packages 에 670개 패키지를 가진
#   공유 user-site 가 있고, python3.8 env 의 sys.path 에서 env 자신의 site-packages 보다
#   앞에 온다. pip 이 거기 있던 torch/mmcv/numpy 등을 "기존 설치"로 인식해 제자리에서
#   교체해 버리므로, env 는 텅 빈 채 공유 환경만 파괴된다.
#   -> PYTHONNOUSERSITE=1 로 user-site 를 sys.path 에서 완전히 제거한 뒤 env 의 python 을
#      직접 호출한다. 또한 conda env config vars 로 영구 등록해 activate 시에도 유지한다.
#
# ── 알려진 함정 3개 선제 처리 ──────────────────────────────────────────────
#   1) mmseg/core/evaluation/metrics.py:89-92 가 np.float 사용
#      -> numpy>=1.24 에서 AttributeError. numpy 1.23.5 고정
#   2) mmseg 7개 모듈이 최상단에서 `from IPython import embed`
#      -> ipython 필수. 또한 IPython 은 pexpect 를 요구하므로 함께 설치
#   3) mmcv 1.3.0 의 Config.pretty_text 가 yapf FormatCode(..., verify=True) 호출
#      -> yapf 0.40 에서 verify 인자가 삭제됨. tools/train.py 가 시작 직후 cfg.dump() 를
#         하므로 학습이 첫 iter 도 못 돌고 TypeError 로 죽는다. yapf 0.32.0 고정
#
# 사용법:
#   bash /home/hsjeong/workspace/vision-seg/SegFormer/scripts/setup_segformer_env.sh

set -euo pipefail

ENV_NAME="segformer_cu111"
CONDA="/home/hsjeong/miniconda3/bin/conda"
ENV_PREFIX="/home/hsjeong/miniconda3/envs/${ENV_NAME}"
PY="${ENV_PREFIX}/bin/python"
SEGFORMER_DIR="/home/hsjeong/workspace/vision-seg/SegFormer/source"

TORCH_IDX="https://download.pytorch.org/whl/torch_stable.html"
MMCV_IDX="https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html"

# 이 스크립트 전체에서 user-site 를 차단한다
export PYTHONNOUSERSITE=1

echo "### 1/7 기존 환경 제거 후 재생성 (python 3.8)"
"$CONDA" env remove -n "$ENV_NAME" -y 2>/dev/null || true
rm -rf "$ENV_PREFIX"
"$CONDA" create -n "$ENV_NAME" python=3.8 -y

echo "### 2/7 PYTHONNOUSERSITE=1 영구 등록 (activate 시에도 user-site 차단)"
"$CONDA" env config vars set PYTHONNOUSERSITE=1 -n "$ENV_NAME"

echo "### 3/7 격리 확인 - user-site 가 sys.path 에 없어야 한다"
"$PY" - <<'PY'
import sys, site
bad = [p for p in sys.path if ".local/lib/python" in p]
print("sys.path 내 user-site:", bad if bad else "없음 (정상)")
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
assert not bad, "user-site 가 아직 sys.path 에 있다. 설치를 중단한다."
PY

echo "### 4/7 torch 1.8.1+cu111 / torchvision 0.9.1+cu111"
"$PY" -m pip install --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f "$TORCH_IDX"

echo "### 5/7 mmcv-full 1.3.0 (사전 빌드 휠) + 부가 패키지"
"$PY" -m pip install --no-cache-dir mmcv-full==1.3.0 -f "$MMCV_IDX"
# timm 0.3.2 는 --no-deps. 의존성 해석이 torch 를 최신으로 끌어올리는 것을 막는다.
"$PY" -m pip install --no-cache-dir --no-deps timm==0.3.2
"$PY" -m pip install --no-cache-dir \
    "numpy==1.23.5" \
    "opencv-python==4.5.1.48" \
    ipython pexpect attrs terminaltables matplotlib prettytable addict "yapf==0.32.0" packaging

echo "### 6/7 SegFormer(mmseg 0.11.0) editable 설치 - --no-deps 로 numpy 재업그레이드 차단"
"$PY" -m pip install --no-cache-dir --no-deps -e "$SEGFORMER_DIR"

echo "### 7/7 스모크 테스트"
# SegFormerHead.linear_fuse 가 norm_cfg 를 무시하고 SyncBN 을 하드코딩한다
# (mmseg/models/decode_heads/segformer_head.py:59). SyncBN 은 train 모드에서
# torch.distributed 초기화를 요구하므로, 학습 스텝 검증을 위해 단일 프로세스 그룹을 띄운다.
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29517 RANK=0 WORLD_SIZE=1
"$PY" - <<'PY'
import sys, os
bad = [p for p in sys.path if ".local/lib/python" in p]
assert not bad, f"user-site 오염: {bad}"

import numpy, torch, mmcv, timm
print("python     :", sys.executable)
print("numpy      :", numpy.__version__, "@", os.path.dirname(numpy.__file__))
print("torch      :", torch.__version__, "@", os.path.dirname(torch.__file__))
print("torch cuda :", torch.version.cuda)
print("cuda avail :", torch.cuda.is_available())
print("gpu count  :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0       :", torch.cuda.get_device_name(0))
print("mmcv       :", mmcv.__version__)
print("timm       :", timm.__version__)

# 설치 위치가 env 안인지 확인
for m in (numpy, torch, mmcv, timm):
    assert "/envs/segformer_cu111/" in m.__file__, f"{m.__name__} 이 env 밖에 있다: {m.__file__}"
print("설치 위치  : 전부 env 내부 확인")

# np.float (numpy 1.24+ 이면 여기서 터짐)
print("np.float ok:", numpy.zeros((3,), dtype=numpy.float).dtype)

# mmcv CUDA 확장 실제 로드
from mmcv.ops import RoIAlign  # noqa: F401
print("mmcv ops   : import OK")

# mmseg 전체 임포트 (mmcv 버전 assert + IPython import 통과 여부)
import mmseg
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset  # noqa: F401
print("mmseg      :", mmseg.__version__)

# SyncBN(하드코딩) 때문에 단일 프로세스 그룹을 초기화한다
import torch.distributed as dist
dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
torch.cuda.set_device(0)

# SegFormer 백본/헤드 실제 구성 (background/solid/dashed 3클래스)
from mmcv.utils import Config
cfg = dict(
    type='EncoderDecoder',
    backbone=dict(type='mit_b0', style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
model = build_segmentor(Config(dict(model=cfg)).model).cuda().train()
print("params     : %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

x = torch.randn(2, 3, 512, 512).cuda()
meta = [dict(ori_shape=(512, 512, 3), img_shape=(512, 512, 3), pad_shape=(512, 512, 3))] * 2
gt = torch.randint(0, 3, (2, 1, 512, 512)).long().cuda()

# 학습 스텝: 손실 + 역전파
losses = model.forward_train(x, meta, gt)
loss = sum(v for k, v in losses.items() if 'loss' in k)
loss.backward()
n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print("train loss :", round(float(loss), 4))
print("gradient   : %d / %d 파라미터에 grad 생성" % (n_grad, sum(1 for _ in model.parameters())))
print("peak VRAM  : %.2f GB (batch 2, 512x512)" % (torch.cuda.max_memory_allocated() / 1024**3))

# 추론
model.eval()
with torch.no_grad():
    out = model.encode_decode(x[:1], meta[:1])
assert tuple(out.shape) == (1, 3, 512, 512), out.shape
print("eval out   :", tuple(out.shape))

dist.destroy_process_group()
print("\n[SMOKE TEST PASSED]")
PY

echo
echo "환경 활성화:  conda activate ${ENV_NAME}"
