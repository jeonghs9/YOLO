#!/usr/bin/env bash
# ~/.local/lib/python3.8/site-packages (공유 user-site) 원복 스크립트
#
# 배경: setup_segformer_env.sh 초판이 `conda run -n segformer_cu111 pip install`을 사용했는데,
#       해당 env의 sys.path에서 user-site가 env site-packages보다 앞에 있어
#       pip이 user-site의 기존 설치를 제자리에서 교체해 버렸다.
#       env site-packages 에는 아무것도 설치되지 않았고, 공유 user-site 만 다운그레이드되었다.
#
# 원복 대상 (setup 로그 기준, "Successfully uninstalled" 로 확인된 것만):
#   torch          1.10.0     <- 1.8.1+cu111
#   torchvision    0.11.1     <- 0.9.1+cu111
#   mmcv-full      1.7.1      <- 1.3.0
#   timm           0.6.7      <- 0.3.2
#   numpy          1.24.4     <- 1.23.5
#   opencv-python  4.9.0.80   <- 4.5.1.48
#   mmsegmentation (editable egg-link) -> 신규 생성된 것이므로 제거
#
# torch 1.10.0 은 PyPI 기본 빌드(cu102)였다. 로그가 로컬 버전 접미사 없이 "1.10.0" 으로
# 기록했기 때문이다. 따라서 mmcv-full 1.7.1 도 cu102/torch1.10.0 사전 빌드 휠을 쓴다.
#
# 모든 설치는 --no-deps 로 한다. 나머지 669개 패키지를 건드리지 않기 위해서다.
#
# 사용법:
#   bash /home/hsjeong/workspace/vision-seg/SegFormer/scripts/restore_user_site_py38.sh

set -euo pipefail

PY="/home/hsjeong/miniconda3/envs/segformer_cu111/bin/python"   # python3.8 이면 무엇이든 동일한 user-site 를 씁니다
USER_SP="/home/hsjeong/.local/lib/python3.8/site-packages"
MMCV_IDX="https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html"

echo "### 0/4 대상 user-site: $USER_SP"
"$PY" -c "import site; print('user site =', site.getusersitepackages())"

echo "### 1/4 SegFormer editable(mmsegmentation) 제거"
"$PY" -m pip uninstall -y mmsegmentation || true

echo "### 2/4 torch 1.10.0 / torchvision 0.11.1 복구"
"$PY" -m pip install --user --no-deps --no-cache-dir \
    torch==1.10.0 torchvision==0.11.1

echo "### 3/4 mmcv-full 1.7.1 복구 (cu102/torch1.10.0 사전 빌드 휠)"
"$PY" -m pip install --user --no-deps --no-cache-dir \
    mmcv-full==1.7.1 -f "$MMCV_IDX"

echo "### 4/4 timm 0.6.7 / numpy 1.24.4 / opencv-python 4.9.0.80 복구"
"$PY" -m pip install --user --no-deps --no-cache-dir \
    timm==0.6.7 numpy==1.24.4 opencv-python==4.9.0.80

echo
echo "### 검증"
PYTHONNOUSERSITE= "$PY" - <<'PY'
import importlib.metadata as md
want = {
    "torch": "1.10.0",
    "torchvision": "0.11.1",
    "mmcv-full": "1.7.1",
    "timm": "0.6.7",
    "numpy": "1.24.4",
    "opencv-python": "4.9.0.80",
}
ok = True
for name, exp in want.items():
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        got = "MISSING"
    mark = "OK " if got == exp else "NG "
    if got != exp:
        ok = False
    print(f"  {mark} {name:16s} expected={exp:12s} got={got}")
try:
    md.version("mmsegmentation")
    print("  NG  mmsegmentation 가 아직 남아 있음")
    ok = False
except md.PackageNotFoundError:
    print("  OK  mmsegmentation 제거됨")
print("\n[RESTORE " + ("PASSED]" if ok else "FAILED]"))
PY
