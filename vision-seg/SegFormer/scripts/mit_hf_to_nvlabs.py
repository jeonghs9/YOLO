#!/usr/bin/env python3
"""HuggingFace `nvidia/mit-b*` 체크포인트를 NVlabs SegFormer 포크 형식으로 변환한다.

NVlabs README 의 google drive 링크에서 `mit_b*.pth` 를 받을 수 없을 때 쓴다.
HuggingFace `nvidia/mit-b{0..5}` 는 동일한 ImageNet-1k 사전학습 MiT 백본이며,
구현체가 달라 state_dict 키 이름만 다르다.

핵심 차이 두 가지:
  1. 키 이름 체계가 전부 다르다 (`segformer.encoder.block.0.0.*` ↔ `block1.0.*`).
  2. HF 는 attention 의 key/value 를 별도 Linear 로 두고, NVlabs 는 하나의
     `kv` Linear (out_features = 2*dim) 로 합쳐 둔다. 따라서 concat 이 필요하다.
     NVlabs Attention.forward 의 `kv(x).reshape(B, -1, 2, heads, hd)` 는 채널을
     row-major 로 펼치므로 앞쪽 dim 채널이 k, 뒤쪽 dim 채널이 v 다.
     → kv.weight = cat([key.weight, value.weight], dim=0)

HF 의 `classifier.*` (ImageNet 1000클래스 분류기) 는 세그멘테이션에서 쓰지 않으므로 버린다.
NVlabs `MixVisionTransformer` 도 분류 head 가 주석 처리되어 있어 애초에 자리가 없다.

주의: NVlabs `MixVisionTransformer.init_weights()` 는 `load_checkpoint(..., strict=False)` 라
키가 하나도 안 맞아도 에러 없이 조용히 랜덤 초기화로 학습이 진행된다. 그래서 이 스크립트는
변환 결과를 `load_state_dict(strict=True)` 로 직접 검증한 뒤에만 저장한다.

반드시 segformer_cu111 환경의 python 으로 실행해야 한다 (mmseg 검증에 필요).
"""

import argparse
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import torch

HF_URL = "https://huggingface.co/nvidia/mit-{model}/resolve/main/pytorch_model.bin"
CACHE_DIR = Path("/home/hsjeong/tmp/mit_hf")
OUT_DIR = Path(
    "/home/hsjeong/workspace/vision-seg/SegFormer/source/pretrained"
)
ALL_MODELS = ["b0", "b1", "b2", "b3", "b4", "b5"]

# HF 블록 내부 경로 -> NVlabs 블록 내부 경로. key/value 는 합쳐야 해서 별도 처리한다.
BLOCK_RENAME = {
    "layer_norm_1": "norm1",
    "layer_norm_2": "norm2",
    "attention.self.query": "attn.q",
    "attention.self.sr": "attn.sr",
    "attention.self.layer_norm": "attn.norm",
    "attention.output.dense": "attn.proj",
    "mlp.dense1": "mlp.fc1",
    "mlp.dense2": "mlp.fc2",
    "mlp.dwconv.dwconv": "mlp.dwconv.dwconv",
}


def download(model: str, force: bool = False) -> Path:
    """HF 체크포인트를 받아 캐시 경로를 돌려준다."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = CACHE_DIR / f"mit-{model}_hf.bin"
    if dst.exists() and not force:
        print(f"  캐시 사용: {dst} ({dst.stat().st_size / 1e6:.1f}MB)")
        return dst
    url = HF_URL.format(model=model)
    print(f"  다운로드: {url}")
    subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(dst), url], check=True)
    print(f"  저장 완료: {dst} ({dst.stat().st_size / 1e6:.1f}MB)")
    return dst


def convert(hf_sd: dict) -> "OrderedDict[str, torch.Tensor]":
    """HF state_dict -> NVlabs state_dict.

    모르는 키를 만나면 조용히 버리지 않고 예외를 던진다. 사전학습 가중치가
    일부만 실리는 사고는 학습이 다 끝난 뒤에야 성능으로 드러나기 때문이다.
    """
    out = OrderedDict()
    kv_parts = {}  # (stage, blk, 'weight'|'bias') -> {'key': t, 'value': t}

    for k, v in hf_sd.items():
        if k.startswith("classifier."):
            continue  # ImageNet 분류기 — 세그멘테이션 백본에는 자리가 없다

        m = re.fullmatch(r"segformer\.encoder\.patch_embeddings\.(\d+)\.(proj|layer_norm)\.(weight|bias)", k)
        if m:
            idx, mod, wb = int(m.group(1)), m.group(2), m.group(3)
            out[f"patch_embed{idx + 1}.{'proj' if mod == 'proj' else 'norm'}.{wb}"] = v
            continue

        m = re.fullmatch(r"segformer\.encoder\.layer_norm\.(\d+)\.(weight|bias)", k)
        if m:
            out[f"norm{int(m.group(1)) + 1}.{m.group(2)}"] = v
            continue

        m = re.fullmatch(r"segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)\.(weight|bias)", k)
        if m:
            stage, blk, inner, wb = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            if inner in ("attention.self.key", "attention.self.value"):
                which = inner.rsplit(".", 1)[1]
                kv_parts.setdefault((stage, blk, wb), {})[which] = v
                continue
            if inner in BLOCK_RENAME:
                out[f"block{stage + 1}.{blk}.{BLOCK_RENAME[inner]}.{wb}"] = v
                continue

        raise KeyError(f"매핑 규칙이 없는 HF 키: {k}")

    # key/value 를 kv 하나로 합친다 (앞쪽이 k, 뒤쪽이 v)
    for (stage, blk, wb), parts in kv_parts.items():
        missing = {"key", "value"} - set(parts)
        if missing:
            raise KeyError(f"block{stage + 1}.{blk} 의 {wb} 에서 {missing} 누락")
        out[f"block{stage + 1}.{blk}.attn.kv.{wb}"] = torch.cat([parts["key"], parts["value"]], dim=0)

    return out


def check_attention_math(hf_sd: dict, nv_sd: dict) -> None:
    """k/v concat 순서가 맞는지 수치로 확인한다.

    표준 MHA(softmax(q@k^T * scale) @ v)를 직접 계산한 값과, 변환된 가중치를 실은
    NVlabs Attention 모듈의 출력을 비교한다. concat 순서가 뒤집혔다면 여기서 걸린다.
    """
    from mmseg.models.backbones.mix_transformer import Attention

    stage, blk = 0, 0  # stage1 의 첫 블록으로 검사 (sr_ratio=8 경로를 함께 탄다)
    p = f"segformer.encoder.block.{stage}.{blk}"
    dim = hf_sd[f"{p}.attention.self.query.weight"].shape[0]
    sr = hf_sd[f"{p}.attention.self.sr.weight"].shape[-1]

    torch.manual_seed(0)
    H = W = sr * 2  # sr 후 2x2 = 4 토큰이 남도록
    x = torch.randn(1, H * W, dim)

    # --- 참조 구현: HF 가중치로 표준 MHA 를 직접 계산 ---
    def lin(name, inp):
        return inp @ hf_sd[f"{p}.{name}.weight"].t() + hf_sd[f"{p}.{name}.bias"]

    q = lin("attention.self.query", x)
    x_ = x.permute(0, 2, 1).reshape(1, dim, H, W)
    x_ = torch.nn.functional.conv2d(
        x_, hf_sd[f"{p}.attention.self.sr.weight"], hf_sd[f"{p}.attention.self.sr.bias"], stride=sr
    )
    x_ = x_.reshape(1, dim, -1).permute(0, 2, 1)
    x_ = torch.nn.functional.layer_norm(
        x_, (dim,), hf_sd[f"{p}.attention.self.layer_norm.weight"], hf_sd[f"{p}.attention.self.layer_norm.bias"]
    )
    k = lin("attention.self.key", x_)
    v = lin("attention.self.value", x_)

    heads = 1  # mit-b* 의 stage1 은 num_heads=1
    hd = dim // heads
    qh = q.reshape(1, -1, heads, hd).permute(0, 2, 1, 3)
    kh = k.reshape(1, -1, heads, hd).permute(0, 2, 1, 3)
    vh = v.reshape(1, -1, heads, hd).permute(0, 2, 1, 3)
    attn = (qh @ kh.transpose(-2, -1)) * (hd**-0.5)
    ref = (attn.softmax(dim=-1) @ vh).transpose(1, 2).reshape(1, -1, dim)
    ref = lin("attention.output.dense", ref)

    # --- 변환된 가중치를 실은 NVlabs 모듈 ---
    mod = Attention(dim, num_heads=heads, qkv_bias=True, sr_ratio=sr).eval()
    sub = {
        key[len(f"block{stage + 1}.{blk}.attn.") :]: val
        for key, val in nv_sd.items()
        if key.startswith(f"block{stage + 1}.{blk}.attn.")
    }
    mod.load_state_dict(sub, strict=True)
    with torch.no_grad():
        got = mod(x, H, W)

    diff = (ref - got).abs().max().item()
    if diff > 1e-4:
        raise AssertionError(f"어텐션 수치 불일치: max|diff|={diff:.3e} — k/v concat 순서 확인 필요")
    print(f"  어텐션 수치 검증 통과 (max|diff| = {diff:.2e})")


def verify(model: str, nv_sd: dict) -> int:
    """실제 mit_b* 모듈에 strict=True 로 실어 본다. 하나라도 안 맞으면 예외."""
    import mmseg.models.backbones.mix_transformer as mt

    net = getattr(mt, f"mit_{model}")()
    net.load_state_dict(nv_sd, strict=True)  # 여기서 통과하면 키·shape 가 전부 일치
    n = sum(t.numel() for t in nv_sd.values())
    print(f"  strict=True 로드 통과 — 텐서 {len(nv_sd)}개 / 파라미터 {n / 1e6:.2f}M")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS, help="변환할 모델 (기본: 전부)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="출력 디렉토리")
    ap.add_argument("--dry-run", action="store_true", help="변환·검증만 하고 저장하지 않는다")
    ap.add_argument("--overwrite", action="store_true", help="기존 mit_b*.pth 를 덮어쓴다")
    ap.add_argument("--force-download", action="store_true", help="캐시를 무시하고 다시 받는다")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    for model in args.models:
        print(f"[mit-{model}]")
        dst = args.out_dir / f"mit_{model}.pth"
        if dst.exists() and not args.overwrite and not args.dry_run:
            print(f"  이미 존재 — 건너뜀 (덮어쓰려면 --overwrite): {dst}")
            continue

        src = download(model, force=args.force_download)
        hf_sd = torch.load(src, map_location="cpu")
        hf_sd = hf_sd.get("state_dict", hf_sd)

        nv_sd = convert(hf_sd)
        n_blocks = len([k for k in hf_sd if k.endswith("attention.self.key.weight")])
        print(f"  HF {len(hf_sd)}개 -> NVlabs {len(nv_sd)}개 (분류기 2개 제외, {n_blocks}개 블록의 k/v 를 kv 로 합침)")

        check_attention_math(hf_sd, nv_sd)
        verify(model, nv_sd)

        if args.dry_run:
            print(f"  [dry-run] 저장 생략 (저장했다면: {dst})")
        else:
            torch.save(nv_sd, dst)
            print(f"  저장: {dst} ({dst.stat().st_size / 1e6:.1f}MB)")
        ok.append(model)

    print(f"\n완료: {len(ok)}개 — {', '.join(ok) if ok else '(없음)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
