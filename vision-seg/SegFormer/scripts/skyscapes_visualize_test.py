#!/usr/bin/env python3
"""학습된 SegFormer 로 test 세트를 추론하고 `원본 | GT | 예측` 3분할 이미지를 만든다.

두 종류를 만든다.
  1. 타일 단위 (1024x1024 x 3패널)  -> tiles/{타일이름}.png
  2. 원본 이미지 단위               -> full/{원본stem}.jpg
     타일 예측을 파일명의 좌표(_x00000_y00512)로 되붙여 5616x3744 로 복원한 뒤 축소한다.

GT 와 예측은 원본 위에 반투명 오버레이로 그린다. 차선이 얇아서 마스크만 따로 보면
도로와의 위치 관계를 알 수 없기 때문이다.
  solid  = 파랑 [0,0,255]   dashed = 빨강 [255,0,0]   (데이터셋 PALETTE 계승)

반드시 segformer_cu111 환경의 python 으로 실행한다.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEGFORMER_DIR = Path("/home/hsjeong/workspace/vision-seg/SegFormer/source")
TILE_ROOT = Path("/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split/tiles")
ORIG_ROOT = Path("/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split")

CLASS_NAMES = ("background", "solid", "dashed")
PALETTE = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0]], dtype=np.uint8)
IGNORE = 255
ALPHA = 0.55  # 오버레이 불투명도

TILE_RE = re.compile(r"^(?P<stem>.+)_x(?P<x>\d{5})_y(?P<y>\d{5})\.png$")


# 한글 제목을 쓰므로 CJK 폰트가 필요하다. DejaVuSans 는 한글 글리프가 없어 □□ 로 깨진다.
# NotoSansCJK-Regular.ttc 의 index=1 이 한국어(Noto Sans CJK KR)다.
_FONT_CANDIDATES = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 1),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 1),
]


def get_font(size):
    for path, idx in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=idx)
            except Exception:
                pass
    try:  # CJK 폰트가 없으면 라틴 문자라도 제대로 나오게
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
        if p.exists():
            return ImageFont.truetype(str(p), size)
    except Exception:
        pass
    return ImageFont.load_default()


def dilate(mask, r):
    """max-pool 방식의 사각 팽창. scipy 없이 numpy 만으로 처리한다."""
    if r <= 0:
        return mask
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def colorize_overlay(img_rgb, label, thicken=0):
    """label(0/1/2/255) 을 img 위에 반투명 오버레이로 얹는다.

    thicken > 0 이면 전경 마스크를 그만큼 두껍게 그린다. 원본(5616x3744)을 화면 폭에
    맞춰 축소하면 폭 2~4px 인 차선이 사라져 버리기 때문에, 전체보기 전용으로 쓴다.
    타일 보기(등배율)에서는 0 이어야 실제 예측 두께를 볼 수 있다.
    """
    out = img_rgb.astype(np.float32).copy()
    for cls in (1, 2):
        m = label == cls
        if not m.any():
            continue
        m = dilate(m, thicken)
        out[m] = (1 - ALPHA) * out[m] + ALPHA * PALETTE[cls].astype(np.float32)
    # ignore(패딩) 영역은 어둡게 깔아 평가 제외 구간임을 표시
    m = label == IGNORE
    if m.any():
        out[m] = out[m] * 0.35
    return out.astype(np.uint8)


def compose(panels, titles, pad=12, header=52, scale=1.0):
    """패널들을 가로로 이어 붙이고 각 패널 위에 제목을 단다."""
    if scale != 1.0:
        panels = [np.array(Image.fromarray(p).resize(
            (int(p.shape[1] * scale), int(p.shape[0] * scale)), Image.BILINEAR)) for p in panels]
    h, w = panels[0].shape[:2]
    n = len(panels)
    canvas = Image.new("RGB", (w * n + pad * (n + 1), h + header + pad * 2), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    font = get_font(max(16, int(h * 0.030)))
    for i, (p, t) in enumerate(zip(panels, titles)):
        x = pad + i * (w + pad)
        canvas.paste(Image.fromarray(p), (x, header))
        draw.text((x + 6, pad + 2), t, fill=(240, 240, 245), font=font)
    return canvas


def legend_bar(width, height=44):
    """solid/dashed 색 범례."""
    bar = Image.new("RGB", (width, height), (24, 24, 28))
    d = ImageDraw.Draw(bar)
    font = get_font(20)
    x = 14
    for cls in (1, 2):
        d.rectangle([x, 12, x + 28, 32], fill=tuple(int(v) for v in PALETTE[cls]))
        d.text((x + 38, 12), CLASS_NAMES[cls], fill=(235, 235, 240), font=font)
        x += 190
    d.text((x, 12), "어두운 영역 = ignore(패딩, 평가 제외)", fill=(170, 170, 178), font=font)
    return bar


def stack_vertical(top, bottom):
    out = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height), (24, 24, 28))
    out.paste(top, (0, 0))
    out.paste(bottom, (0, top.height))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(SEGFORMER_DIR / "local_configs/segformer/B0/segformer.b0.1024x1024.skyscapes.20k.py"))
    ap.add_argument("--checkpoint", default=str(SEGFORMER_DIR / "work_dirs/b0_skyscapes_20k/iter_10000.pth"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", default=str(SEGFORMER_DIR / "work_dirs/b0_skyscapes_20k/vis_test"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tile-scale", type=float, default=1.0, help="타일 3분할 이미지 축소 배율")
    ap.add_argument("--full-width", type=int, default=1600, help="원본 복원 3분할의 패널 1개 가로 폭")
    ap.add_argument("--no-tiles", action="store_true", help="타일 단위 출력 생략")
    ap.add_argument("--no-full", action="store_true", help="원본 복원 출력 생략")
    args = ap.parse_args()

    from mmseg.apis import inference_segmentor, init_segmentor

    img_dir = TILE_ROOT / args.split / "images"
    ann_dir = TILE_ROOT / args.split / "labels"
    names = sorted(os.listdir(img_dir))
    print(f"{args.split} 타일 {len(names)}장")

    print(f"모델 로드: {Path(args.checkpoint).name}")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    out_root = Path(args.out_dir)
    (out_root / "tiles").mkdir(parents=True, exist_ok=True)
    (out_root / "full").mkdir(parents=True, exist_ok=True)

    # 원본 복원용 버퍼: stem -> (pred canvas, gt canvas)
    canvases = {}
    inter = np.zeros(3, np.int64)
    union = np.zeros(3, np.int64)

    for i, name in enumerate(names, 1):
        ip, lp = img_dir / name, ann_dir / name
        img = np.array(Image.open(ip).convert("RGB"))
        gt = np.array(Image.open(lp))
        pred = inference_segmentor(model, str(ip))[0].astype(np.uint8)

        valid = gt != IGNORE
        for c in range(3):
            p, g = (pred == c) & valid, (gt == c) & valid
            inter[c] += np.logical_and(p, g).sum()
            union[c] += np.logical_or(p, g).sum()

        if not args.no_tiles:
            fg = int(((gt == 1) | (gt == 2)).sum())
            panels = [img, colorize_overlay(img, gt), colorize_overlay(img, pred)]
            titles = ["원본", f"정답 GT (전경 {fg:,}px)", f"예측 {Path(args.checkpoint).stem}"]
            tri = compose(panels, titles, scale=args.tile_scale)
            full = stack_vertical(tri, legend_bar(tri.width))
            full.save(out_root / "tiles" / name)

        if not args.no_full:
            m = TILE_RE.match(name)
            if m:
                stem, x, y = m["stem"], int(m["x"]), int(m["y"])
                if stem not in canvases:
                    # 02단계 타일링은 격자+패딩 방식이라 패딩 포함 크기로 캔버스를 잡는다
                    canvases[stem] = {"pred": {}, "gt": {}}
                canvases[stem]["pred"][(x, y)] = pred
                canvases[stem]["gt"][(x, y)] = gt

        if i % 12 == 0 or i == len(names):
            print(f"  추론 {i}/{len(names)}")

    iou = np.where(union > 0, inter / np.maximum(union, 1) * 100, np.nan)
    print("\n재계산한 IoU (검증용, dist_test 결과와 일치해야 한다)")
    for c in range(3):
        print(f"  {CLASS_NAMES[c]:11s} {iou[c]:6.2f}")
    print(f"  전경평균    {np.nanmean(iou[1:]):6.2f}")

    if not args.no_full and canvases:
        print("\n원본 좌표로 타일 되붙이는 중")
        for stem, d in canvases.items():
            coords = list(d["pred"].keys())
            W = max(x for x, _ in coords) + 1024
            H = max(y for _, y in coords) + 1024
            pred_c = np.zeros((H, W), np.uint8)
            gt_c = np.full((H, W), IGNORE, np.uint8)
            for (x, y), p in d["pred"].items():
                pred_c[y:y + 1024, x:x + 1024] = p
            for (x, y), g in d["gt"].items():
                gt_c[y:y + 1024, x:x + 1024] = g

            src = next((ORIG_ROOT / args.split / "images").glob(stem + ".*"))
            orig = np.array(Image.open(src).convert("RGB"))
            oh, ow = orig.shape[:2]
            pred_c, gt_c = pred_c[:oh, :ow], gt_c[:oh, :ow]  # 패딩 제거

            sc = args.full_width / ow
            # 축소 배율의 역수만큼 선을 두껍게 해야 축소 후에도 남는다
            th = max(1, int(round(1 / sc)))
            panels = [orig,
                      colorize_overlay(orig, gt_c, thicken=th),
                      colorize_overlay(orig, pred_c, thicken=th)]
            ck = Path(args.checkpoint).stem
            titles = [f"원본 {stem[-8:]} ({ow}x{oh})",
                      f"정답 GT (선 {2 * th + 1}px 굵게 표시)",
                      f"예측 {ck} (선 {2 * th + 1}px 굵게 표시)"]
            tri = compose(panels, titles, scale=sc)
            out = stack_vertical(tri, legend_bar(tri.width))
            dst = out_root / "full" / f"{stem}.jpg"
            out.save(dst, quality=92)
            print(f"  {dst.name}  {out.width}x{out.height}")

    print(f"\n출력: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
