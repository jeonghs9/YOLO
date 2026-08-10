#!/usr/bin/env python3
"""split/ 의 원본 해상도 이미지(5616x3744)를 SegFormer 학습용 타일로 자른다.

출력 구조 (mmseg CustomDataset 이 그대로 읽는 형태):

    split/tiles/
    ├── train/{images, labels}/
    ├── val/{images, labels}/
    └── test/{images, labels}/

CustomDataset 은 img_dir 를 재귀 스캔해 img_suffix 로 끝나는 파일을 모으고,
같은 상대경로에서 img_suffix -> seg_map_suffix 로 바꾼 파일을 ann_dir 에서 찾는다
(mmseg/datasets/custom.py:146-151). 따라서 images/ 와 labels/ 의 파일명이
확장자까지 완전히 같으면 된다. 여기서는 둘 다 .png 로 저장한다.

타일 규칙:
  - train: stride 512 (50% 오버랩), 가장자리 타일은 이미지 안쪽으로 당겨 붙인다.
    데이터가 6장뿐이라 오버랩으로 샘플 수를 늘리는 것이 목적이고, 중복은 증강으로 본다.
  - val/test: 겹침 없는 격자 + 가장자리 패딩. **각 픽셀이 정확히 한 번만 평가되어야 하기 때문이다.**
    5616 은 1024 로 나누어떨어지지 않으므로, 마지막 타일을 안쪽으로 당기면 528px(가로)
    352px(세로)가 이중 집계되어 지표가 왜곡된다(실측: test solid 88,856 -> 175,017).
    대신 오른쪽/아래를 패딩한다. 이미지 패딩은 0, 라벨 패딩은 ignore_index(255)라
    손실/지표 계산에서 자동 제외된다(CustomDataset 기본 ignore_index=255).
  - 라벨은 최근접 방식인 배열 슬라이싱만 쓴다. 리사이즈를 하지 않으므로 없는 클래스 ID가 생길 여지가 없다.

전경 없는 타일:
  train 에서만 솎아낸다. 전경 타일 수 대비 --train-empty-ratio 배까지만 남기고,
  정렬된 목록에서 균등 간격으로 결정론적으로 고른다. val/test 는 평가 커버리지를 위해 전부 남긴다.

사용법:
    python3 /home/hsjeong/workspace/vision-seg/SegFormer/scripts/skyscapes_make_tiles.py --overwrite
    python3 .../skyscapes_make_tiles.py --dry-run
    python3 .../skyscapes_make_tiles.py --tile 512 --stride-train 256 --overwrite
"""

import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SRC = "/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split"
DST = os.path.join(SRC, "tiles")
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("background", "solid", "dashed")
IGNORE_INDEX = 255


def tile_origins(length, tile, stride, pad):
    """타일 시작 좌표.

    pad=False: [0, length-tile] 안에서만 잡고 마지막을 length-tile 로 당겨 붙인다(겹침 발생).
    pad=True : 0 부터 tile 간격 격자. 마지막 타일이 이미지 밖으로 나가면 패딩한다(겹침 없음).
    """
    if pad:
        return list(range(0, length, tile))
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile + 1, stride))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def crop(arr, x, y, tile, pad_val):
    """(y, x) 에서 tile 크기로 자른다. 이미지 밖은 pad_val 로 채운다."""
    h, w = arr.shape[:2]
    y1, x1 = min(y + tile, h), min(x + tile, w)
    patch = arr[y:y1, x:x1]
    ph, pw = tile - (y1 - y), tile - (x1 - x)
    if ph or pw:
        pad_width = [(0, ph), (0, pw)] + [(0, 0)] * (arr.ndim - 2)
        patch = np.pad(patch, pad_width, mode="constant", constant_values=pad_val)
    return patch


def plan_tiles(src_split_dir, tile, stride, pad):
    """(stem, x, y, fg_pixels) 목록을 만든다. 픽셀 계산은 라벨만 읽는다."""
    lab_dir = os.path.join(src_split_dir, "labels", "grayscale")
    stems = sorted(f[:-4] for f in os.listdir(lab_dir) if f.endswith(".png"))
    plan = []
    for stem in stems:
        lab = np.array(Image.open(os.path.join(lab_dir, stem + ".png")))
        h, w = lab.shape
        for y in tile_origins(h, tile, stride, pad):
            for x in tile_origins(w, tile, stride, pad):
                patch = crop(lab, x, y, tile, IGNORE_INDEX)
                fg = int(np.count_nonzero((patch == 1) | (patch == 2)))
                plan.append((stem, x, y, fg))
    return plan


def select_train_tiles(plan, min_fg, empty_ratio):
    """전경 타일은 전부 남기고, 빈 타일은 균등 간격으로 일부만 남긴다."""
    keep_fg = [t for t in plan if t[3] >= min_fg]
    empties = [t for t in plan if t[3] < min_fg]
    budget = int(round(len(keep_fg) * empty_ratio))
    if budget >= len(empties):
        keep_empty = empties
    elif budget <= 0:
        keep_empty = []
    else:
        idx = np.linspace(0, len(empties) - 1, budget).round().astype(int)
        keep_empty = [empties[i] for i in sorted(set(idx.tolist()))]
    return keep_fg, keep_empty


def write_tiles(src_split_dir, out_split_dir, selected, tile):
    """선택된 타일을 잘라 저장한다. 이미지는 stem 단위로 한 번만 연다."""
    os.makedirs(os.path.join(out_split_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_split_dir, "labels"), exist_ok=True)
    by_stem = {}
    for stem, x, y, fg in selected:
        by_stem.setdefault(stem, []).append((x, y))

    n = 0
    for stem in sorted(by_stem):
        img = np.array(Image.open(os.path.join(src_split_dir, "images", stem + ".jpg")).convert("RGB"))
        lab = np.array(Image.open(os.path.join(src_split_dir, "labels", "grayscale", stem + ".png")))
        for x, y in sorted(by_stem[stem]):
            name = f"{stem}_x{x:05d}_y{y:05d}.png"
            Image.fromarray(crop(img, x, y, tile, 0)).save(
                os.path.join(out_split_dir, "images", name))
            Image.fromarray(crop(lab, x, y, tile, IGNORE_INDEX), mode="L").save(
                os.path.join(out_split_dir, "labels", name))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--stride-train", type=int, default=512)
    ap.add_argument("--eval-overlap", action="store_true",
                    help="val/test 도 train 처럼 겹치게 자른다. 지표가 이중 집계되므로 권장하지 않는다")
    ap.add_argument("--min-fg", type=int, default=1,
                    help="이 값 이상의 전경 픽셀이 있으면 전경 타일로 본다")
    ap.add_argument("--train-empty-ratio", type=float, default=0.3,
                    help="train 에서 남길 빈 타일 수 = 전경 타일 수 x 이 값")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    eval_mode = "겹침(비권장)" if args.eval_overlap else "겹침없는 격자 + ignore 패딩"
    print(f"타일 {args.tile}x{args.tile} | train stride {args.stride_train} (겹침) "
          f"| val/test {eval_mode} | min_fg {args.min_fg} "
          f"| train 빈타일 비율 {args.train_empty_ratio}\n")

    plans, selections = {}, {}
    for s in SPLITS:
        src_split_dir = os.path.join(args.src, s)
        if not os.path.isdir(src_split_dir):
            sys.exit(f"[중단] 입력 split 없음: {src_split_dir}")
        is_train = s == "train"
        stride = args.stride_train if is_train else args.tile
        pad = not is_train and not args.eval_overlap
        plan = plan_tiles(src_split_dir, args.tile, stride, pad)
        plans[s] = plan
        if is_train:
            fg, empty = select_train_tiles(plan, args.min_fg, args.train_empty_ratio)
            selections[s] = fg + empty
        else:
            selections[s] = plan  # 평가 커버리지 유지

    print(f"  {'split':6s} {'전체':>6s} {'전경':>6s} {'빈':>6s} {'선택':>6s}  비고")
    for s in SPLITS:
        plan = plans[s]
        n_fg = sum(1 for t in plan if t[3] >= args.min_fg)
        n_em = len(plan) - n_fg
        sel = len(selections[s])
        note = "빈 타일 솎아냄" if s == "train" else "전부 사용"
        print(f"  {s:6s} {len(plan):6d} {n_fg:6d} {n_em:6d} {sel:6d}  {note}")

    if args.dry_run:
        print("\n[dry-run] 파일을 생성하지 않았다.")
        return

    if os.path.isdir(args.dst) and os.listdir(args.dst):
        if not args.overwrite:
            sys.exit(f"\n[중단] 출력 디렉토리가 비어 있지 않다: {args.dst}\n"
                     f"       덮어쓰려면 --overwrite 를 붙여라.")
        print(f"\n[삭제] {args.dst}")
        shutil.rmtree(args.dst)

    total = 0
    for s in SPLITS:
        n = write_tiles(os.path.join(args.src, s), os.path.join(args.dst, s),
                        selections[s], args.tile)
        print(f"  {s:6s} {n:5d}장 저장")
        total += n

    print(f"\n총 {total}타일 생성")
    print(f"출력: {args.dst}")
    print(f"클래스: {', '.join(f'{i}={c}' for i, c in enumerate(CLASS_NAMES))}")


if __name__ == "__main__":
    main()
