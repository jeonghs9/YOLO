#!/usr/bin/env python3
"""AI Hub 차선 데이터(JSON 폴리라인)를 SegFormer 학습용 마스크 PNG 로 변환한다.

AI Hub 라벨은 차선의 **중심선 폴리라인**이라 두께가 없다. semantic segmentation 은
픽셀마다 정답이 필요하므로 선을 일정 폭으로 칠해 면으로 만든다(rasterize).

## 원근 보정이 필수인 이유 (실측)

차량 전방 시점이라 같은 차선이라도 화면 위(멀리)는 가늘고 아래(가까이)는 굵게 보인다.
실선 구간에서 선에 수직으로 밝기 프로파일을 떠 실제 도색 폭을 측정한 결과:

    y  625 ->  5.75px      y  925 -> 13.00px
    y  725 ->  6.00px      y 1025 -> 26.25px
    y  825 ->  8.00px      y 1175 -> 32.25px   (daylight 1920x1200)

차선 라벨의 y 중앙값이 690 인데 그 지점 실제 폭은 6px 이다. CULane 방식대로 고정 폭
(예: 12px)으로 칠하면 **대다수 라벨이 실제의 2 배로 부풀려진다.** 그래서 y 좌표별
실측 중앙값 표를 선형보간해 폭을 정한다.

카메라 장착 위치가 달라 서브셋마다 곡선이 다르므로 표도 서브셋별로 둔다.
`--measure` 로 표를 다시 뽑을 수 있다.

## 클래스

    0 background  : 그 외 전부 (crosswalk, stop_line 포함)
    1 solid       : traffic_lane 중 lane_type=solid
    2 dashed      : traffic_lane 중 lane_type=dotted

crosswalk(polygon)와 stop_line 은 background 로 흡수한다. ignore 로 두면 그 자리에
차선이 무작위로 예측된다 (SkyScapes 01단계 3절과 같은 판단).

**dashed 라벨은 점선의 빈칸까지 덮는다.** 폴리라인이 빈칸을 관통하기 때문이다. 즉 이
데이터의 라벨은 "도색된 픽셀"이 아니라 "차선이 지나가는 경로(띠)"를 뜻한다.
SkyScapes 와 라벨의 의미가 다르므로 두 데이터의 IoU 를 직접 비교하면 안 된다.
"""

import argparse
import csv
import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

SRC = Path("/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE")
DST = Path("/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE_SEG")

CLASS_NAMES = ("background", "solid", "dashed")
LANE_TYPE_TO_ID = {"solid": 1, "dotted": 2}

# y 구간 중앙값 -> 실측 도색 폭(px). measure_width 결과. 사이 값은 선형보간, 밖은 클램프.
WIDTH_TABLE = {
    "c_1920_1200_daylight_validation_3": [
        (575, 9.25), (625, 5.75), (675, 5.75), (725, 6.00), (775, 6.50), (825, 8.00),
        (875, 9.50), (925, 13.00), (975, 22.00), (1025, 26.25), (1075, 29.25),
        (1125, 31.75), (1175, 32.25)],
    "c_1920_1200_night_validation_1": [
        (625, 4.25), (675, 5.00), (725, 6.25), (775, 8.25), (825, 11.50), (875, 15.00),
        (925, 19.50), (975, 23.25), (1025, 26.50), (1075, 29.50), (1125, 32.25),
        (1175, 33.25)],
    "1920_1080_night_validation_d_1": [
        (525, 4.00), (575, 3.25), (625, 3.50), (675, 5.25), (725, 7.50), (775, 10.00),
        (825, 15.25), (875, 20.50), (925, 24.62), (975, 26.75), (1025, 29.25),
        (1075, 30.50)],
}
# daylight_validation_2 는 같은 카메라 규격(c_1920_1200_daylight)이라 _3 의 표를 쓴다
WIDTH_TABLE["c_1920_1200_daylight_validation_2"] = WIDTH_TABLE["c_1920_1200_daylight_validation_3"]

MIN_W, MAX_W = 3, 36


def width_at(table_y, table_w, y):
    """y 에서의 선 폭. 표 밖은 양 끝값으로 클램프한다."""
    return float(np.clip(np.interp(y, table_y, table_w), MIN_W, MAX_W))


def label_dir(sub):
    p = SRC / f"[라벨]{sub}"
    if not any(f.endswith(".json") for f in os.listdir(p)):
        p = p / next(d for d in os.listdir(p) if (p / d).is_dir())
    return p


def subsets():
    return sorted(d.name[4:] for d in SRC.iterdir() if d.name.startswith("[원천]"))


def render_one(args, sub, ty, tw, step, overwrite):
    """JSON 하나 -> 마스크 PNG 하나. (stem, solid_px, dashed_px, H, W, err) 반환."""
    stem, lpath, ipath, opath = args
    if opath.exists() and not overwrite:
        m = cv2.imread(str(opath), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return (stem, int((m == 1).sum()), int((m == 2).sum()), m.shape[0], m.shape[1], "")
    try:
        d = json.load(open(lpath, encoding="utf-8"))
    except Exception as e:
        return (stem, 0, 0, 0, 0, f"json: {e}")
    H, W = d["image"]["image_size"]  # [height, width]
    mask = np.zeros((H, W), np.uint8)

    # dotted 를 먼저, solid 를 나중에 그린다. 교차 지점에서 solid 가 살아남게 하려는 것이다
    # (실선 침범 판정이 하류 목적이라 solid 를 보수적으로 지킨다).
    for want in ("dotted", "solid"):
        for a in d.get("annotations", []):
            if a.get("class") != "traffic_lane":
                continue
            t = {x["code"]: x["value"] for x in a.get("attributes", [])}.get("lane_type")
            if t != want:
                continue
            cid = LANE_TYPE_TO_ID[t]
            pts = [(p["x"], p["y"]) for p in a.get("data", [])]
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                (x1, y1), (x2, y2) = pts[i], pts[i + 1]
                ln = float(np.hypot(x2 - x1, y2 - y1))
                n = max(1, int(ln / step))
                for k in range(n):
                    ax = x1 + (x2 - x1) * k / n
                    ay = y1 + (y2 - y1) * k / n
                    bx = x1 + (x2 - x1) * (k + 1) / n
                    by = y1 + (y2 - y1) * (k + 1) / n
                    w = int(round(width_at(ty, tw, (ay + by) / 2)))
                    cv2.line(mask, (int(round(ax)), int(round(ay))),
                             (int(round(bx)), int(round(by))), cid, w, lineType=cv2.LINE_8)

    opath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(opath), mask)
    return (stem, int((mask == 1).sum()), int((mask == 2).sum()), H, W, "")


def clip_ids(stems):
    """파일명이 연속 정수인 구간을 하나의 영상 클립으로 본다.

    이 데이터는 영상에서 뽑은 연속 프레임이라 인접 프레임이 거의 같은 장면이다
    (무작위 쌍 대비 3배 유사). 프레임 단위로 무작위 분할하면 train/test 누수가 생기므로
    분할은 반드시 클립 단위로 해야 한다. 그 클립 ID 를 여기서 매긴다.
    """
    nums = sorted(int(s) for s in stems)
    out, cid = {}, 0
    for i, n in enumerate(nums):
        if i and n - nums[i - 1] != 1:
            cid += 1
        out[str(n)] = cid
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DST)
    ap.add_argument("--subsets", nargs="+", default=None, help="기본: 전체")
    ap.add_argument("--step", type=float, default=4.0, help="폴리라인 분할 간격(px). 작을수록 폭 변화가 매끄럽다")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="서브셋당 N개만 (시험용)")
    args = ap.parse_args()

    subs = args.subsets or subsets()
    rows = []
    for sub in subs:
        if sub not in WIDTH_TABLE:
            print(f"[!] {sub}: 폭 보정표가 없다. 건너뜀")
            continue
        ldir, idir = label_dir(sub), SRC / f"[원천]{sub}"
        odir = args.out / "masks" / sub
        stems = sorted(f[:-5] for f in os.listdir(ldir) if f.endswith(".json"))
        if args.limit:
            stems = stems[:args.limit]
        cid = clip_ids(stems)
        ty = [p[0] for p in WIDTH_TABLE[sub]]
        tw = [p[1] for p in WIDTH_TABLE[sub]]
        jobs = [(s, ldir / f"{s}.json", idir / f"{s}.jpg", odir / f"{s}.png") for s in stems]
        fn = partial(render_one, sub=sub, ty=ty, tw=tw, step=args.step, overwrite=args.overwrite)
        print(f"[{sub}] {len(jobs):,}장, 클립 {max(cid.values())+1:,}개 -> {odir}")
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap(fn, jobs, chunksize=32), 1):
                stem, s_px, d_px, H, W, err = r
                if err:
                    print(f"  [!] {stem}: {err}")
                    continue
                rows.append(dict(subset=sub, stem=stem, clip=cid[stem], H=H, W=W,
                                 solid_px=s_px, dashed_px=d_px,
                                 img=str(idir / f"{stem}.jpg"), mask=str(odir / f"{stem}.png")))
                if i % 2000 == 0:
                    print(f"  {i:,}/{len(jobs):,}")

    args.out.mkdir(parents=True, exist_ok=True)
    mf = args.out / "manifest.csv"
    with open(mf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n총 {len(rows):,}장 -> {mf}")
    tot_s = sum(r["solid_px"] for r in rows)
    tot_d = sum(r["dashed_px"] for r in rows)
    tot_p = sum(r["H"] * r["W"] for r in rows)
    print(f"픽셀 비율: background {(tot_p-tot_s-tot_d)/tot_p*100:.3f}%  "
          f"solid {tot_s/tot_p*100:.3f}%  dashed {tot_d/tot_p*100:.3f}%")
    n_empty = sum(1 for r in rows if r["solid_px"] + r["dashed_px"] == 0)
    print(f"전경 없는 장: {n_empty:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
