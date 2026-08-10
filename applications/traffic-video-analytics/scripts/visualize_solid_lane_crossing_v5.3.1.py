#!/usr/bin/env python3
"""Standalone solid-lane crossing pipeline, release 5.3.

The complete lane classification, temporal stabilization, crossing detection,
rendering, and OCR runtime is implemented in this file without project-local
Python imports.

Release 5.2 keeps the last stable lane type when too much of the continuity
signal is unknown (X). Such frames do not enter the temporal vote history.

Release 5.3 changes only how the unknown parts of the continuity signal are
handled, so that dotted lanes stop collapsing into SOLID near vehicles.

1. Off-frame slices are no longer merged with vehicle-occluded slices. Only
   vehicle occlusion counts toward the unknown ratio that suppresses a vote.
2. A dotted gap is accepted when lane pixels exist on both sides after
   skipping over unknown slices, instead of requiring lane pixels in the two
   directly adjacent slices.

The crossing trigger, the vehicle centre probe, and every rendering path are
identical to release 5.2.
"""

from __future__ import annotations
import argparse
import csv
import itertools
import math
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# roi 너무 커지거나 작아지는 거 방지
def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


# 차선 세그멘테이션 폴리곤의 중심점을 계산
def polygon_center(polygon: np.ndarray) -> tuple[float, float]:
    contour = np.rint(polygon).astype(np.int32)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-06:
        return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    center = polygon.mean(axis=0)
    return (float(center[0]), float(center[1]))


# 차선 폴리곤 점들을 PCA로 분석해서 차선의 긴 방향을 각도로 변환
def pca_angle_deg(polygon: np.ndarray) -> float:
    points = polygon.astype(np.float32)
    _, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=2)
    direction = eigenvectors[0]
    return math.degrees(math.atan2(float(direction[1]), float(direction[0])))


# 차선 폴리곤에서 면적, 짧은 변 두께, 긴변/짧은변 비율을 계산, 종횡비가 낮으면 차선처럼 보이지 않으므로 필터링
def mask_geometry(polygon: np.ndarray) -> tuple[float, float, float]:
    """Return mask area, short-side thickness, and long/short aspect ratio."""
    contour = np.rint(polygon).astype(np.int32)
    area = abs(float(cv2.contourArea(contour)))
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
    short_side = min(float(side_a), float(side_b))
    long_side = max(float(side_a), float(side_b))
    aspect_ratio = long_side / max(short_side, 1e-06)
    return (area, short_side, aspect_ratio)


# 차선 seg 모델 및 ocr 모델

DEFAULT_LPR_DIR = Path(
    os.environ.get(
        "OCR_CAR_CATEGORY_LPR_DIR",
        "/home/hsjeong/workspace/Yolo26/ultralytics/UTIL/LPR/License-Plate-Recognition-System",
    )
)
DEFAULT_PADDLE_CACHE_DIR = DEFAULT_LPR_DIR / ".paddleocr-cache"
PADDLE_DET_MODEL = None
PADDLE_REC_MODEL = None


# yolo 디바이스 형식을 paddleocr 형식으로 바꿈
def normalize_paddle_device(device):
    """Convert YOLO-style device values to PaddleOCR/PaddleX device strings."""
    if device is None:
        return None
    value = str(device).strip()
    if not value:
        return None
    lower = value.lower()
    if lower == "cpu":
        return "cpu"
    if lower.startswith("cuda:"):
        return f"gpu:{value.split(':', 1)[1]}"
    if value.isdigit():
        return f"gpu:{value}"
    return value


# 번호판 OCR에서 숫자와 문자 혼동을 교정 (깃허브 khoi03 방식)
NUM_TO_CHAR = {
    "0": "D",
    "1": "T",
    "2": "Z",
    "3": "B",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
    "$": "S",
}
CHAR_TO_NUM = {
    "A": "4",
    "B": "8",
    "C": "0",
    "D": "0",
    "E": "6",
    "F": "5",
    "G": "6",
    "H": "4",
    "I": "1",
    "J": "1",
    "L": "4",
    "O": "0",
    "P": "6",
    "Q": "0",
    "R": "8",
    "S": "5",
    "T": "7",
    "U": "0",
    "V": "0",
    "Y": "1",
    "Z": "2",
}


# 번호판 OCR 결과를 track_id 별로 누적해서 안정화 / 최소 3회 결과가 누적되면 가장 안정적인 번호판 문자열을 확정
class PlateTextStabilizer:
    """track_id 별로 여러 프레임의 OCR 결과를 모아 최빈값으로 안정화한다.

    동일 track_id 에 대해 누적된 후보 중 (빈도 * 평균 신뢰도) 가 가장 높은 텍스트를 확정값으로 사용한다.

    """

    def __init__(self, min_votes: int = 3, history: int = 30):
        self.min_votes = min_votes
        self._buffer = defaultdict(lambda: deque(maxlen=history))
        self._confirmed = {}

    def update(self, track_id: int, text: str, conf: float) -> str:
        if not text:
            return self._confirmed.get(track_id, "")
        buf = self._buffer[track_id]
        buf.append((text, conf))
        score = defaultdict(lambda: [0, 0.0])
        for t, c in buf:
            score[t][0] += 1
            score[t][1] += c
        best_text, best_metric = ("", -1.0)
        for t, (cnt, conf_sum) in score.items():
            avg_conf = conf_sum / cnt
            metric = cnt * avg_conf
            if metric > best_metric:
                best_text, best_metric = (t, metric)
        if len(buf) >= self.min_votes:
            self._confirmed[track_id] = best_text
        return self._confirmed.get(track_id, text)


# OCR 원문에서 영문/숫자만 남기고 대문자로 정규화
def clean_plate_text(raw: str) -> str:
    return "".join((ch for ch in raw.upper() if ch.isalnum()))


PLATE_FORMAT_RE = re.compile("^[0-9]{2}[A-Z][0-9]{5}$")
DIGIT_CORRECTIONS = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}
LETTER_CORRECTIONS = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "B",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
}


# OCR 오인식을 번호판 위치에 따라 보정
def format_plate_text(raw: str) -> str:
    """Normalize OCR text to '##A ###.##'.

    The compact OCR token must contain two digits, one uppercase letter, and
    five digits. Common OCR confusions are corrected by position.
    """
    token = clean_plate_text(raw)
    if len(token) < 8:
        return ""
    best = ""
    best_exact_matches = -1
    for start in range(len(token) - 7):
        window = token[start : start + 8]
        chars = list(window)
        for idx in (0, 1, 3, 4, 5, 6, 7):
            chars[idx] = DIGIT_CORRECTIONS.get(chars[idx], chars[idx])
        chars[2] = LETTER_CORRECTIONS.get(chars[2], chars[2])
        normalized = "".join(chars)
        if PLATE_FORMAT_RE.match(normalized):
            exact_matches = sum((a == b for a, b in zip(window, normalized)))
            if exact_matches > best_exact_matches:
                best = normalized
                best_exact_matches = exact_matches
    if not best:
        return ""
    return f"{best[:3]} {best[3:6]}.{best[6:]}"


def extract_ocr_texts(result):
    """Extract OCR texts/confidences from PaddleOCR 2.x or 3.x results."""
    if not result:
        return ([], [])
    first = result[0]
    if first is None:
        return ([], [])
    if hasattr(first, "get") and "rec_texts" in first:
        texts = [str(t) for t in first.get("rec_texts", []) if t]
        confs = [float(c) for c in first.get("rec_scores", [])]
        return (texts, confs)
    lines = first
    texts = [line[1][0] for line in lines if line and line[1]]
    confs = [float(line[1][1]) for line in lines if line and line[1]]
    return (texts, confs)


# 번호판 크롭 전처리 2층구조 번호판 읽기
class KhoiPaddleOCR:
    """PaddleOCR post-processing style from khoi03 ALPR Colab.

    YOLO26 already supplies the plate crop, so this backend reuses only the
    repository's OCR handling: keep multiple OCR lines and join two-line plates.
    """

    def __init__(self, paddle_ocr):
        self.paddle_ocr = paddle_ocr

    def preprocess(self, img: np.ndarray):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def num2char(self, num: str):
        return "".join((NUM_TO_CHAR.get(ch, ch) for ch in num))

    def char2num(self, char: str):
        return "".join((CHAR_TO_NUM.get(ch, ch) for ch in char))

    def correct(self, text: str):
        if len(text) <= 4:
            return text
        if text[3] == "-":
            return self.char2num(text[:2]) + self.num2char(text[2]) + self.char2num(text[3:])
        if text[4] == "-":
            return self.char2num(text[:2]) + self.num2char(text[2]) + text[3] + self.char2num(text[4:])
        return text

    def read(self, crop: np.ndarray):
        if crop is None or crop.size == 0:
            return ("", 0.0)
        crop = self.preprocess(crop)
        result = self.paddle_ocr.predict(crop)
        texts, confs = extract_ocr_texts(result)
        if not texts:
            return ("", 0.0)
        if len(texts) == 2:
            text = texts[0].replace("-", "") + "-" + texts[1].replace("-", "")
        else:
            text = "".join(texts)
        text = re.sub("['\\\",\\\\.\\\\?:\\\\!]", "", text).strip().upper()
        text = "".join((ch for ch in text if ch.isalnum() or ch == "-"))
        if 7 <= len(text) <= 10:
            text = self.correct(text)
        text = format_plate_text(text)
        return (text, float(np.mean(confs)) if confs else 0.0)


def run_ocr(ocr_engine, crop: np.ndarray):
    """번호판 crop 에 대해 OCR 수행. (text, conf) 반환. 실패 시 ('', 0.0)."""
    if crop is None or crop.size == 0:
        return ("", 0.0)
    h, w = crop.shape[:2]
    if h < 32:
        scale = 32.0 / max(h, 1)
        crop = cv2.resize(crop, (int(w * scale), 32), interpolation=cv2.INTER_CUBIC)
    return ocr_engine.read(crop)


def intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = (max(a[0], b[0]), max(a[1], b[1]))
    x2, y2 = (min(a[2], b[2]), min(a[3], b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# 차와 번호판 박스의 겹침 정도를 계산.
def overlap_score(car_box: np.ndarray, plate_box: np.ndarray, metric: str) -> float:
    inter = intersection_area(car_box, plate_box)
    car_area = max(0.0, car_box[2] - car_box[0]) * max(0.0, car_box[3] - car_box[1])
    plate_area = max(0.0, plate_box[2] - plate_box[0]) * max(0.0, plate_box[3] - plate_box[1])
    denominator = plate_area if metric == "plate_ioa" else car_area + plate_area - inter
    return inter / denominator if denominator > 0 else 0.0


def clip_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(np.clip(np.floor(box[0]), 0, width))
    y1 = int(np.clip(np.floor(box[1]), 0, height))
    x2 = int(np.clip(np.ceil(box[2]), 0, width))
    y2 = int(np.clip(np.ceil(box[3]), 0, height))
    return (x1, y1, x2, y2)


def draw_box(frame, xyxy, color, label=None):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_plate_text(frame, xyxy, text, scale=1.4, thickness=3):
    """번호판 박스 위에 인식된 텍스트를 표시."""
    if not text:
        return
    x1, y1, x2, y2 = map(int, xyxy)
    _, frame_w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad_x, pad_y = (8, 8)
    cx = x1 + (x2 - x1) // 2 - tw // 2
    cx = max(pad_x, min(cx, frame_w - tw - pad_x))
    ty = max(y1 - 14, th + pad_y)
    cv2.rectangle(frame, (cx - pad_x, ty - th - pad_y), (cx + tw + pad_x, ty + pad_y), (0, 0, 0), -1)
    cv2.putText(frame, text, (cx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thickness)


def _prepare_paddleocr_environment() -> None:
    """Apply the legacy OCR cache settings immediately before PaddleOCR import."""
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(DEFAULT_LPR_DIR / ".paddlex-cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_LPR_DIR / ".matplotlib-cache"))
    os.environ.setdefault("PADDLE_HOME", str(DEFAULT_PADDLE_CACHE_DIR))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.object_, np.bytes_, np.str_],
        }


DET_CONFIDENCE = 0.25
DET_NMS_IOU = 0.7
TRACKER = "bytetrack.yaml"
COLOR_CAR = (255, 128, 0)
COLOR_FIRE = (0, 0, 255)
COLOR_CENTER = (0, 255, 255)
COLOR_SOLID = (0, 255, 0)
COLOR_DOTTED = (0, 165, 255)
COLOR_VIRTUAL_SOLID = (255, 255, 0)
COLOR_PLATE = (0, 0, 255)
BORDER_MARGIN_PX = 8.0

# 차선 조각 필터 — seg 폴리곤을 차선 후보로 볼지 판단하는 고정 기준
MIN_MASK_AREA = 50.0  # 이보다 작은 폴리곤은 차선 후보에서 제외(px^2)
MIN_ASPECT_RATIO = 2.0  # 긴변/짧은변 비율이 이보다 작으면 차선으로 보지 않음
MIN_OVERLAP_PIXELS = 20  # ROI 체인과 이 픽셀 수 이상 겹쳐야 같은 차선 조각으로 인정
MAX_INSTANCE_ANGLE_DIFF = 20.0  # 씨앗 조각과 각도 차이가 이보다 크면 다른 차선으로 간주(도)

# 번호판을 차량에 연결하는 고정 기준
ASSOCIATION_THRESHOLD = 0.7  # 최소 겹침 점수
ASSOCIATION_METRIC = "plate_ioa"  # 겹침 점수 방식 (plate_ioa | iou)

CSV_COLUMNS = [
    "frame",
    "car_id",
    "cx",
    "cy",
    "box_w",
    "box_h",
    "vehicle_scale",
    "stab_dx",
    "stab_dy",
    "motion_px",
    "hit_lane_id",
    "lane_type",
    "seg_count",
    "dynamic_threshold",
    "roi_length",
    "roi_width",
    "roi_visible_ratio",
    "normal_motion_ratio",
    "baseline_deg",
    "baseline_coherence",
    "heading_deviation_deg",
    "heading_hold_ratio",
    "heading_fired",
    "crossing_in_hold",
    "border_clipped",
    "crossing_fired",
    "flagged",
    "plate_text",
    "zooming",
    "homography_suspended",
]


@dataclass(frozen=True)
# 차선 폴리곤 기하 정보
class LaneGeometry:
    instance_id: int
    polygon: np.ndarray
    center: tuple[float, float]
    area: float
    thickness: float
    aspect_ratio: float
    angle_deg: float


@dataclass(frozen=True)
# 차선 판정
class LaneDecision:
    instance_id: int
    roi_polygon: np.ndarray
    center: tuple[float, float]
    thickness: float
    lane_type: str
    member_ids: tuple[int, ...]
    roi_width: float
    angle_deg: float
    raw_lane_type: str = ""
    stable_lane_type: str = ""
    lane_track_id: int = -1
    vote_samples: int = 0
    vote_solid_ratio: float = 0.0
    vote_dotted_ratio: float = 0.0
    temporal_confirmed: bool = True
    allow_occlusion_override: bool = True
    continuity_occupancy: float = 0.0
    continuity_gap_count: int = 0
    continuity_longest_gap: float = 0.0
    continuity_unknown_ratio: float = 0.0
    temporal_vote_skipped_unknown: bool = False


@dataclass(frozen=True)
# 차선이 차량에 의해 가려진 경우, 가상으로 그려진 차선의 시작점과 끝점 좌표
class OcclusionBridge:
    instance_id: int
    car_id: int
    start: tuple[int, int]
    end: tuple[int, int]
    thickness: int


@dataclass(frozen=True)
# 기존 옵티컬플로우때 사용하던 정보 현재 사용 X CSV 정보기록 호환목적으로만 사용
class CrossingHit:
    instance_id: int = -1
    lane_type: str = ""
    seg_count: int = 0
    dotted_threshold: int = 0
    roi_length: float = 0.0
    roi_width: float = 0.0
    roi_visible_ratio: float = 0.0
    normal_motion_ratio: float = 0.0
    raw_lane_type: str = ""
    stable_lane_type: str = ""
    lane_track_id: int = -1
    vote_samples: int = 0
    vote_solid_ratio: float = 0.0
    vote_dotted_ratio: float = 0.0
    continuity_occupancy: float = 0.0
    continuity_gap_count: int = 0
    continuity_longest_gap: float = 0.0
    continuity_unknown_ratio: float = 0.0
    fired: bool = False


@dataclass(frozen=True)
# 차량 최종 발화 결과
class FinalTriggerDecision:
    fired: bool = False
    instance_id: int = -1
    reason: str = ""
    solid_contact: bool = False
    solid_contact_pixels: int = 0
    solid_contact_hold_frames: int = 0
    solid_contact_hold_seconds: float = 0.0
    solid_center_crossed: bool = False
    solid_center_pixels: int = 0


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone stable-SOLID lane crossing detector with optional plate OCR"
    )

    # ── 필수 입출력과 실행 환경 ──
    parser.add_argument(
        "--det-model", required=True, help="차량·번호판 검출 모델(.pt). 4클래스 person/car/motorcycle/plate_number"
    )
    parser.add_argument("--lane-model", required=True, help="차선 세그멘테이션 모델(.pt). 단일 클래스 lane")
    parser.add_argument("--input", required=True, help="입력 영상 경로")
    parser.add_argument("--output", required=True, help="결과 영상 저장 경로")
    parser.add_argument("--csv", default=None, help="프레임·차량별 판정 로그 CSV 경로. 생략하면 기록하지 않음")
    parser.add_argument("--device", default="1", help="YOLO 추론 GPU 번호. cpu 도 가능")
    parser.add_argument(
        "--paddle-device", default=None, help="OCR 전용 GPU(gpu:0 형식). 생략하면 --device 값을 변환해 사용"
    )
    parser.add_argument("--max-frames", type=int, default=0, help="처리할 최대 프레임 수. 0이면 전체")
    parser.add_argument(
        "--output-scale", type=float, default=0.5, help="결과 영상 저장 배율. 판정은 원본 해상도에서 수행됨"
    )

    # ── 검출·추적 ──
    parser.add_argument("--car-class", type=int, default=1, help="검출 모델에서 차량에 해당하는 클래스 번호")
    parser.add_argument("--plate-class", type=int, default=3, help="검출 모델에서 번호판에 해당하는 클래스 번호")
    parser.add_argument("--det-imgsz", type=int, default=1024, help="차량 검출 입력 해상도")
    parser.add_argument("--lane-imgsz", type=int, default=1024, help="차선 seg 입력 해상도")
    parser.add_argument("--det-conf", type=float, default=DET_CONFIDENCE, help="차량·번호판 검출 confidence 임계값")
    parser.add_argument("--lane-conf", type=float, default=0.25, help="차선 seg confidence 임계값")
    parser.add_argument("--nms-iou", type=float, default=DET_NMS_IOU, help="검출 NMS IoU 임계값")
    parser.add_argument(
        "--tracker", default=TRACKER, help="차량 추적기 설정. botsort.yaml 은 optical flow GMC 때문에 거부됨"
    )

    # ── 차선 조각 필터 ──

    # ── ROI 체인 — 차선을 따라 뻗는 긴 상자 ──
    parser.add_argument(
        "--chain-roi-cell-length",
        type=float,
        default=300.0,
        help="ROI 한 칸의 길이(px). 전체 길이 = 이 값 x 칸수(홀수)",
    )
    parser.add_argument("--chain-roi-min-cells", type=int, default=3, help="ROI 최소 칸 수. 항상 홀수로 보정됨")
    parser.add_argument("--chain-roi-max-cells", type=int, default=9, help="ROI 최대 칸 수. 항상 홀수로 보정됨")
    parser.add_argument(
        "--chain-roi-recenter-ratio",
        type=float,
        default=0.75,
        help="다음 칸을 실제 차선 쪽으로 옮기는 비율. 1이면 완전히 따라감",
    )
    parser.add_argument(
        "--chain-roi-angle-blend", type=float, default=0.35, help="다음 칸 각도를 실제 차선 각도로 섞는 비율"
    )
    parser.add_argument("--chain-roi-line-thickness", type=int, default=1, help="화면에 그리는 ROI 상자 선 두께(px)")
    parser.add_argument("--roi-width-lane-scale", type=float, default=4.0, help="ROI 폭 후보 = 차선 두께 x 이 값")
    parser.add_argument(
        "--roi-width-car-scale", type=float, default=0.08, help="ROI 폭 후보 = 가장 가까운 차량 대각선 x 이 값"
    )
    parser.add_argument("--roi-length-lane-scale", type=float, default=80.0, help="ROI 길이 후보 = 차선 두께 x 이 값")
    parser.add_argument(
        "--roi-length-car-scale", type=float, default=2.5, help="ROI 길이 후보 = 가장 가까운 차량 대각선 x 이 값"
    )
    parser.add_argument("--roi-min-width", type=float, default=40.0, help="ROI 폭 하한(px)")
    parser.add_argument(
        "--roi-max-width", type=float, default=180.0, help="ROI 폭 상한(px). 너무 크면 옆 차선까지 샘플링됨"
    )
    parser.add_argument("--roi-min-length", type=float, default=500.0, help="ROI 요청 길이 하한(px)")
    parser.add_argument("--roi-max-length", type=float, default=2000.0, help="ROI 요청 길이 상한(px)")

    # ── 연속성 신호 1/0/-1/-2 — 실선·점선 판정 ──
    parser.add_argument(
        "--continuity-slice-length",
        type=float,
        default=10.0,
        help="신호 한 칸의 길이(px). ROI 를 이 단위로 잘라 1/0 판정",
    )
    parser.add_argument(
        "--continuity-cross-sample-step", type=float, default=2.0, help="한 칸 안에서 점을 찍는 간격(px)"
    )
    parser.add_argument(
        "--continuity-min-lane-samples", type=int, default=6, help="칸을 1(차선 있음)로 만들 최소 차선 점 개수"
    )
    parser.add_argument(
        "--continuity-min-lane-ratio",
        type=float,
        default=0.02,
        help="칸을 1로 만들 최소 차선 점 비율. 넓은 ROI 에서 얼룩 오인 방지",
    )
    parser.add_argument(
        "--continuity-vehicle-unknown-ratio",
        type=float,
        default=0.3,
        help="칸의 이 비율 이상이 차량 bbox 에 가리면 -1(판단 불가)",
    )
    parser.add_argument(
        "--continuity-max-unknown-ratio-for-vote",
        type=float,
        default=0.5,
        help="-1 비율이 이 값을 넘으면 그 프레임은 투표하지 않고 이전 판단 유지. 화면 밖(-2)은 분모에서 제외",
    )
    parser.add_argument(
        "--continuity-close-zero-gap-pixels",
        type=float,
        default=30.0,
        help="이 길이 이하의 내부 빈칸은 seg 노이즈로 보고 1로 메움(px)",
    )
    parser.add_argument(
        "--continuity-dotted-min-gap-pixels",
        type=float,
        default=60.0,
        help="이 길이 이상의 내부 빈칸을 점선 gap 으로 인정(px)",
    )
    parser.add_argument(
        "--continuity-dotted-min-gaps", type=int, default=1, help="점선으로 판정하는 데 필요한 유효 gap 개수"
    )

    # ── 시간 투표 — 프레임 간 차선 종류 안정화 ──
    parser.add_argument(
        "--lane-vote-window-frames", type=int, default=30, help="차선 종류 투표를 기억하는 최근 프레임 수"
    )
    parser.add_argument("--lane-vote-min-samples", type=int, default=10, help="판정을 시작하는 최소 표 수")
    parser.add_argument(
        "--lane-vote-lock-ratio", type=float, default=0.7, help="처음 SOLID/DOTTED 를 확정하는 데 필요한 비율"
    )
    parser.add_argument(
        "--lane-vote-switch-ratio", type=float, default=0.8, help="이미 확정된 종류를 반대로 바꾸는 데 필요한 비율"
    )
    parser.add_argument(
        "--lane-track-max-missed-frames", type=int, default=15, help="이 프레임 수 이상 안 보이면 차선 track 삭제"
    )
    parser.add_argument(
        "--lane-track-max-angle-diff", type=float, default=20.0, help="프레임 간 차선 track 매칭 허용 각도 차이(도)"
    )
    parser.add_argument(
        "--lane-track-lateral-width-ratio",
        type=float,
        default=1.5,
        help="track 매칭 횡방향 허용 = ROI 폭 x 이 값",
    )
    parser.add_argument(
        "--lane-track-min-lateral-pixels", type=float, default=80.0, help="track 매칭 횡방향 허용 하한(px)"
    )
    parser.add_argument(
        "--lane-track-max-longitudinal-pixels",
        type=float,
        default=2000.0,
        help="track 매칭 종방향 허용 상한(px)",
    )

    # ── 차량 발화 판정 ──
    parser.add_argument(
        "--solid-contact-box-width-ratio",
        type=float,
        default=0.8,
        help="접촉 검사에 쓰는 차량 bbox 중앙 영역의 가로 비율",
    )
    parser.add_argument(
        "--solid-contact-box-height-ratio",
        type=float,
        default=0.8,
        help="접촉 검사에 쓰는 차량 bbox 중앙 영역의 세로 비율",
    )
    parser.add_argument("--solid-contact-min-pixels", type=int, default=12, help="접촉으로 인정할 최소 실선 픽셀 수")
    parser.add_argument(
        "--solid-contact-hold-seconds",
        type=float,
        default=1.5,
        help="발화까지 필요한 누적 접촉 시간(초). 최대 8프레임 공백은 같은 구간으로 이어짐",
    )
    parser.add_argument(
        "--solid-center-radius", type=int, default=15, help="차량 중앙 판정 원의 반경(px). 화면 노란 원과 동일"
    )
    parser.add_argument(
        "--solid-center-min-pixels", type=int, default=3, help="중앙 원 안에서 통과로 인정할 최소 실선 픽셀 수"
    )
    parser.add_argument(
        "--border-margin",
        type=float,
        default=BORDER_MARGIN_PX,
        help="화면 경계에서 이 거리 안에 걸친 차량은 발화 차단. 0이면 차단 해제(px)",
    )

    # ── 청록 가상선 bridge — 차량에 가려진 실선 연결 ──
    parser.add_argument(
        "--occlusion-box-margin-ratio",
        type=float,
        default=0.08,
        help="bridge 판정용 차량 bbox 확장 여유 = 차량 대각선 x 이 값",
    )
    parser.add_argument(
        "--occlusion-max-gap-car-ratio",
        type=float,
        default=1.5,
        help="이을 수 있는 두 조각 사이 간격 상한 = 차량 대각선 x 이 값",
    )
    parser.add_argument(
        "--occlusion-max-lateral-car-ratio",
        type=float,
        default=0.35,
        help="두 조각의 횡방향 어긋남 상한 = 차량 대각선 x 이 값",
    )
    parser.add_argument(
        "--virtual-line-thickness-scale",
        type=float,
        default=1.2,
        help="bridge 두께 = 실측 차선 두께 x 이 값(최대 16px)",
    )
    parser.add_argument("--virtual-line-min-thickness", type=int, default=3, help="bridge 최소 두께(px)")

    # ── 번호판 OCR과 화면 표시 ──
    parser.add_argument("--no-ocr", action="store_true", help="번호판 OCR 비활성화. paddle 계열 패키지가 없어도 실행됨")
    parser.add_argument("--plate-text-scale", type=float, default=1.4, help="화면에 그리는 번호판 글자 크기 배율")
    parser.add_argument("--plate-text-thickness", type=int, default=3, help="화면에 그리는 번호판 글자 두께")
    parser.add_argument("--hide-lane-masks", action="store_true", help="차선 마스크 오버레이를 그리지 않음")
    parser.add_argument("--hide-virtual-lanes", action="store_true", help="청록 가상선을 그리지 않음")
    parser.add_argument(
        "--show-lane-rois", action="store_true", help="ROI 상자와 진단 라벨(종류/occ/gap/X/투표수)을 표시"
    )
    return parser.parse_args()


def validate_cli_args(args: argparse.Namespace) -> None:
    if Path(args.tracker).name.lower() == "botsort.yaml":
        raise ValueError(
            "botsort.yaml is not allowed because its default GMC uses sparseOptFlow; use bytetrack.yaml or a verified no-flow tracker"
        )
    if args.chain_roi_cell_length <= 0:
        raise ValueError("--chain-roi-cell-length must be positive")
    if args.chain_roi_min_cells < 1 or args.chain_roi_max_cells < args.chain_roi_min_cells:
        raise ValueError("chain ROI cell count limits are invalid")
    if not 0 <= args.chain_roi_recenter_ratio <= 1:
        raise ValueError("--chain-roi-recenter-ratio must be in [0, 1]")
    if not 0 <= args.chain_roi_angle_blend <= 1:
        raise ValueError("--chain-roi-angle-blend must be in [0, 1]")
    if args.chain_roi_line_thickness < 1:
        raise ValueError("--chain-roi-line-thickness must be at least 1")
    if not 0 < args.roi_min_width <= args.roi_max_width:
        raise ValueError("ROI width limits are invalid")
    if not 0 < args.roi_min_length <= args.roi_max_length:
        raise ValueError("ROI length limits are invalid")
    if args.border_margin < 0:
        raise ValueError("--border-margin cannot be negative")
    if args.lane_vote_window_frames < 1:
        raise ValueError("--lane-vote-window-frames must be at least 1")
    if not 1 <= args.lane_vote_min_samples <= args.lane_vote_window_frames:
        raise ValueError("--lane-vote-min-samples must be within the vote window")
    if not 0.5 <= args.lane_vote_lock_ratio <= 1:
        raise ValueError("--lane-vote-lock-ratio must be in [0.5, 1]")
    if not args.lane_vote_lock_ratio <= args.lane_vote_switch_ratio <= 1:
        raise ValueError("--lane-vote-switch-ratio must be >= lock ratio and <= 1")
    if args.lane_track_max_missed_frames < 0 or not 0 <= args.lane_track_max_angle_diff <= 90:
        raise ValueError("lane track frame/angle limits are invalid")
    if (
        args.lane_track_lateral_width_ratio <= 0
        or args.lane_track_min_lateral_pixels <= 0
        or args.lane_track_max_longitudinal_pixels <= 0
    ):
        raise ValueError("lane track distance thresholds must be positive")
    if args.continuity_slice_length <= 0 or args.continuity_cross_sample_step <= 0:
        raise ValueError("continuity sampling lengths must be positive")
    if args.continuity_min_lane_samples < 1:
        raise ValueError("--continuity-min-lane-samples must be at least 1")
    if not 0 <= args.continuity_min_lane_ratio <= 1:
        raise ValueError("--continuity-min-lane-ratio must be in [0, 1]")
    if not 0 <= args.continuity_vehicle_unknown_ratio <= 1:
        raise ValueError("--continuity-vehicle-unknown-ratio must be in [0, 1]")
    if not 0 <= args.continuity_max_unknown_ratio_for_vote <= 1:
        raise ValueError("--continuity-max-unknown-ratio-for-vote must be in [0, 1]")
    if args.continuity_close_zero_gap_pixels < 0:
        raise ValueError("--continuity-close-zero-gap-pixels cannot be negative")
    if args.continuity_dotted_min_gap_pixels <= 0 or args.continuity_dotted_min_gaps < 1:
        raise ValueError("dotted continuity gap thresholds are invalid")
    if not 0 < args.solid_contact_box_width_ratio <= 1:
        raise ValueError("--solid-contact-box-width-ratio must be in (0, 1]")
    if not 0 < args.solid_contact_box_height_ratio <= 1:
        raise ValueError("--solid-contact-box-height-ratio must be in (0, 1]")
    if args.solid_contact_min_pixels < 1:
        raise ValueError("--solid-contact-min-pixels must be at least 1")
    if args.solid_contact_hold_seconds <= 0:
        raise ValueError("--solid-contact-hold-seconds must be positive")
    if args.solid_center_radius < 0 or args.solid_center_min_pixels < 1:
        raise ValueError("solid center radius/minimum pixels are invalid")
    if (
        args.occlusion_box_margin_ratio < 0
        or args.occlusion_max_gap_car_ratio <= 0
        or args.occlusion_max_lateral_car_ratio <= 0
        or (args.virtual_line_thickness_scale <= 0)
        or (args.virtual_line_min_thickness < 1)
    ):
        raise ValueError("occlusion bridge parameters are invalid")
    if not 0 < args.output_scale <= 1:
        raise ValueError("--output-scale must be in (0, 1]")


def axial_angle_difference(a_deg: float, b_deg: float) -> float:
    """Smallest angle difference for unoriented axes, in [0, 90]."""
    return abs((a_deg - b_deg + 90.0) % 180.0 - 90.0)


# 차선 seg 폴리곤마다 lanege geometry 정보 생성 (면적이 너무 작거나 두께가 1px 이하이거나 종횡비가 너무 작은 경우 제외)
def build_lane_geometries(polygons: list[np.ndarray]) -> list[LaneGeometry]:
    geometries: list[LaneGeometry] = []
    for index, polygon in enumerate(polygons, start=1):
        area, thickness, aspect_ratio = mask_geometry(polygon)
        if area < MIN_MASK_AREA or thickness < 1.0 or aspect_ratio < MIN_ASPECT_RATIO:
            continue
        geometries.append(
            LaneGeometry(
                instance_id=index,
                polygon=np.rint(polygon).astype(np.int32),
                center=polygon_center(polygon),
                area=area,
                thickness=thickness,
                aspect_ratio=aspect_ratio,
                angle_deg=pca_angle_deg(polygon),
            )
        )
    return geometries


# 각 차선 polygon에 대해 instance_map 생성 (차선 seg 마스크에서 각 차선 seg마다 고유 id 부여)
def build_instance_map(height: int, width: int, polygons: list[np.ndarray]) -> np.ndarray:
    instance_map = np.zeros((height, width), dtype=np.int32)
    for instance_id, polygon in enumerate(polygons, start=1):
        cv2.fillPoly(instance_map, [np.rint(polygon).astype(np.int32)], instance_id)
    return instance_map


# 차선 중심과 가장 가까운 차량을 찾아 차량 대각선 길이 반환
def nearest_vehicle_scale(lane_center: tuple[float, float], per_car: dict[int, dict]) -> float:
    if not per_car:
        return 0.0
    lx, ly = lane_center
    _, info = min(per_car.items(), key=lambda item: (item[1]["cx"] - lx) ** 2 + (item[1]["cy"] - ly) ** 2)
    return math.hypot(float(info["box_w"]), float(info["box_h"]))


def clipped_bounds(polygon: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y, box_width, box_height = cv2.boundingRect(polygon)
    x1, y1 = (max(0, x), max(0, y))
    x2, y2 = (min(width, x + box_width), min(height, y + box_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def overlapping_instance_ids(
    instance_map: np.ndarray, roi_polygon: np.ndarray, min_overlap_pixels: int
) -> dict[int, int]:
    height, width = instance_map.shape
    bounds = clipped_bounds(roi_polygon, width, height)
    if bounds is None:
        return {}
    x1, y1, x2, y2 = bounds
    local_roi = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    translated = roi_polygon - np.array([x1, y1], dtype=np.int32)
    cv2.fillPoly(local_roi, [translated], 1)
    ids = instance_map[y1:y2, x1:x2][local_roi.astype(bool)]
    ids = ids[ids > 0]
    if ids.size == 0:
        return {}
    unique_ids, counts = np.unique(ids, return_counts=True)
    return {
        int(instance_id): int(count) for instance_id, count in zip(unique_ids, counts) if count >= min_overlap_pixels
    }


def expanded_car_rect(info: dict, width: int, height: int, margin_ratio: float) -> tuple[int, int, int, int]:
    box = np.asarray(info["box"], dtype=np.float32)
    vehicle_scale = math.hypot(float(info["box_w"]), float(info["box_h"]))
    margin = vehicle_scale * margin_ratio
    x1 = max(0, int(math.floor(box[0] - margin)))
    y1 = max(0, int(math.floor(box[1] - margin)))
    x2 = min(width, int(math.ceil(box[2] + margin)))
    y2 = min(height, int(math.ceil(box[3] + margin)))
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


# 차량 객체 때문에 차선 끊어진거 gap 계산
def fragment_projection_gap(
    first_polygon: np.ndarray, second_polygon: np.ndarray, angle_deg: float
) -> tuple[float, tuple[float, float], float]:
    radians = math.radians(angle_deg)
    direction = np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    first_projection = np.asarray(first_polygon, dtype=np.float32) @ direction
    second_projection = np.asarray(second_polygon, dtype=np.float32) @ direction
    first_interval = (float(first_projection.min()), float(first_projection.max()))
    second_interval = (float(second_projection.min()), float(second_projection.max()))
    if first_interval[0] <= second_interval[0]:
        left, right = (first_interval, second_interval)
    else:
        left, right = (second_interval, first_interval)
    gap_interval = (left[1], right[0])
    gap = max(0.0, gap_interval[1] - gap_interval[0])
    lateral = abs(
        float((np.asarray(polygon_center(second_polygon)) - np.asarray(polygon_center(first_polygon))) @ normal)
    )
    return (gap, gap_interval, lateral)


# 차량으로 분리돼 점선처럼 보이는 두 조각을 OCCLUDED_SOLID로 재분류하는 기존 로직.
# 반환되는 옛 bridge 목록은 현재 stable_solid_fragment_bridges에서 사용하지 않지만,
# overridden decisions는 build_solid_id_map의 실선 판정에 실제로 사용된다.
# v5.3.1: apply_occlusion_overrides() 와 clip_lane_axis_to_rect() 를 삭제했다.
# 진입 조건이 lane_type == "DOTTED" AND allow_occlusion_override 인데,
# stabilize_lane_decisions() 가 allow_occlusion_override = (stable_type == "SOLID") 로,
# lane_type = stable_type or raw_type 으로 설정하므로 두 조건은 동시에 성립할 수 없다.
# 6가지 상태 조합 전수 확인 결과 도달 불가능한 코드였고 항상 (입력 그대로, []) 를 반환했다.
# 따라서 OCCLUDED_SOLID 는 한 번도 생성되지 않았다.


# solid 맵 화면 표시
def build_solid_id_map(
    height: int, width: int, polygons: list[np.ndarray], decisions: list[LaneDecision]
) -> np.ndarray:
    solid_ids = {
        decision.instance_id for decision in decisions if decision.lane_type == "SOLID" and decision.temporal_confirmed
    }
    solid_map = np.zeros((height, width), dtype=np.int32)
    for instance_id in solid_ids:
        polygon = np.rint(polygons[instance_id - 1]).astype(np.int32)
        cv2.fillPoly(solid_map, [polygon], instance_id)
    return solid_map


def draw_virtual_lane_overlay(frame: np.ndarray, virtual_mask: np.ndarray) -> np.ndarray:
    active = virtual_mask > 0
    if not np.any(active):
        return frame
    overlay = frame.copy()
    overlay[active] = COLOR_VIRTUAL_SOLID
    output = frame.copy()
    output[active] = cv2.addWeighted(frame, 0.25, overlay, 0.75, 0)[active]
    return output


def draw_vehicle_box(frame: np.ndarray, box: np.ndarray, car_id: int, flagged: bool) -> None:
    x1, y1, x2, y2 = map(int, box)
    color = COLOR_FIRE if flagged else COLOR_CAR
    thickness = 4 if flagged else 2
    label = f"Car #{car_id} SOLID LINE CROSSING" if flagged else f"Car #{car_id}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)


# 영상 실행
def process_video(args: argparse.Namespace) -> None:
    validate_cli_args(args)
    det_model_path = Path(args.det_model).expanduser().resolve()
    lane_model_path = Path(args.lane_model).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path, label in (
        (det_model_path, "detection model"),
        (lane_model_path, "lane model"),
        (input_path, "input video"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    render_total = (
        min(total_frames, args.max_frames)
        if args.max_frames > 0 and total_frames > 0
        else args.max_frames or total_frames
    )
    out_width = max(2, int(round(width * args.output_scale)) // 2 * 2)
    out_height = max(2, int(round(height * args.output_scale)) // 2 * 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    csv_file = csv_writer = None
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_COLUMNS)
    det_model = YOLO(str(det_model_path))
    lane_model = YOLO(str(lane_model_path))
    tracker = VehicleTracker(width, height, args.car_class)
    ocr_engine = stabilizer = None
    if not args.no_ocr:
        _prepare_paddleocr_environment()
        from paddleocr import PaddleOCR

        paddle_engine = PaddleOCR(
            lang="en",
            text_detection_model_name=PADDLE_DET_MODEL,
            text_recognition_model_name=PADDLE_REC_MODEL,
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            device=normalize_paddle_device(args.paddle_device or args.device),
        )
        ocr_engine = KhoiPaddleOCR(paddle_engine)
        stabilizer = PlateTextStabilizer(min_votes=3, history=30)
    det_results = det_model.track(
        source=str(input_path),
        conf=args.det_conf,
        iou=args.nms_iou,
        imgsz=args.det_imgsz,
        device=args.device,
        tracker=args.tracker,
        stream=True,
        persist=True,
        verbose=False,
    )
    flagged_ids: set[int] = set()
    fired_frame: dict[int, int] = {}
    fired_lane: dict[int, int] = {}
    plate_text_by_car: dict[int, str] = {}
    try:
        with tqdm(
            total=render_total or None, desc="Solid lane crossing + OCR", unit="frame", dynamic_ncols=True
        ) as progress:
            for frame_index, result in enumerate(det_results):
                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break
                original = result.orig_img
                frame = original.copy()
                per_car, cam = tracker.process(result)
                lane_result = lane_model.predict(
                    source=original, imgsz=args.lane_imgsz, conf=args.lane_conf, device=args.device, verbose=False
                )[0]
                polygons = (
                    []
                    if lane_result.masks is None
                    else [polygon for polygon in lane_result.masks.xy if len(polygon) >= 3]
                )
                decisions, _ = classify_lane_continuity(polygons, per_car, width, height, args)
                decisions = stabilize_lane_decisions(decisions=decisions, cam=cam, frame_index=frame_index)
                bridges = stable_solid_fragment_bridges(decisions, polygons, per_car, width, height, args)
                visible_solid_map = build_solid_id_map(height, width, polygons, decisions)
                solid_map, virtual_mask = add_fragment_bridges_to_solid_map(visible_solid_map, bridges)
                if not args.hide_lane_masks:
                    frame = draw_lane_analysis_overlay(frame, polygons, decisions, args.show_lane_rois)
                if not args.hide_virtual_lanes:
                    frame = draw_virtual_lane_overlay(frame, virtual_mask)
                car_boxes: dict[int, np.ndarray] = {}
                for car_id, info in per_car.items():
                    box = info["box"]
                    car_boxes[car_id] = box
                    center = np.array([info["cx"], info["cy"]], dtype=np.float32)
                    vehicle_scale = math.hypot(float(info["box_w"]), float(info["box_h"]))
                    # v5.3.1: hit 는 아직 채워지지 않는다. CSV 의 차선 관련 17개 열이
                    # 전 행 빈 값인 원인이며, 채우는 작업은 별도 버전에서 다룬다.
                    hit = CrossingHit()
                    x1, y1, x2, y2 = map(float, box)
                    margin = args.border_margin
                    border_clipped = bool(
                        margin > 0 and (x1 <= margin or y1 <= margin or x2 >= width - margin or (y2 >= height - margin))
                    )
                    final_trigger = evaluate_solid_crossing_trigger(
                        border_clipped=border_clipped,
                        box=box,
                        solid_map=solid_map,
                        car_id=car_id,
                        frame_index=frame_index,
                        fps=fps,
                    )
                    combined_fired = final_trigger.fired
                    if combined_fired and car_id not in flagged_ids:
                        flagged_ids.add(car_id)
                        fired_frame[car_id] = frame_index
                        fired_lane[car_id] = final_trigger.instance_id
                        print(
                            f"[SOLID-CROSSING] frame={frame_index} car={car_id} lane={final_trigger.instance_id} reason={final_trigger.reason} heading_dev=n/a hold={final_trigger.solid_contact_hold_seconds:.2f}s heading_hold=0.00"
                        )
                    cv2.circle(
                        frame, tuple(np.rint(center).astype(int)), args.center_radius, COLOR_CENTER, -1, cv2.LINE_AA
                    )
                    draw_vehicle_box(frame, box, car_id, car_id in flagged_ids)
                    if csv_writer is not None:
                        delta = info.get("stab_delta")
                        delta_array = np.asarray(delta, dtype=np.float32) if delta is not None else None
                        csv_row = [
                            frame_index,
                            car_id,
                            f"{info['cx']:.2f}",
                            f"{info['cy']:.2f}",
                            f"{info['box_w']:.2f}",
                            f"{info['box_h']:.2f}",
                            f"{vehicle_scale:.2f}",
                            f"{delta_array[0]:.4f}" if delta_array is not None else "",
                            f"{delta_array[1]:.4f}" if delta_array is not None else "",
                            f"{float(np.linalg.norm(delta_array)):.4f}" if delta_array is not None else "",
                            hit.instance_id if hit.instance_id >= 0 else "",
                            hit.lane_type,
                            hit.seg_count if hit.instance_id >= 0 else "",
                            hit.dotted_threshold if hit.instance_id >= 0 else "",
                            f"{hit.roi_length:.2f}" if hit.instance_id >= 0 else "",
                            f"{hit.roi_width:.2f}" if hit.instance_id >= 0 else "",
                            f"{hit.roi_visible_ratio:.3f}" if hit.instance_id >= 0 else "",
                            f"{hit.normal_motion_ratio:.3f}" if hit.instance_id >= 0 else "",
                            "",  # baseline_deg        (optical flow 제거로 항상 빈 값)
                            "0.000",  # baseline_coherence
                            "",  # heading_deviation_deg
                            "0.000",  # heading_hold_ratio
                            0,  # heading_fired
                            0,  # crossing_in_hold
                            int(border_clipped),
                            int(combined_fired),
                            int(car_id in flagged_ids),
                            plate_text_by_car.get(car_id, ""),
                            int(bool(cam.get("zooming"))),
                            int(bool(cam.get("suspended"))),
                        ]
                        if "trigger_reason" in CSV_COLUMNS:
                            csv_row.extend(
                                [
                                    int(final_trigger.solid_contact),
                                    final_trigger.solid_contact_pixels,
                                    final_trigger.reason,
                                ]
                            )
                        if "solid_contact_hold_seconds" in CSV_COLUMNS:
                            csv_row.extend(
                                [
                                    final_trigger.solid_contact_hold_frames,
                                    f"{final_trigger.solid_contact_hold_seconds:.3f}",
                                ]
                            )
                        if "solid_center_crossed" in CSV_COLUMNS:
                            csv_row.extend([int(final_trigger.solid_center_crossed), final_trigger.solid_center_pixels])
                        if "lane_track_id" in CSV_COLUMNS:
                            csv_row.extend(
                                [
                                    hit.raw_lane_type,
                                    hit.stable_lane_type,
                                    hit.lane_track_id if hit.lane_track_id >= 0 else "",
                                    hit.vote_samples,
                                    f"{hit.vote_solid_ratio:.3f}",
                                    f"{hit.vote_dotted_ratio:.3f}",
                                ]
                            )
                        if "lane_continuity_occupancy" in CSV_COLUMNS:
                            csv_row.extend(
                                [
                                    f"{hit.continuity_occupancy:.3f}",
                                    hit.continuity_gap_count,
                                    f"{hit.continuity_longest_gap:.1f}",
                                    f"{hit.continuity_unknown_ratio:.3f}",
                                ]
                            )
                        csv_writer.writerow(csv_row)
                if ocr_engine is not None and result.boxes is not None and (len(result.boxes) > 0):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    track_ids = (
                        result.boxes.id.cpu().numpy().astype(int)
                        if result.boxes.id is not None
                        else np.full(len(classes), -1)
                    )
                    car_indices = [
                        index
                        for index, class_id in enumerate(classes)
                        if class_id == args.car_class and track_ids[index] >= 0
                    ]
                    plate_drawn: set[int] = set()
                    for plate_index, class_id in enumerate(classes):
                        if class_id != args.plate_class or not car_indices:
                            continue
                        score, car_index = max(
                            (
                                (overlap_score(boxes[index], boxes[plate_index], ASSOCIATION_METRIC), index)
                                for index in car_indices
                            ),
                            key=lambda item: item[0],
                        )
                        if score < ASSOCIATION_THRESHOLD:
                            continue
                        car_id = int(track_ids[car_index])
                        if car_id not in flagged_ids:
                            continue
                        x1, y1, x2, y2 = clip_box(boxes[plate_index], width, height)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        text, confidence = run_ocr(ocr_engine, original[y1:y2, x1:x2])
                        stable = stabilizer.update(car_id, text, confidence)
                        if stable:
                            plate_text_by_car[car_id] = stable
                        shown = plate_text_by_car.get(car_id, stable)
                        draw_box(frame, boxes[plate_index], COLOR_PLATE)
                        draw_plate_text(
                            frame,
                            boxes[plate_index],
                            shown,
                            scale=args.plate_text_scale,
                            thickness=args.plate_text_thickness,
                        )
                        plate_drawn.add(car_id)
                    for car_id in flagged_ids:
                        if car_id in plate_drawn or car_id not in car_boxes:
                            continue
                        text = plate_text_by_car.get(car_id)
                        if not text:
                            continue
                        car_box = car_boxes[car_id]
                        anchor = np.array([car_box[0], car_box[1] - 40, car_box[2], car_box[1] - 40], dtype=np.float32)
                        draw_plate_text(
                            frame, anchor, text, scale=args.plate_text_scale, thickness=args.plate_text_thickness
                        )
                cv2.putText(
                    frame,
                    f"frame={frame_index} lane_masks={len(polygons)} solid={sum((d.lane_type == 'SOLID' for d in decisions))} occluded=0 dotted={sum((d.lane_type == 'DOTTED' for d in decisions))} bridges={len(bridges)} flagged={len(flagged_ids)}",
                    (24, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if (out_width, out_height) != (width, height):
                    frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                progress.update(1)
                progress.set_postfix(
                    cars=len(per_car),
                    solid=sum((d.lane_type == "SOLID" for d in decisions)),
                    bridge=len(bridges),
                    flagged=len(flagged_ids),
                    refresh=False,
                )
    finally:
        writer.release()
        if csv_file is not None:
            csv_file.close()
    if flagged_ids:
        details = "  ".join(
            (
                f"#{car_id}@f{fired_frame[car_id]} lane={fired_lane[car_id]} plate={plate_text_by_car.get(car_id, '?')}"
                for car_id in sorted(flagged_ids, key=lambda cid: fired_frame[cid])
            )
        )
        print(f"[DETECTED] {details}")
    else:
        print("[DETECTED] none")
    if args.csv:
        print(f"[CSV] {Path(args.csv).expanduser().resolve()}")
    print(f"[DONE] {output_path} ({out_width}x{out_height})")


# 짧은 ROI 박스 하나
@dataclass(frozen=True)
class ChainCell:
    polygon: np.ndarray
    center: tuple[float, float]
    angle_deg: float


@dataclass(frozen=True)
class ChainVisual:
    cells: tuple[ChainCell, ...]
    member_ids: tuple[int, ...]


CHAIN_VISUALS: list[ChainVisual] = []
CHAIN_ROI_LINE_THICKNESS = 1


def odd_cell_count(desired_length: float, args: argparse.Namespace) -> int:
    count = max(1, int(math.ceil(desired_length / args.chain_roi_cell_length)))
    count = max(args.chain_roi_min_cells, min(args.chain_roi_max_cells, count))
    if count % 2 == 0:
        if count < args.chain_roi_max_cells:
            count += 1
        elif count > args.chain_roi_min_cells:
            count -= 1
    return count


def axis_vector(angle_deg: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    return np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)


def aligned_blended_angle(current_deg: float, target_deg: float, blend: float) -> float:
    current = axis_vector(current_deg)
    target = axis_vector(target_deg)
    if float(np.dot(current, target)) < 0:
        target = -target
    mixed = current * (1.0 - blend) + target * blend
    norm = float(np.linalg.norm(mixed))
    if norm < 1e-06:
        return current_deg
    mixed /= norm
    return math.degrees(math.atan2(float(mixed[1]), float(mixed[0])))


def make_cell(center: np.ndarray, angle_deg: float, length: float, width: float) -> ChainCell:
    polygon = np.rint(
        cv2.boxPoints(((float(center[0]), float(center[1])), (float(length), float(width)), float(angle_deg)))
    ).astype(np.int32)
    return ChainCell(polygon=polygon, center=(float(center[0]), float(center[1])), angle_deg=float(angle_deg))


def compatible_overlaps(
    cell: ChainCell,
    seed_angle_deg: float,
    instance_map: np.ndarray,
    geometry_by_id: dict[int, LaneGeometry],
) -> dict[int, int]:
    overlaps = overlapping_instance_ids(instance_map, cell.polygon, 1)
    return {
        instance_id: pixels
        for instance_id, pixels in overlaps.items()
        if instance_id in geometry_by_id
        and axial_angle_difference(seed_angle_deg, geometry_by_id[instance_id].angle_deg)
        <= MAX_INSTANCE_ANGLE_DIFF
    }


def corrected_next_cell(
    previous: ChainCell,
    direction_sign: float,
    seed_angle_deg: float,
    cell_length: float,
    roi_width: float,
    instance_map: np.ndarray,
    geometry_by_id: dict[int, LaneGeometry],
    args: argparse.Namespace,
) -> ChainCell:
    direction = axis_vector(previous.angle_deg)
    expected = np.asarray(previous.center, dtype=np.float32) + direction * direction_sign * cell_length
    preliminary = make_cell(expected, previous.angle_deg, cell_length, roi_width)
    overlaps = compatible_overlaps(preliminary, seed_angle_deg, instance_map, geometry_by_id)
    if not overlaps:
        return preliminary
    candidate_id = max(overlaps, key=overlaps.get)
    candidate = geometry_by_id[candidate_id]
    candidate_points = candidate.polygon.astype(np.float32)
    distances = np.linalg.norm(candidate_points - expected[None, :], axis=1)
    closest = candidate_points[int(np.argmin(distances))]
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    lateral = float(np.dot(closest - expected, normal))
    lateral = float(np.clip(lateral, -roi_width * 0.5, roi_width * 0.5))
    corrected_center = expected + normal * lateral * args.chain_roi_recenter_ratio
    corrected_angle = aligned_blended_angle(previous.angle_deg, candidate.angle_deg, args.chain_roi_angle_blend)
    return make_cell(corrected_center, corrected_angle, cell_length, roi_width)


# 기준 차선에서 앞뒤 양방향으로 roi cell 확장
def build_chain(
    seed: LaneGeometry,
    cell_count: int,
    cell_length: float,
    roi_width: float,
    instance_map: np.ndarray,
    geometry_by_id: dict[int, LaneGeometry],
    args: argparse.Namespace,
) -> tuple[ChainCell, ...]:
    center = make_cell(np.asarray(seed.center, dtype=np.float32), seed.angle_deg, cell_length, roi_width)
    side_count = cell_count // 2
    forward: list[ChainCell] = []
    previous = center
    for _ in range(side_count):
        previous = corrected_next_cell(
            previous, 1.0, seed.angle_deg, cell_length, roi_width, instance_map, geometry_by_id, args
        )
        forward.append(previous)
    backward: list[ChainCell] = []
    previous = center
    for _ in range(side_count):
        previous = corrected_next_cell(
            previous, -1.0, seed.angle_deg, cell_length, roi_width, instance_map, geometry_by_id, args
        )
        backward.append(previous)
    return tuple([*reversed(backward), center, *forward])


def chain_member_ids(
    cells: tuple[ChainCell, ...],
    seed_angle_deg: float,
    instance_map: np.ndarray,
    geometry_by_id: dict[int, LaneGeometry],
) -> tuple[int, ...]:
    overlap_totals: dict[int, int] = {}
    for cell in cells:
        for instance_id, pixels in compatible_overlaps(cell, seed_angle_deg, instance_map, geometry_by_id).items():
            overlap_totals[instance_id] = overlap_totals.get(instance_id, 0) + pixels
    return tuple(
        sorted((instance_id for instance_id, pixels in overlap_totals.items() if pixels >= MIN_OVERLAP_PIXELS))
    )


def classify_connected_lane_groups(
    polygons: list[np.ndarray], per_car: dict[int, dict], width: int, height: int, args: argparse.Namespace
) -> tuple[list[LaneDecision], np.ndarray]:
    global CHAIN_VISUALS
    CHAIN_VISUALS = []
    geometries = build_lane_geometries(polygons)
    geometry_by_id = {geometry.instance_id: geometry for geometry in geometries}
    instance_map = build_instance_map(height, width, polygons)
    decisions_by_id: dict[int, LaneDecision] = {}
    assigned: set[int] = set()
    seeds = sorted(geometries, key=lambda geometry: geometry.area, reverse=True)
    for seed in seeds:
        if seed.instance_id in assigned:
            continue
        vehicle_scale = nearest_vehicle_scale(seed.center, per_car)
        desired_width = max(seed.thickness * args.roi_width_lane_scale, vehicle_scale * args.roi_width_car_scale)
        desired_length = max(seed.thickness * args.roi_length_lane_scale, vehicle_scale * args.roi_length_car_scale)
        roi_width = clamp(desired_width, args.roi_min_width, args.roi_max_width)
        requested_length = clamp(desired_length, args.roi_min_length, args.roi_max_length)
        cell_count = odd_cell_count(requested_length, args)
        cells = build_chain(seed, cell_count, args.chain_roi_cell_length, roi_width, instance_map, geometry_by_id, args)
        member_ids = chain_member_ids(cells, seed.angle_deg, instance_map, geometry_by_id)
        member_ids = tuple((instance_id for instance_id in member_ids if instance_id not in assigned))
        if seed.instance_id not in member_ids:
            member_ids = tuple(sorted((*member_ids, seed.instance_id)))
        all_points = np.concatenate([cell.polygon for cell in cells], axis=0)
        display_hull = cv2.convexHull(all_points).reshape(-1, 2)
        # lane_type 은 여기서 정하지 않는다. 바로 다음 단계인 classify_lane_continuity() 가
        # 종방향 1/0/-1/-2 신호로 모든 decision 을 덮어쓴다.
        for member_id in member_ids:
            member = geometry_by_id[member_id]
            decisions_by_id[member_id] = LaneDecision(
                instance_id=member_id,
                roi_polygon=display_hull,
                center=member.center,
                thickness=member.thickness,
                lane_type="",
                member_ids=member_ids,
                roi_width=roi_width,
                angle_deg=member.angle_deg,
            )
        assigned.update(member_ids)
        CHAIN_VISUALS.append(ChainVisual(cells=cells, member_ids=member_ids))
    for geometry in geometries:
        if geometry.instance_id in decisions_by_id:
            continue
        decisions_by_id[geometry.instance_id] = LaneDecision(
            instance_id=geometry.instance_id,
            roi_polygon=geometry.polygon,
            center=geometry.center,
            thickness=geometry.thickness,
            lane_type="",
            member_ids=(geometry.instance_id,),
            roi_width=geometry.thickness,
            angle_deg=geometry.angle_deg,
        )
    return (list(decisions_by_id.values()), instance_map)


@dataclass(frozen=True)
class SolidContactConfig:
    box_width_ratio: float = 0.8
    box_height_ratio: float = 0.8
    min_pixels: int = 12


CONTACT_CONFIG = SolidContactConfig()


def centered_contact_rect(
    box: np.ndarray, width: int, height: int, config: SolidContactConfig
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = map(float, box)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    horizontal_margin = box_width * (1.0 - config.box_width_ratio) * 0.5
    vertical_margin = box_height * (1.0 - config.box_height_ratio) * 0.5
    left = max(0, int(round(x1 + horizontal_margin)))
    top = max(0, int(round(y1 + vertical_margin)))
    right = min(width, int(round(x2 - horizontal_margin)))
    bottom = min(height, int(round(y2 - vertical_margin)))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def solid_contact_instance(box: np.ndarray, solid_map: np.ndarray, config: SolidContactConfig) -> tuple[int, int]:
    height, width = solid_map.shape
    rect = centered_contact_rect(box, width, height, config)
    if rect is None:
        return (-1, 0)
    x1, y1, x2, y2 = rect
    ids = solid_map[y1:y2, x1:x2]
    ids = ids[ids > 0]
    if ids.size == 0:
        return (-1, 0)
    instance_ids, counts = np.unique(ids, return_counts=True)
    best_index = int(np.argmax(counts))
    pixels = int(counts[best_index])
    if pixels < config.min_pixels:
        return (-1, pixels)
    return (int(instance_ids[best_index]), pixels)


@dataclass
class ContactRun:
    last_frame: int
    consecutive_frames: int
    instance_id: int
    contact_pixels: int


CONTACT_RUNS: dict[int, ContactRun] = {}


@dataclass(frozen=True)
# 차선 시간 추적 및 투표 구성
class LaneVoteConfig:
    window_frames: int = 30  # 투표창
    min_samples: int = 10  # 10개 표본
    lock_ratio: float = 0.7  # 70퍼 이상이면 유지 확정
    switch_ratio: float = 0.8  # 반대타입이 80퍼 이상이변 변경
    max_missed_frames: int = 15
    max_angle_diff: float = 20.0
    lateral_width_ratio: float = 1.5
    min_lateral_pixels: float = 80.0
    max_longitudinal_pixels: float = 2000.0


@dataclass
class LaneTemporalTrack:
    track_id: int
    center: np.ndarray
    angle_deg: float
    roi_width: float
    last_frame: int
    missed_frames: int = 0
    stable_type: str = ""
    history: deque[str] = field(default_factory=deque)


@dataclass(frozen=True)
class CurrentLaneGroup:
    member_ids: tuple[int, ...]
    decision_indices: tuple[int, ...]
    center: np.ndarray
    angle_deg: float
    roi_width: float
    raw_type: str


VOTE_CONFIG = LaneVoteConfig()
LANE_TRACKS: dict[int, LaneTemporalTrack] = {}
NEXT_LANE_TRACK_ID = 1


def axial_mean_deg(angles: list[float]) -> float:
    doubled = np.radians(np.asarray(angles, dtype=np.float64) * 2.0)
    sine = float(np.sin(doubled).sum())
    cosine = float(np.cos(doubled).sum())
    if abs(sine) < 1e-09 and abs(cosine) < 1e-09:
        return angles[0]
    return math.degrees(math.atan2(sine, cosine)) * 0.5


def current_lane_groups(decisions: list[LaneDecision]) -> list[CurrentLaneGroup]:
    indices_by_members: dict[tuple[int, ...], list[int]] = {}
    for index, decision in enumerate(decisions):
        members = tuple(sorted(decision.member_ids or (decision.instance_id,)))
        indices_by_members.setdefault(members, []).append(index)
    visuals = {tuple(sorted(visual.member_ids)): visual for visual in CHAIN_VISUALS}
    groups: list[CurrentLaneGroup] = []
    for member_ids, indices in indices_by_members.items():
        group_decisions = [decisions[index] for index in indices]
        visual = visuals.get(member_ids)
        if visual is not None and visual.cells:
            centers = np.asarray([cell.center for cell in visual.cells], dtype=np.float32)
            center = centers.mean(axis=0)
            angle_deg = axial_mean_deg([cell.angle_deg for cell in visual.cells])
        else:
            center = np.asarray([decision.center for decision in group_decisions], dtype=np.float32).mean(axis=0)
            angle_deg = axial_mean_deg([decision.angle_deg for decision in group_decisions])
        raw_type = max(
            ("SOLID", "DOTTED"),
            key=lambda lane_type: sum((decision.lane_type == lane_type for decision in group_decisions)),
        )
        groups.append(
            CurrentLaneGroup(
                member_ids=member_ids,
                decision_indices=tuple(indices),
                center=center,
                angle_deg=angle_deg,
                roi_width=float(np.mean([decision.roi_width for decision in group_decisions])),
                raw_type=raw_type,
            )
        )
    return groups


def track_match_score(group: CurrentLaneGroup, track: LaneTemporalTrack, config: LaneVoteConfig) -> float | None:
    angle_diff = axial_angle_difference(group.angle_deg, track.angle_deg)
    if angle_diff > config.max_angle_diff:
        return None
    direction = axis_vector(track.angle_deg)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    delta = group.center - track.center
    lateral = abs(float(np.dot(delta, normal)))
    longitudinal = abs(float(np.dot(delta, direction)))
    max_lateral = max(config.min_lateral_pixels, max(group.roi_width, track.roi_width) * config.lateral_width_ratio)
    if lateral > max_lateral or longitudinal > config.max_longitudinal_pixels:
        return None
    return (
        lateral / max_lateral
        + longitudinal / config.max_longitudinal_pixels
        + angle_diff / max(config.max_angle_diff, 1.0)
    )


def match_groups_to_tracks(groups: list[CurrentLaneGroup], config: LaneVoteConfig) -> dict[int, int]:
    candidates: list[tuple[float, int, int]] = []
    for group_index, group in enumerate(groups):
        for track_id, track in LANE_TRACKS.items():
            score = track_match_score(group, track, config)
            if score is not None:
                candidates.append((score, group_index, track_id))
    matches: dict[int, int] = {}
    used_tracks: set[int] = set()
    for _, group_index, track_id in sorted(candidates):
        if group_index in matches or track_id in used_tracks:
            continue
        matches[group_index] = track_id
        used_tracks.add(track_id)
    return matches


def vote_ratios(track: LaneTemporalTrack) -> tuple[int, float, float]:
    samples = len(track.history)
    if samples == 0:
        return (0, 0.0, 0.0)
    solid_ratio = sum((item == "SOLID" for item in track.history)) / samples
    dotted_ratio = 1.0 - solid_ratio
    return (samples, solid_ratio, dotted_ratio)


def update_stable_type(track: LaneTemporalTrack, config: LaneVoteConfig) -> tuple[int, float, float]:
    samples, solid_ratio, dotted_ratio = vote_ratios(track)
    if samples < config.min_samples:
        return (samples, solid_ratio, dotted_ratio)
    if not track.stable_type:
        if solid_ratio >= config.lock_ratio:
            track.stable_type = "SOLID"
        elif dotted_ratio >= config.lock_ratio:
            track.stable_type = "DOTTED"
    elif track.stable_type == "SOLID" and dotted_ratio >= config.switch_ratio:
        track.stable_type = "DOTTED"
    elif track.stable_type == "DOTTED" and solid_ratio >= config.switch_ratio:
        track.stable_type = "SOLID"
    return (samples, solid_ratio, dotted_ratio)


def create_lane_track(group: CurrentLaneGroup, frame_index: int, config: LaneVoteConfig) -> LaneTemporalTrack:
    global NEXT_LANE_TRACK_ID
    track = LaneTemporalTrack(
        track_id=NEXT_LANE_TRACK_ID,
        center=group.center.copy(),
        angle_deg=group.angle_deg,
        roi_width=group.roi_width,
        last_frame=frame_index,
        history=deque(maxlen=config.window_frames),
    )
    NEXT_LANE_TRACK_ID += 1
    LANE_TRACKS[track.track_id] = track
    return track


def stabilize_lane_decisions(*, decisions: list[LaneDecision], cam: dict, frame_index: int) -> list[LaneDecision]:
    if cam.get("suspended"):
        LANE_TRACKS.clear()
    for track in LANE_TRACKS.values():
        track.missed_frames += 1
    groups = current_lane_groups(decisions)
    matches = match_groups_to_tracks(groups, VOTE_CONFIG)
    output = list(decisions)
    for group_index, group in enumerate(groups):
        track_id = matches.get(group_index)
        track = LANE_TRACKS[track_id] if track_id is not None else create_lane_track(group, frame_index, VOTE_CONFIG)
        track.center = group.center.copy()
        track.angle_deg = aligned_blended_angle(track.angle_deg, group.angle_deg, 0.35)
        track.roi_width = group.roi_width
        track.last_frame = frame_index
        track.missed_frames = 0
        # X가 임계값보다 많으면 현재 판정에는 근거가 부족하다.
        # 이 프레임은 SOLID/DOTTED 어느 쪽에도 투표하지 않고 기존 이력을 유지한다.
        group_unknown_ratio = max(
            (decisions[index].continuity_unknown_ratio for index in group.decision_indices),
            default=0.0,
        )
        vote_skipped_unknown = group_unknown_ratio > CONTINUITY_CONFIG.max_unknown_ratio_for_vote
        if not vote_skipped_unknown:
            track.history.append(group.raw_type)
        samples, solid_ratio, dotted_ratio = update_stable_type(track, VOTE_CONFIG)
        stable_type = track.stable_type
        confirmed = bool(stable_type)
        lane_type = stable_type or group.raw_type
        allow_occlusion = stable_type == "SOLID"
        for decision_index in group.decision_indices:
            output[decision_index] = replace(
                decisions[decision_index],
                lane_type=lane_type,
                raw_lane_type=group.raw_type,
                stable_lane_type=stable_type,
                lane_track_id=track.track_id,
                vote_samples=samples,
                vote_solid_ratio=solid_ratio,
                vote_dotted_ratio=dotted_ratio,
                temporal_confirmed=confirmed,
                allow_occlusion_override=allow_occlusion,
                temporal_vote_skipped_unknown=vote_skipped_unknown,
            )
    stale_ids = [
        track_id for track_id, track in LANE_TRACKS.items() if track.missed_frames > VOTE_CONFIG.max_missed_frames
    ]
    for track_id in stale_ids:
        del LANE_TRACKS[track_id]
    return output


@dataclass(frozen=True)
class ContinuityConfig:
    slice_length: float = 10.0
    cross_sample_step: float = 2.0
    min_lane_samples: int = 6
    min_lane_ratio: float = 0.02
    vehicle_unknown_ratio: float = 0.3
    max_unknown_ratio_for_vote: float = 0.5
    close_zero_gap_pixels: float = 30.0
    dotted_min_gap_pixels: float = 60.0
    dotted_min_gaps: int = 1


@dataclass(frozen=True)
class ContinuityResult:
    raw_type: str
    signal: tuple[int, ...]
    occupancy: float
    gap_count: int
    longest_gap: float
    unknown_ratio: float


CONTINUITY_CONFIG = ContinuityConfig()

# 종방향 연속성 신호의 slice 값.
# v5.2까지는 화면 밖과 차량 가림이 모두 -1이었다. v5.3은 둘을 분리한다.
# 화면 밖은 "증거가 없는 것"이고 차량 가림은 "가려서 모르는 것"이므로,
# 무효표 판정에는 차량 가림만 사용해야 한다.
SIGNAL_LANE = 1
SIGNAL_EMPTY = 0
SIGNAL_VEHICLE_HIDDEN = -1
SIGNAL_OUT_OF_FRAME = -2


# 전체 차선 마스크와 차량 마스크를 만듦
def build_binary_masks(
    polygons: list[np.ndarray], per_car: dict[int, dict], width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    lane_mask = np.zeros((height, width), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(lane_mask, [np.rint(polygon).astype(np.int32) for polygon in polygons], 1)
    vehicle_mask = np.zeros((height, width), dtype=np.uint8)
    for info in per_car.values():
        x1, y1, x2, y2 = np.rint(info["box"]).astype(int)
        x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
        if x2 >= x1 and y2 >= y1:
            vehicle_mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return (lane_mask, vehicle_mask)


def sample_cell_signal(
    cell: ChainCell,
    roi_width: float,
    cell_length: float,
    lane_mask: np.ndarray,
    vehicle_mask: np.ndarray,
    config: ContinuityConfig,
) -> list[int]:
    height, width = lane_mask.shape
    bin_count = max(1, int(math.ceil(cell_length / config.slice_length)))
    bin_length = cell_length / bin_count
    longitudinal_centers = (np.arange(bin_count, dtype=np.float32) + 0.5) * bin_length - cell_length * 0.5
    longitudinal_offsets = np.linspace(
        -bin_length * 0.4,
        bin_length * 0.4,
        max(1, int(math.ceil(bin_length / config.cross_sample_step))),
        dtype=np.float32,
    )
    lateral_offsets = np.arange(
        -roi_width * 0.5, roi_width * 0.5 + config.cross_sample_step * 0.25, config.cross_sample_step, dtype=np.float32
    )
    angle = math.radians(cell.angle_deg)
    axis = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    center = np.asarray(cell.center, dtype=np.float32)
    points = (
        center[None, None, None, :]
        + (longitudinal_centers[:, None, None, None] + longitudinal_offsets[None, :, None, None])
        * axis[None, None, None, :]
        + lateral_offsets[None, None, :, None] * normal[None, None, None, :]
    )
    xs = np.rint(points[..., 0]).astype(np.int32)
    ys = np.rint(points[..., 1]).astype(np.int32)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    safe_xs = np.clip(xs, 0, width - 1)
    safe_ys = np.clip(ys, 0, height - 1)
    lane = (lane_mask[safe_ys, safe_xs] > 0) & inside
    vehicle = (vehicle_mask[safe_ys, safe_xs] > 0) & inside
    signal: list[int] = []
    for index in range(bin_count):
        inside_count = int(inside[index].sum())
        if inside_count == 0:
            signal.append(SIGNAL_OUT_OF_FRAME)
            continue
        vehicle_ratio = float(vehicle[index].sum()) / inside_count
        if vehicle_ratio >= config.vehicle_unknown_ratio:
            signal.append(SIGNAL_VEHICLE_HIDDEN)
            continue
        visible = inside[index] & ~vehicle[index]
        visible_count = int(visible.sum())
        lane_count = int((lane[index] & visible).sum())
        lane_ratio = lane_count / max(visible_count, 1)
        signal.append(1 if lane_count >= config.min_lane_samples and lane_ratio >= config.min_lane_ratio else 0)
    return signal


def zero_runs(signal: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate([*signal, 1]):
        if value == 0 and start is None:
            start = index
        elif value != 0 and start is not None:
            runs.append((start, index))
            start = None
    return runs


def fill_short_internal_gaps(signal: list[int], max_bins: int) -> list[int]:
    output = list(signal)
    for start, end in zero_runs(signal):
        bounded = start > 0 and end < len(signal) and (signal[start - 1] == 1) and (signal[end] == 1)
        if bounded and end - start <= max_bins:
            output[start:end] = [1] * (end - start)
    return output


def nearest_known_value(signal: list[int], index: int, step: int) -> int | None:
    """Walk outward past unknown slices and return the first observed 0/1 value.

    A dotted gap next to a vehicle is still a dotted gap. Looking only at the
    directly adjacent slice discards that evidence whenever a vehicle happens to
    sit at the end of a dash, which is the common case in dense traffic.
    """
    while 0 <= index < len(signal):
        if signal[index] >= 0:
            return signal[index]
        index += step
    return None


# 내부에 긴 0 구간이 있으면 dotted로 판정
def analyze_signal(signal: list[int], config: ContinuityConfig) -> ContinuityResult:
    close_bins = int(math.floor(config.close_zero_gap_pixels / config.slice_length))
    cleaned = fill_short_internal_gaps(signal, close_bins)
    gap_lengths: list[float] = []
    for start, end in zero_runs(cleaned):
        length = (end - start) * config.slice_length
        if length < config.dotted_min_gap_pixels:
            continue
        # 배열 양 끝의 0은 ROI가 실제 차선보다 길어서 생긴 것이므로 gap이 아니다.
        if start == 0 or end >= len(cleaned):
            continue
        # 차량 가림/화면 밖을 건너뛰고 바깥쪽에 실제 차선이 있는지 확인한다.
        if nearest_known_value(cleaned, start - 1, -1) != SIGNAL_LANE:
            continue
        if nearest_known_value(cleaned, end, 1) != SIGNAL_LANE:
            continue
        gap_lengths.append(length)
    known = sum((value >= 0 for value in cleaned))
    occupied = sum((value == SIGNAL_LANE for value in cleaned))
    hidden = sum((value == SIGNAL_VEHICLE_HIDDEN for value in cleaned))
    out_of_frame = sum((value == SIGNAL_OUT_OF_FRAME for value in cleaned))
    occupancy = occupied / max(known, 1)
    # 화면 밖은 차량 가림이 아니므로 무효표 판정의 분모에서 제외한다.
    unknown_ratio = hidden / max(len(cleaned) - out_of_frame, 1)
    raw_type = "DOTTED" if len(gap_lengths) >= config.dotted_min_gaps else "SOLID"
    return ContinuityResult(
        raw_type=raw_type,
        signal=tuple(cleaned),
        occupancy=occupancy,
        gap_count=len(gap_lengths),
        longest_gap=max(gap_lengths, default=0.0),
        unknown_ratio=unknown_ratio,
    )


def classify_lane_continuity(
    polygons: list[np.ndarray], per_car: dict[int, dict], width: int, height: int, args: argparse.Namespace
) -> tuple[list[LaneDecision], np.ndarray]:
    global CHAIN_VISUALS
    decisions, instance_map = classify_connected_lane_groups(polygons, per_car, width, height, args)
    if not decisions:
        return (decisions, instance_map)
    lane_mask, vehicle_mask = build_binary_masks(polygons, per_car, width, height)
    decisions_by_id = {decision.instance_id: decision for decision in decisions}
    output_by_id = dict(decisions_by_id)
    updated_visuals: list[ChainVisual] = []
    for visual in CHAIN_VISUALS:
        group_decisions = [
            decisions_by_id[instance_id] for instance_id in visual.member_ids if instance_id in decisions_by_id
        ]
        if not group_decisions:
            updated_visuals.append(visual)
            continue
        roi_width = float(np.mean([decision.roi_width for decision in group_decisions]))
        signal: list[int] = []
        for cell in visual.cells:
            signal.extend(
                sample_cell_signal(
                    cell, roi_width, args.chain_roi_cell_length, lane_mask, vehicle_mask, CONTINUITY_CONFIG
                )
            )
        result = analyze_signal(signal, CONTINUITY_CONFIG)
        for decision in group_decisions:
            output_by_id[decision.instance_id] = replace(
                decision,
                lane_type=result.raw_type,
                continuity_occupancy=result.occupancy,
                continuity_gap_count=result.gap_count,
                continuity_longest_gap=result.longest_gap,
                continuity_unknown_ratio=result.unknown_ratio,
            )
        updated_visuals.append(visual)
    CHAIN_VISUALS = updated_visuals
    return (list(output_by_id.values()), instance_map)


def lane_display_type(decision: LaneDecision) -> str:
    if decision.stable_lane_type:
        return decision.stable_lane_type
    return decision.raw_lane_type or decision.lane_type


def draw_lane_analysis_overlay(
    frame: np.ndarray, polygons: list[np.ndarray], decisions: list[LaneDecision], show_rois: bool
) -> np.ndarray:
    if not decisions:
        return frame
    overlay = frame.copy()
    decisions_by_id = {decision.instance_id: decision for decision in decisions}
    for decision in decisions:
        polygon = np.rint(polygons[decision.instance_id - 1]).astype(np.int32)
        color = COLOR_SOLID if lane_display_type(decision) == "SOLID" else COLOR_DOTTED
        cv2.fillPoly(overlay, [polygon], color)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    if not show_rois:
        return frame
    stable_count = sum((bool(track.stable_type) for track in LANE_TRACKS.values()))
    skipped_x_count = sum((decision.temporal_vote_skipped_unknown for decision in decisions))
    cv2.putText(
        frame,
        f"lane_tracks={len(LANE_TRACKS)} stable={stable_count} skipX={skipped_x_count}",
        (24, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for visual in CHAIN_VISUALS:
        group_decisions = [
            decisions_by_id[instance_id] for instance_id in visual.member_ids if instance_id in decisions_by_id
        ]
        if not group_decisions:
            continue
        decision = group_decisions[0]
        display_type = lane_display_type(decision)
        color = COLOR_SOLID if display_type == "SOLID" else COLOR_DOTTED
        for cell in visual.cells:
            cv2.polylines(frame, [cell.polygon], True, color, CHAIN_ROI_LINE_THICKNESS, cv2.LINE_AA)
        anchor_cell = min(visual.cells, key=lambda cell: min((point[1] for point in cell.polygon)))
        anchor = anchor_cell.polygon[np.argmin(anchor_cell.polygon[:, 1])]
        state = decision.stable_lane_type or f"PENDING-{decision.raw_lane_type}"
        if decision.temporal_vote_skipped_unknown:
            state += " SKIP-X"
        label = f"T{decision.lane_track_id} {state} occ={decision.continuity_occupancy:.2f} gap={decision.continuity_gap_count}/{decision.continuity_longest_gap:.0f}px X={decision.continuity_unknown_ratio:.2f} v={decision.vote_samples}"
        cv2.putText(
            frame,
            label,
            (max(0, int(anchor[0])), max(24, int(anchor[1]))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


@dataclass(frozen=True)
class CenterPassConfig:
    radius: int = 9
    min_pixels: int = 3


@dataclass
class CenterPassState:
    crossed: bool = False
    max_pixels: int = 0
    instance_id: int = -1


CENTER_CONFIG = CenterPassConfig()
CENTER_PASS_STATES: dict[int, CenterPassState] = {}
CONTACT_HOLD_SECONDS = 1.5


def solid_center_instance(box: np.ndarray, solid_map: np.ndarray, config: CenterPassConfig) -> tuple[int, int]:
    height, width = solid_map.shape
    x1, y1, x2, y2 = map(float, box)
    center_x = int(round((x1 + x2) * 0.5))
    center_y = int(round((y1 + y2) * 0.5))
    radius = config.radius
    left = max(0, center_x - radius)
    right = min(width, center_x + radius + 1)
    top = max(0, center_y - radius)
    bottom = min(height, center_y + radius + 1)
    if right <= left or bottom <= top:
        return (-1, 0)
    region = solid_map[top:bottom, left:right]
    yy, xx = np.ogrid[top:bottom, left:right]
    circle = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    ids = region[circle & (region > 0)]
    if ids.size == 0:
        return (-1, 0)
    instance_ids, counts = np.unique(ids, return_counts=True)
    best_index = int(np.argmax(counts))
    pixels = int(counts[best_index])
    if pixels < config.min_pixels:
        return (-1, pixels)
    return (int(instance_ids[best_index]), pixels)


class VehicleTracker:
    """Convert YOLO track boxes into the per-car structure without frame flow."""

    def __init__(self, width: int, height: int, car_class: int):
        self.width = width
        self.height = height
        self.car_class = car_class

    def process(self, result) -> tuple[dict[int, dict], dict]:
        boxes = result.boxes
        per_car: dict[int, dict] = {}
        if boxes is None or len(boxes) == 0:
            return (per_car, {"zooming": False, "suspended": False})
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        track_ids = (
            boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(classes), -1, dtype=np.int32)
        )
        frame_area = max(float(self.width * self.height), 1.0)
        for index, class_id in enumerate(classes):
            car_id = int(track_ids[index])
            if class_id != self.car_class or car_id < 0:
                continue
            box = xyxy[index]
            box_width = float(box[2] - box[0])
            box_height = float(box[3] - box[1])
            center_x = float((box[0] + box[2]) * 0.5)
            center_y = float((box[1] + box[3]) * 0.5)
            per_car[car_id] = {
                "box": box,
                "box_w": box_width,
                "box_h": box_height,
                "cx": center_x,
                "cy": center_y,
                "turn": 0.0,
                "head": 0.0,
                "travel": 0.0,
                "moving": False,
                "has_signal": False,
                "parallax_ok": False,
                "area": box_width * box_height / frame_area,
                "stab_delta": None,
            }
        return (per_car, {"zooming": False, "suspended": False})


CENTER_HIT_STREAKS: dict[int, int] = {}
CONTACT_GAP_GRACE_FRAMES = 8
MAX_BRIDGE_THICKNESS = 16


def update_contact_run_with_grace(
    car_id: int, frame_index: int, instance_id: int, contact_pixels: int, contact_now: bool, border_clipped: bool
) -> ContactRun | None:
    """Keep a contact run across a short segmentation-only dropout."""
    previous = CONTACT_RUNS.get(car_id)
    if border_clipped:
        CONTACT_RUNS.pop(car_id, None)
        return None
    if not contact_now:
        if previous is not None and frame_index - previous.last_frame > CONTACT_GAP_GRACE_FRAMES:
            CONTACT_RUNS.pop(car_id, None)
        return None
    frame_gap = 1 if previous is None else frame_index - previous.last_frame
    if previous is None or frame_gap < 1 or frame_gap > CONTACT_GAP_GRACE_FRAMES + 1:
        consecutive_frames = 1
    else:
        consecutive_frames = previous.consecutive_frames + frame_gap
    current = ContactRun(
        last_frame=frame_index,
        consecutive_frames=consecutive_frames,
        instance_id=instance_id,
        contact_pixels=contact_pixels,
    )
    CONTACT_RUNS[car_id] = current
    return current


def facing_fragment_endpoints(
    first_polygon: np.ndarray, second_polygon: np.ndarray, angle_deg: float
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Return the two polygon points that face the open longitudinal gap."""
    angle = math.radians(angle_deg)
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    first = np.asarray(first_polygon, dtype=np.float32)
    second = np.asarray(second_polygon, dtype=np.float32)
    first_projection = first @ direction
    second_projection = second @ direction
    if float(first_projection.mean()) <= float(second_projection.mean()):
        start = first[int(np.argmax(first_projection))]
        end = second[int(np.argmin(second_projection))]
    else:
        start = second[int(np.argmax(second_projection))]
        end = first[int(np.argmin(first_projection))]
    interval = (float(start @ direction), float(end @ direction))
    return (start, end, interval)


def point_to_rect_distance(point: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    """Euclidean distance from a point to an axis-aligned rectangle."""
    x, y, width, height = rect
    px, py = map(float, point)
    dx = max(float(x) - px, 0.0, px - float(x + width))
    dy = max(float(y) - py, 0.0, py - float(y + height))
    return math.hypot(dx, dy)


def stable_solid_fragment_bridges(
    decisions: list[LaneDecision],
    polygons: list[np.ndarray],
    per_car: dict[int, dict],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[OcclusionBridge]:
    """Connect only two visible, temporally confirmed stable-SOLID fragments."""
    stable_solid_decisions = [
        decision
        for decision in decisions
        if decision.stable_lane_type == "SOLID"
        and decision.lane_track_id >= 0
        and decision.temporal_confirmed
        and decision.allow_occlusion_override
        and (0 < decision.instance_id <= len(polygons))
    ]
    cars = [
        (
            car_id,
            math.hypot(float(info["box_w"]), float(info["box_h"])),
            max(1.0, min(float(info["box_w"]), float(info["box_h"]))),
            np.array([info["cx"], info["cy"]], dtype=np.float32),
            expanded_car_rect(info, width, height, args.occlusion_box_margin_ratio),
        )
        for car_id, info in per_car.items()
    ]
    candidates: list[tuple[float, int, tuple[int, int], OcclusionBridge]] = []
    for first, second in itertools.combinations(stable_solid_decisions, 2):
        first_id = first.instance_id
        second_id = second.instance_id
        if axial_angle_difference(first.angle_deg, second.angle_deg) > MAX_INSTANCE_ANGLE_DIFF:
            continue
        angle_deg = axial_mean_deg([first.angle_deg, second.angle_deg])
        first_polygon = polygons[first_id - 1]
        second_polygon = polygons[second_id - 1]
        gap, gap_interval, lateral = fragment_projection_gap(first_polygon, second_polygon, angle_deg)
        if gap <= 0:
            continue
        start, end, endpoint_interval = facing_fragment_endpoints(first_polygon, second_polygon, angle_deg)
        if endpoint_interval[1] <= endpoint_interval[0]:
            continue
        pair_key = (first_id, second_id)
        angle = math.radians(angle_deg)
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        rounded_start = tuple(np.rint(start).astype(int))
        rounded_end = tuple(np.rint(end).astype(int))
        measured_thickness = max(first.thickness, second.thickness)
        thickness = min(
            MAX_BRIDGE_THICKNESS,
            max(
                args.virtual_line_min_thickness,
                int(round(measured_thickness * args.virtual_line_thickness_scale)),
            ),
        )
        gap_midpoint = sum(gap_interval) * 0.5
        for car_id, vehicle_scale, minimum_vehicle_side, car_center, rect in cars:
            if gap > vehicle_scale * args.occlusion_max_gap_car_ratio:
                continue
            if lateral > vehicle_scale * args.occlusion_max_lateral_car_ratio:
                continue
            car_projection = float(car_center @ direction)
            projection_margin = vehicle_scale * 0.15
            if not gap_interval[0] - projection_margin <= car_projection <= gap_interval[1] + projection_margin:
                continue
            intersects, _, _ = cv2.clipLine(rect, rounded_start, rounded_end)
            if not intersects:
                continue
            endpoint_margin = vehicle_scale * 0.35
            if (
                point_to_rect_distance(start, rect) > endpoint_margin
                or point_to_rect_distance(end, rect) > endpoint_margin
            ):
                continue
            if measured_thickness > max(32.0, minimum_vehicle_side * 0.35):
                continue
            bridge = OcclusionBridge(
                instance_id=first_id, car_id=car_id, start=rounded_start, end=rounded_end, thickness=thickness
            )
            center_offset = abs(car_projection - gap_midpoint)
            separate_track_penalty = 0.0 if first.lane_track_id == second.lane_track_id else vehicle_scale * 0.1
            candidates.append((center_offset + lateral + separate_track_penalty, car_id, pair_key, bridge))
    output: list[OcclusionBridge] = []
    bridged_cars: set[int] = set()
    used_pairs: set[tuple[int, int]] = set()
    for _, car_id, pair_key, bridge in sorted(candidates, key=lambda item: item[0]):
        if car_id in bridged_cars or pair_key in used_pairs:
            continue
        output.append(bridge)
        bridged_cars.add(car_id)
        used_pairs.add(pair_key)
    return output


def add_fragment_bridges_to_solid_map(
    solid_map: np.ndarray, bridges: list[OcclusionBridge]
) -> tuple[np.ndarray, np.ndarray]:
    """Use the exact same bridge pixels for crossing detection and visualization."""
    combined = solid_map.copy()
    virtual_mask = np.zeros(solid_map.shape, dtype=np.uint8)
    for bridge in bridges:
        # 서로 다른 픽셀 값으로 LINE_AA를 두 번 그리면 가장자리 반올림 결과가 달라질 수 있다.
        # 브리지 마스크를 한 번만 만든 뒤 표시와 판정에 함께 사용해 픽셀 범위를 일치시킨다.
        bridge_mask = np.zeros(solid_map.shape, dtype=np.uint8)
        cv2.line(bridge_mask, bridge.start, bridge.end, 255, bridge.thickness, cv2.LINE_AA)
        active = bridge_mask > 0
        combined[active] = int(bridge.instance_id)
        virtual_mask[active] = 255
    return (combined, virtual_mask)


def evaluate_solid_crossing_trigger(
    *,
    border_clipped: bool,
    box: np.ndarray,
    solid_map: np.ndarray,
    car_id: int,
    frame_index: int,
    fps: float,
) -> FinalTriggerDecision:
    contact_instance_id, contact_pixels = solid_contact_instance(box, solid_map, CONTACT_CONFIG)
    contact_now = contact_instance_id >= 0
    contact_run = update_contact_run_with_grace(
        car_id, frame_index, contact_instance_id, contact_pixels, contact_now, border_clipped
    )
    center_instance_id, center_pixels = solid_center_instance(box, solid_map, CENTER_CONFIG)
    center_now = center_instance_id >= 0
    if contact_run is None:
        CENTER_PASS_STATES.pop(car_id, None)
        CENTER_HIT_STREAKS.pop(car_id, None)
        center_state = None
    else:
        if contact_run.consecutive_frames == 1:
            CENTER_PASS_STATES[car_id] = CenterPassState()
            CENTER_HIT_STREAKS[car_id] = 0
        center_state = CENTER_PASS_STATES.setdefault(car_id, CenterPassState())
        streak = CENTER_HIT_STREAKS.get(car_id, 0) + 1 if center_now else 0
        CENTER_HIT_STREAKS[car_id] = streak
        if streak >= 2:
            center_state.crossed = True
            if center_pixels >= center_state.max_pixels:
                center_state.max_pixels = center_pixels
                center_state.instance_id = center_instance_id
    center_crossed = bool(center_state is not None and center_state.crossed)
    remembered_center_pixels = 0 if center_state is None else center_state.max_pixels
    hold_frames = 0 if contact_run is None else contact_run.consecutive_frames
    hold_seconds = hold_frames / max(float(fps), 1e-06)
    required_frames = max(1, int(round(CONTACT_HOLD_SECONDS * fps)))
    fired = bool(hold_frames >= required_frames and (center_now or center_crossed) and (not border_clipped))
    if not fired:
        return FinalTriggerDecision(
            solid_contact=contact_now,
            solid_contact_pixels=contact_pixels,
            solid_contact_hold_frames=hold_frames,
            solid_contact_hold_seconds=hold_seconds,
            solid_center_crossed=center_crossed,
            solid_center_pixels=remembered_center_pixels,
        )
    return FinalTriggerDecision(
        fired=True,
        instance_id=center_state.instance_id
        if center_state is not None and center_state.instance_id >= 0
        else center_instance_id,
        reason=f"solid_contact_{CONTACT_HOLD_SECONDS:g}s+center_crossing_confirmed",
        solid_contact=contact_now,
        solid_contact_pixels=contact_pixels,
        solid_contact_hold_frames=hold_frames,
        solid_contact_hold_seconds=hold_seconds,
        solid_center_crossed=True,
        solid_center_pixels=max(remembered_center_pixels, center_pixels),
    )


def parse_and_configure_runtime() -> argparse.Namespace:
    """Parse the single CLI and construct immutable runtime configurations."""
    global CENTER_CONFIG, CONTACT_HOLD_SECONDS, CONTINUITY_CONFIG
    global VOTE_CONFIG, CONTACT_CONFIG, CHAIN_ROI_LINE_THICKNESS
    args = parse_cli_args()
    validate_cli_args(args)
    CHAIN_ROI_LINE_THICKNESS = args.chain_roi_line_thickness
    CONTACT_CONFIG = SolidContactConfig(
        box_width_ratio=args.solid_contact_box_width_ratio,
        box_height_ratio=args.solid_contact_box_height_ratio,
        min_pixels=args.solid_contact_min_pixels,
    )
    CONTACT_HOLD_SECONDS = args.solid_contact_hold_seconds
    CENTER_CONFIG = CenterPassConfig(radius=args.solid_center_radius, min_pixels=args.solid_center_min_pixels)
    VOTE_CONFIG = LaneVoteConfig(
        window_frames=args.lane_vote_window_frames,
        min_samples=args.lane_vote_min_samples,
        lock_ratio=args.lane_vote_lock_ratio,
        switch_ratio=args.lane_vote_switch_ratio,
        max_missed_frames=args.lane_track_max_missed_frames,
        max_angle_diff=args.lane_track_max_angle_diff,
        lateral_width_ratio=args.lane_track_lateral_width_ratio,
        min_lateral_pixels=args.lane_track_min_lateral_pixels,
        max_longitudinal_pixels=args.lane_track_max_longitudinal_pixels,
    )
    CONTINUITY_CONFIG = ContinuityConfig(
        slice_length=args.continuity_slice_length,
        cross_sample_step=args.continuity_cross_sample_step,
        min_lane_samples=args.continuity_min_lane_samples,
        min_lane_ratio=args.continuity_min_lane_ratio,
        vehicle_unknown_ratio=args.continuity_vehicle_unknown_ratio,
        max_unknown_ratio_for_vote=args.continuity_max_unknown_ratio_for_vote,
        close_zero_gap_pixels=args.continuity_close_zero_gap_pixels,
        dotted_min_gap_pixels=args.continuity_dotted_min_gap_pixels,
        dotted_min_gaps=args.continuity_dotted_min_gaps,
    )
    args.center_radius = CENTER_CONFIG.radius
    return args


def reset_runtime_state() -> None:
    """Reset mutable cross-frame state for a clean standalone invocation."""
    global NEXT_LANE_TRACK_ID
    CHAIN_VISUALS.clear()
    CONTACT_RUNS.clear()
    LANE_TRACKS.clear()
    NEXT_LANE_TRACK_ID = 1
    CENTER_PASS_STATES.clear()
    CENTER_HIT_STREAKS.clear()


def main() -> None:
    for column in (
        "solid_contact_now",
        "solid_contact_pixels",
        "trigger_reason",
        "solid_contact_hold_frames",
        "solid_contact_hold_seconds",
        "solid_center_crossed",
        "solid_center_pixels",
        "raw_lane_type",
        "stable_lane_type",
        "lane_track_id",
        "lane_vote_samples",
        "lane_vote_solid_ratio",
        "lane_vote_dotted_ratio",
        "lane_continuity_occupancy",
        "lane_continuity_gap_count",
        "lane_continuity_longest_gap",
        "lane_continuity_unknown_ratio",
    ):
        if column not in CSV_COLUMNS:
            CSV_COLUMNS.append(column)
    reset_runtime_state()
    process_video(parse_and_configure_runtime())


if __name__ == "__main__":
    main()
