from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


HANGUL_EN_TEXT_RE = re.compile(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s.,·:/\\-]")
ENGLISH_ROAD_TERMS = [
    "Ahead",
    "Airport",
    "Apgujeong",
    "Assembly",
    "Bank",
    "Beotigogae",
    "Bldg",
    "Br",
    "Busan",
    "Cemetery",
    "Changdeokgung",
    "Changgyeonggung",
    "Cheonggye",
    "City",
    "Council",
    "Daebang",
    "Euljiro",
    "Expressway",
    "Expwy",
    "Gangbyeon",
    "Gimpo",
    "Hall",
    "Hangang",
    "Hangang-daero",
    "Hangangdaegyo",
    "Hannamdaegyo",
    "Heunginjimun",
    "High",
    "Hanyang",
    "Incheon",
    "Itaewon",
    "Jongno",
    "Junction",
    "Korea",
    "Myeongdong",
    "Namsan",
    "Natl",
    "Olympic",
    "Oksu",
    "Park",
    "Rotary",
    "Sch",
    "Seodaemun",
    "Seongsandaegyo",
    "Seoul",
    "Seoulgyo",
    "Sinchon",
    "Sinsa",
    "Station",
    "Stadium",
    "Stn",
    "Sookmyung",
    "Sungnyemun",
    "Tunnel",
    "Toegye-ro",
    "Univ",
    "Wonhyodaegyo",
    "World",
    "Womens",
    "Yaksu",
    "Yanghwa",
    "Yangjae",
    "Yeomchang",
    "Yeomcheongyo",
    "Yeongdeungpo",
    "Yeouidaebang-ro",
    "Yeouidong-ro",
    "Yeouigyo",
    "Yongsan",
]
KOREAN_ROAD_TERMS = [
    "강변북로",
    "경부고속도로",
    "고속도로",
    "공항",
    "교차로",
    "금호사거리",
    "남산",
    "남산공원",
    "남산터널",
    "대교",
    "대로",
    "로터리",
    "버스전용차로",
    "사거리",
    "서울역",
    "성산대교",
    "시청",
    "시의회",
    "신사역",
    "안전속도",
    "양재역",
    "역",
    "올림픽대로",
    "을지로",
    "종로",
    "차로",
    "청계",
    "터널",
    "한강대교",
    "한남대교",
    "흥인지문",
]
ROAD_TEXT_REPLACEMENTS = [
    (re.compile(r"\bNamsar\s*Park\b", re.IGNORECASE), "Namsan Park"),
    (re.compile(r"\bNamsarPark\b", re.IGNORECASE), "Namsan Park"),
    (re.compile(r"\bNamsanark\b", re.IGNORECASE), "Namsan Park"),
    (re.compile(r"\bNamsan\s+ark\b", re.IGNORECASE), "Namsan Park"),
    (re.compile(r"\bHannandaegyo\b", re.IGNORECASE), "Hannamdaegyo"),
    (re.compile(r"\bHannane\b", re.IGNORECASE), "Hannam"),
    (re.compile(r"\bHangangdar\b", re.IGNORECASE), "Hangangdaegyo"),
    (re.compile(r"\bSeongsandaegyo\s*Bi\b", re.IGNORECASE), "Seongsandaegyo Br"),
    (re.compile(r"\bSeongsandaegyo\s*B\b", re.IGNORECASE), "Seongsandaegyo Br"),
    (re.compile(r"\bAlrport\b", re.IGNORECASE), "Airport"),
    (re.compile(r"\bArport\b", re.IGNORECASE), "Airport"),
    (re.compile(r"\bArporl\b", re.IGNORECASE), "Airport"),
    (re.compile(r"\bWord\s+Cup\b", re.IGNORECASE), "World Cup"),
    (re.compile(r"\bWordd\s+Cup\b", re.IGNORECASE), "World Cup"),
    (re.compile(r"\bStadum\b", re.IGNORECASE), "Stadium"),
    (re.compile(r"\bCouncll\b", re.IGNORECASE), "Council"),
    (re.compile(r"\bCounol\b", re.IGNORECASE), "Council"),
    (re.compile(r"\bCily\b", re.IGNORECASE), "City"),
    (re.compile(r"\bChy\b", re.IGNORECASE), "City"),
    (re.compile(r"\bHal\b", re.IGNORECASE), "Hall"),
    (re.compile(r"\bHab\b", re.IGNORECASE), "Hall"),
    (re.compile(r"\bltaewon\b"), "Itaewon"),
    (re.compile(r"\bSooul\b", re.IGNORECASE), "Seoul"),
    (re.compile(r"\bSooul\s+sin\b", re.IGNORECASE), "Seoul Stn"),
    (re.compile(r"\bSeoulstn\b", re.IGNORECASE), "Seoul Stn"),
    (re.compile(r"\bGyeongou\b", re.IGNORECASE), "Gyeongbu"),
    (re.compile(r"\bGyeongu\b", re.IGNORECASE), "Gyeongbu"),
    (re.compile(r"\bDANGEI\b", re.IGNORECASE), "DANGER"),
    (re.compile(r"\bDANAEJ\b", re.IGNORECASE), "DANGER"),
    (re.compile(r"\bRolary\b", re.IGNORECASE), "Rotary"),
    (re.compile(r"\bNotin\b", re.IGNORECASE), "North"),
    (re.compile(r"\bBn\b", re.IGNORECASE), "Br"),
    (re.compile(r"\bBrM\b", re.IGNORECASE), "Br"),
    (re.compile(r"(\d)치로"), r"\1차로"),
    (re.compile(r"(\d)차모"), r"\1차로"),
    (re.compile(r"울지로|올지로"), "을지로"),
    (re.compile(r"증로"), "종로"),
    (re.compile(r"승례문|송례문"), "숭례문"),
    (re.compile(r"청겨|성계|원계"), "청계"),
    (re.compile(r"남태문로"), "남대문로"),
    (re.compile(r"신청·신의회"), "시청·시의회"),
]


@dataclass
class Detection:
    frame_index: int
    bbox: list[float]
    confidence: float


@dataclass
class CropSample:
    frame_index: int
    bbox: list[float]
    detection_confidence: float
    sharpness: float
    area_ratio: float
    quality_score: float
    image: np.ndarray
    crop_path: Path | None = None


@dataclass
class Track:
    track_id: int
    first_frame: int
    last_frame: int
    last_bbox: list[float]
    detections: list[Detection] = field(default_factory=list)
    samples: list[CropSample] = field(default_factory=list)
    missed: int = 0
    hits: int = 0

    def update(self, detection: Detection) -> None:
        self.last_frame = detection.frame_index
        self.last_bbox = detection.bbox
        self.detections.append(detection)
        self.hits += 1
        self.missed = 0


class SimpleTracker:
    def __init__(
        self,
        max_age: int = 25,
        match_threshold: float = 0.22,
        max_center_distance: float = 2.8,
    ) -> None:
        self.max_age = max_age
        self.match_threshold = match_threshold
        self.max_center_distance = max_center_distance
        self.next_track_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], frame_index: int) -> list[tuple[Track, Detection]]:
        active_tracks = [
            track
            for track in self.tracks.values()
            if frame_index - track.last_frame <= self.max_age
        ]

        pairs: list[tuple[float, int, int]] = []
        for track in active_tracks:
            for det_index, detection in enumerate(detections):
                score = match_score(track.last_bbox, detection.bbox, self.max_center_distance)
                if score >= self.match_threshold:
                    pairs.append((score, track.track_id, det_index))
        pairs.sort(reverse=True, key=lambda item: item[0])

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: list[tuple[Track, Detection]] = []
        for _, track_id, det_index in pairs:
            if track_id in matched_tracks or det_index in matched_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[det_index]
            track.update(detection)
            matched_tracks.add(track_id)
            matched_detections.add(det_index)
            assignments.append((track, detection))

        for det_index, detection in enumerate(detections):
            if det_index in matched_detections:
                continue
            track = Track(
                track_id=self.next_track_id,
                first_frame=frame_index,
                last_frame=frame_index,
                last_bbox=detection.bbox,
            )
            track.update(detection)
            self.tracks[track.track_id] = track
            self.next_track_id += 1
            assignments.append((track, detection))

        for track in self.tracks.values():
            if track.track_id not in matched_tracks and track.last_frame != frame_index:
                track.missed = frame_index - track.last_frame

        return assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect, track, OCR, and aggregate traffic signs in an MP4.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weight", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/video_ocr"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence. Default reads outputs/models/recommended_conf.txt, then falls back to 0.30.",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-age", type=int, default=25)
    parser.add_argument("--min-track-frames", type=int, default=2)
    parser.add_argument("--max-crops-per-track", type=int, default=8)
    parser.add_argument("--crop-padding", type=float, default=0.18)
    parser.add_argument("--ocr-lang", default="korean")
    parser.add_argument(
        "--ocr-variants",
        default="original,enhanced,gray",
        help="Comma-separated OCR image variants: original, enhanced, gray.",
    )
    parser.add_argument("--group-similarity", type=float, default=0.58)
    parser.add_argument("--save-empty-tracks", action="store_true")
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Run detection/tracking only. Useful for fast YOLO quality checks.",
    )
    parser.add_argument(
        "--draw-stale-tracks",
        action="store_true",
        help="Also draw tracks that were not detected on the current frame. Disabled by default to avoid laggy overlays.",
    )
    return parser.parse_args()


def resolve_weight(weight: Path | None) -> Path:
    if weight is not None:
        if not weight.exists():
            raise FileNotFoundError(f"Weight not found: {weight}")
        return weight

    pointer = Path("outputs/models/best_weight.txt")
    if not pointer.exists():
        raise FileNotFoundError("No --weight provided and outputs/models/best_weight.txt does not exist.")
    resolved = Path(pointer.read_text(encoding="utf-8").strip())
    if not resolved.exists():
        raise FileNotFoundError(f"Weight listed in {pointer} does not exist: {resolved}")
    return resolved


def resolve_conf(conf: float | None) -> float:
    if conf is not None:
        return conf
    pointer = Path("outputs/models/recommended_conf.txt")
    if pointer.exists():
        try:
            value = float(pointer.read_text(encoding="utf-8").strip())
            if 0.0 < value < 1.0:
                return value
        except ValueError:
            pass
    return 0.8


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def match_score(a: list[float], b: list[float], max_center_distance: float) -> float:
    iou = bbox_iou(a, b)
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    distance = math.hypot(ax - bx, ay - by)
    avg_diag = (math.sqrt(max(bbox_area(a), 1.0)) + math.sqrt(max(bbox_area(b), 1.0))) * 0.5
    center_norm = distance / max(avg_diag, 1.0)
    center_score = max(0.0, 1.0 - (center_norm / max_center_distance))
    size_ratio = min(bbox_area(a), bbox_area(b)) / max(bbox_area(a), bbox_area(b), 1.0)
    return (0.65 * iou) + (0.25 * center_score) + (0.10 * size_ratio)


def read_frame_detections(model: YOLO, frame: np.ndarray, frame_index: int, args: argparse.Namespace) -> list[Detection]:
    predict_kwargs: dict[str, Any] = {
        "source": frame,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "classes": [0],
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    result = model.predict(**predict_kwargs)[0]
    detections: list[Detection] = []
    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    height, width = frame.shape[:2]
    for bbox, conf in zip(boxes, confs):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width - 1), x2))
        y2 = max(0.0, min(float(height - 1), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(frame_index=frame_index, bbox=[x1, y1, x2, y2], confidence=float(conf)))
    return detections


def crop_with_padding(frame: np.ndarray, bbox: list[float], padding_ratio: float) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    ix1 = max(0, int(math.floor(x1 - pad_x)))
    iy1 = max(0, int(math.floor(y1 - pad_y)))
    ix2 = min(width, int(math.ceil(x2 + pad_x)))
    iy2 = min(height, int(math.ceil(y2 + pad_y)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    crop = frame[iy1:iy2, ix1:ix2].copy()
    if crop.size == 0:
        return None
    return crop


def laplacian_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_score(confidence: float, sharpness: float, area_ratio: float) -> float:
    sharpness_score = min(1.0, math.log1p(max(sharpness, 0.0)) / 8.0)
    area_score = min(1.0, math.sqrt(max(area_ratio, 0.0)) * 8.0)
    return (0.35 * confidence) + (0.35 * sharpness_score) + (0.30 * area_score)


def maybe_add_sample(track: Track, detection: Detection, frame: np.ndarray, args: argparse.Namespace) -> None:
    crop = crop_with_padding(frame, detection.bbox, args.crop_padding)
    if crop is None:
        return
    crop_h, crop_w = crop.shape[:2]
    if crop_w < 12 or crop_h < 12:
        return

    frame_area = frame.shape[0] * frame.shape[1]
    area_ratio = bbox_area(detection.bbox) / max(frame_area, 1)
    sharpness = laplacian_sharpness(crop)
    score = quality_score(detection.confidence, sharpness, area_ratio)
    sample = CropSample(
        frame_index=detection.frame_index,
        bbox=detection.bbox,
        detection_confidence=detection.confidence,
        sharpness=sharpness,
        area_ratio=area_ratio,
        quality_score=score,
        image=crop,
    )

    track.samples.append(sample)
    track.samples.sort(key=lambda item: item.quality_score, reverse=True)
    del track.samples[args.max_crops_per_track :]


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def upscale_for_ocr(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest >= 640:
        return image
    scale = min(4.0, max(1.0, 640 / max(longest, 1)))
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return resized


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    base = upscale_for_ocr(image)
    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)


def gray_for_ocr(image: np.ndarray) -> np.ndarray:
    base = upscale_for_ocr(image)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def ocr_image_variants(image: np.ndarray, variant_names: str) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    requested = [name.strip().lower() for name in variant_names.split(",") if name.strip()]
    if not requested:
        requested = ["original"]

    seen: set[str] = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        if name == "original":
            variants.append(("original", upscale_for_ocr(image)))
        elif name == "enhanced":
            variants.append(("enhanced", enhance_for_ocr(image)))
        elif name == "gray":
            variants.append(("gray", gray_for_ocr(image)))
        else:
            raise ValueError(f"Unsupported OCR variant: {name}")
    return variants


def build_ocr(lang: str) -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    constructor_attempts = [
        {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {"use_angle_cls": True, "lang": lang, "show_log": False},
        {"use_angle_cls": True, "lang": lang},
        {"lang": lang},
    ]
    last_error: Exception | None = None
    for kwargs in constructor_attempts:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:  # PaddleOCR changed constructor args across versions.
            last_error = exc
    raise RuntimeError(f"Failed to initialize PaddleOCR with lang={lang!r}: {last_error}")


def normalize_text(text: str) -> str:
    text = HANGUL_EN_TEXT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_text(text: str) -> str:
    return re.sub(r"[\s.,·:/\\-]+", "", normalize_text(text)).lower()


def fuzzy_correct_english_token(match: re.Match[str]) -> str:
    token = match.group(0)
    if len(token) < 4 or any(char.isdigit() for char in token):
        return token

    token_key = token.lower().replace(".", "")
    has_hyphen = "-" in token_key
    best_term = token
    best_ratio = 0.0
    for term in ENGLISH_ROAD_TERMS:
        if has_hyphen != ("-" in term):
            continue
        term_key = term.lower().replace(".", "")
        ratio = SequenceMatcher(None, token_key, term_key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_term = term

    threshold = 0.88 if len(token) <= 6 else 0.84
    if has_hyphen:
        threshold = 0.80
    return best_term if best_ratio >= threshold else token


def postprocess_road_text(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    for pattern, replacement in ROAD_TEXT_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    text = re.sub(r"\b([A-Za-z]+)daegyo\s+B\b", r"\1daegyo Br", text)
    text = re.sub(r"\b([A-Za-z]+)\s+Stn\b", r"\1 Stn", text)
    text = re.sub(r"([가-힣])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])([가-힣])", r"\1 \2", text)
    text = re.sub(r"(\d)\s*-\s*ga\b", r"\1-ga", text)
    text = re.sub(r"\b([A-Za-z][A-Za-z.-]*)\b", fuzzy_correct_english_token, text)
    text = re.sub(r"\s+([,·/])", r"\1", text)
    text = re.sub(r"([,·/])(?=\S)", r"\1 ", text)
    text = re.sub(r"(\d{1,2}):\s+(\d{2})", r"\1:\2", text)
    text = re.sub(r"(\d),\s+(\d)", r"\1,\2", text)
    text = re.sub(r"\bCity\s+City\s+Council\b", "City Council", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def domain_keyword_score(text: str) -> float:
    compact = compact_text(text)
    if not compact:
        return 0.0

    hits = 0
    for term in KOREAN_ROAD_TERMS:
        if term in text:
            hits += 1
    lowered = text.lower()
    for term in ENGLISH_ROAD_TERMS:
        if term.lower() in lowered:
            hits += 1
    return min(1.0, hits / 5.0)


def road_text_quality(text: str) -> float:
    compact = compact_text(text)
    if not compact:
        return 0.0
    length_score = min(1.0, len(compact) / 28.0)
    mixed_script = 1.0 if re.search(r"[가-힣]", text) and re.search(r"[A-Za-z]", text) else 0.65
    return (
        (0.34 * char_quality(text))
        + (0.30 * domain_keyword_score(text))
        + (0.22 * length_score)
        + (0.14 * mixed_script)
    )


def length_penalty(text: str) -> float:
    compact = compact_text(text)
    if not compact:
        return 0.0
    if len(compact) == 1:
        return 0.35
    if len(compact) == 2:
        return 0.7
    if compact.isdigit() and len(compact) <= 2:
        return 0.75
    return 1.0


def char_quality(text: str) -> float:
    compact = compact_text(text)
    if not compact:
        return 0.0
    useful = sum(1 for char in compact if char.isalnum() or ("가" <= char <= "힣"))
    useful_ratio = useful / len(compact)
    length_score = min(1.0, len(compact) / 12.0)
    return (0.65 * useful_ratio) + (0.35 * length_score)


def text_quality_penalty(text: str) -> float:
    compact = compact_text(text)
    if not compact:
        return 0.0
    penalty = length_penalty(text)
    if len(compact) >= 4:
        unique_ratio = len(set(compact)) / len(compact)
        if unique_ratio < 0.35:
            penalty *= 0.75
    return penalty * max(0.25, road_text_quality(text))


def merge_ocr_lines(lines: list[tuple[str, float]]) -> tuple[str, float]:
    normalized = [(normalize_text(text), float(conf)) for text, conf in lines]
    normalized = [(text, conf) for text, conf in normalized if text]
    if not normalized:
        return "", 0.0

    merged: list[tuple[str, float]] = []
    for text, conf in normalized:
        key = compact_text(text)
        if not key:
            continue
        if any(SequenceMatcher(None, key, compact_text(existing)).ratio() >= 0.92 for existing, _ in merged):
            continue
        merged.append((text, conf))

    if not merged:
        return "", 0.0
    text = " ".join(item[0] for item in merged)
    confidence = sum(item[1] for item in merged) / len(merged)
    return text, confidence


def parse_paddle_lines(result: Any) -> list[tuple[str, float]]:
    if result is None:
        return []

    if isinstance(result, dict):
        texts = result.get("rec_texts") or result.get("texts") or []
        scores = result.get("rec_scores") or result.get("scores") or []
        return [
            (str(text), float(score))
            for text, score in zip(texts, scores)
            if str(text).strip()
        ]

    if hasattr(result, "json"):
        try:
            return parse_paddle_lines(result.json)
        except Exception:
            pass

    if isinstance(result, (tuple, list)):
        if len(result) == 2 and isinstance(result[0], str):
            return [(str(result[0]), float(result[1]))]

        lines: list[tuple[str, float]] = []
        if (
            len(result) == 1
            and isinstance(result[0], list)
            and (not result[0] or not (len(result[0]) == 2 and isinstance(result[0][0], str)))
        ):
            result = result[0]

        for item in result:
            if item is None:
                continue
            if isinstance(item, dict):
                lines.extend(parse_paddle_lines(item))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                second = item[1]
                if isinstance(second, (tuple, list)) and len(second) >= 2 and isinstance(second[0], str):
                    lines.append((str(second[0]), float(second[1])))
                else:
                    lines.extend(parse_paddle_lines(item))
        return lines

    return []


def run_ocr(ocr: Any, image: np.ndarray) -> list[tuple[str, float]]:
    attempts = []
    if hasattr(ocr, "predict"):
        attempts.append(lambda: ocr.predict(image))
    if hasattr(ocr, "ocr"):
        attempts.extend(
            [
                lambda: ocr.ocr(image, cls=True),
                lambda: ocr.ocr(image),
            ]
        )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return parse_paddle_lines(attempt())
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"PaddleOCR inference failed: {last_error}")


def candidate_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_key = compact_text(a["text"])
    b_key = compact_text(b["text"])
    if not a_key or not b_key:
        return 0.0
    compact_similarity = SequenceMatcher(None, a_key, b_key).ratio()
    token_similarity = SequenceMatcher(
        None,
        normalize_text(a["text"]).lower(),
        normalize_text(b["text"]).lower(),
    ).ratio()
    return (0.75 * compact_similarity) + (0.25 * token_similarity)


def aggregate_ocr(candidates: list[dict[str, Any]], similarity_threshold: float) -> tuple[str, float, list[dict[str, Any]]]:
    valid = [candidate for candidate in candidates if compact_text(candidate["text"])]
    if not valid:
        return "", 0.0, candidates

    valid.sort(key=lambda item: item["weighted_score"], reverse=True)
    groups: list[dict[str, Any]] = []
    for candidate in valid:
        best_group: dict[str, Any] | None = None
        best_similarity = 0.0
        for group in groups:
            similarity = max(candidate_similarity(candidate, item) for item in group["items"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_group = group
        if best_group is not None and best_similarity >= similarity_threshold:
            best_group["items"].append(candidate)
        else:
            groups.append({"items": [candidate]})

    def group_score(group: dict[str, Any]) -> float:
        items = group["items"]
        avg = sum(item["weighted_score"] for item in items) / len(items)
        support = min(1.0, sum(item["weighted_score"] for item in items) / 4.0)
        frame_count = len({item.get("frame") for item in items})
        repeat = min(1.0, math.log1p(frame_count) / math.log(6.0))
        quality = sum(road_text_quality(item["text"]) for item in items) / len(items)
        return (0.36 * avg) + (0.22 * support) + (0.20 * repeat) + (0.22 * quality)

    def representative_score(candidate: dict[str, Any], group: dict[str, Any]) -> float:
        peers = [item for item in group["items"] if item is not candidate]
        consistency = (
            sum(candidate_similarity(candidate, peer) for peer in peers) / len(peers)
            if peers
            else 0.65
        )
        return (
            (0.55 * candidate["weighted_score"])
            + (0.30 * consistency)
            + (0.15 * road_text_quality(candidate["text"]))
        )

    best_group = max(groups, key=group_score)
    score = min(1.0, group_score(best_group))
    best_item = max(best_group["items"], key=lambda item: representative_score(item, best_group))
    return best_item["text"], score, candidates


def ocr_track(track: Track, ocr: Any, crop_root: Path, args: argparse.Namespace) -> tuple[str, float, list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    crop_paths: list[str] = []

    for rank, sample in enumerate(sorted(track.samples, key=lambda item: item.quality_score, reverse=True), start=1):
        crop_path = crop_root / f"track_{track.track_id:04d}" / f"rank_{rank:02d}_frame_{sample.frame_index:06d}.jpg"
        imwrite_unicode(crop_path, sample.image)
        sample.crop_path = crop_path
        crop_paths.append(crop_path.resolve().as_posix())

        sample_candidates: list[dict[str, Any]] = []
        for variant_name, variant_image in ocr_image_variants(sample.image, args.ocr_variants):
            lines = run_ocr(ocr, variant_image)
            text, ocr_conf = merge_ocr_lines(lines)
            text = postprocess_road_text(text)
            if not text:
                continue

            bbox_size_score = min(1.0, math.sqrt(max(sample.area_ratio, 0.0)) * 8.0)
            variant_bonus = 0.03 if variant_name == "enhanced" else 0.0
            score = (
                (0.42 * ocr_conf)
                + (0.23 * sample.quality_score)
                + (0.15 * sample.detection_confidence)
                + (0.12 * bbox_size_score)
                + (0.08 * road_text_quality(text))
                + variant_bonus
            ) * text_quality_penalty(text)
            sample_candidates.append(
                {
                    "text": text,
                    "ocr_confidence": round(float(ocr_conf), 6),
                    "weighted_score": round(float(min(1.0, score)), 6),
                    "frame": sample.frame_index,
                    "variant": variant_name,
                    "crop_path": crop_path.resolve().as_posix(),
                    "crop_quality": round(sample.quality_score, 6),
                    "sharpness": round(sample.sharpness, 3),
                    "bbox": round_bbox(sample.bbox),
                    "detection_confidence": round(sample.detection_confidence, 6),
                }
            )

        if not sample_candidates:
            candidates.append(
                {
                    "text": "",
                    "ocr_confidence": 0.0,
                    "weighted_score": 0.0,
                    "frame": sample.frame_index,
                    "variant": "",
                    "crop_path": crop_path.resolve().as_posix(),
                    "crop_quality": round(sample.quality_score, 6),
                    "sharpness": round(sample.sharpness, 3),
                    "bbox": round_bbox(sample.bbox),
                    "detection_confidence": round(sample.detection_confidence, 6),
                }
            )
            continue

        candidates.extend(sample_candidates)

    best_text, best_score, candidates = aggregate_ocr(candidates, args.group_similarity)
    return best_text, best_score, candidates, crop_paths


def round_bbox(bbox: list[float]) -> list[float]:
    return [round(float(value), 2) for value in bbox]


def track_to_json(
    track: Track,
    ocr: Any | None,
    crop_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if ocr is None:
        best_text, best_score, candidates, crop_paths = "", 0.0, [], []
    else:
        best_text, best_score, candidates, crop_paths = ocr_track(track, ocr, crop_root, args)
    confidences = [detection.confidence for detection in track.detections]
    return {
        "track_id": track.track_id,
        "best_ocr_text": best_text,
        "ocr_score": round(float(best_score), 6),
        "first_frame": track.first_frame,
        "last_frame": track.last_frame,
        "frame_count": len(track.detections),
        "bbox_history": [
            {
                "frame": detection.frame_index,
                "bbox": round_bbox(detection.bbox),
                "confidence": round(float(detection.confidence), 6),
            }
            for detection in track.detections
        ],
        "detection_confidence": {
            "mean": round(float(sum(confidences) / max(len(confidences), 1)), 6),
            "max": round(float(max(confidences) if confidences else 0.0), 6),
        },
        "ocr_candidates": candidates,
        "crop_paths": crop_paths,
    }


def should_export_track(track: Track, args: argparse.Namespace) -> bool:
    if args.save_empty_tracks:
        return True
    if len(track.detections) >= args.min_track_frames:
        return True
    return any(detection.confidence >= 0.5 for detection in track.detections)


def open_video(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if capture.isOpened():
        return capture
    capture.release()
    capture = cv2.VideoCapture(video_path.resolve().as_posix())
    if capture.isOpened():
        return capture
    raise FileNotFoundError(f"Failed to open video: {video_path}")


def color_for_track(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 9973)
    color = rng.integers(80, 256, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    frame_index: int,
    max_age: int,
    draw_stale_tracks: bool,
) -> np.ndarray:
    annotated = frame.copy()
    for track in tracks:
        age = frame_index - track.last_frame
        if age < 0 or age > max_age:
            continue
        if age != 0 and not draw_stale_tracks:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in track.last_bbox]
        color = color_for_track(track.track_id)
        thickness = 3 if age == 0 else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        confidence = track.detections[-1].confidence if track.detections else 0.0
        label = f"ID {track.track_id}  {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y1 = max(0, y1 - label_size[1] - baseline - 6)
        label_y2 = min(annotated.shape[0] - 1, label_y1 + label_size[1] + baseline + 6)
        label_x2 = min(annotated.shape[1] - 1, x1 + label_size[0] + 8)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        f"frame {frame_index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def create_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int], codec: str = "mp4v") -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    return writer


def write_clean_json(result: dict[str, Any], clean_path: Path, max_candidates: int = 5) -> None:
    group_similarity = result.get("settings", {}).get("group_similarity", 0.58)

    def _clean_candidate(c: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": postprocess_road_text(c.get("text", "")),
            "score": c.get("weighted_score", 0.0),
            "ocr_confidence": c.get("ocr_confidence", 0.0),
            "frame": c.get("frame"),
            "variant": c.get("variant", ""),
        }

    tracks = []
    for t in result.get("tracks", []):
        candidates = [
            _clean_candidate(c)
            for c in t.get("ocr_candidates", [])
            if c.get("text")
        ]
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        if candidates:
            vote = [{"text": c["text"], "weighted_score": c["score"], "frame": c["frame"]} for c in candidates]
            best_text, best_score, _ = aggregate_ocr(vote, group_similarity)
        else:
            best_text = postprocess_road_text(t.get("best_ocr_text", ""))
            best_score = t.get("ocr_score", 0.0)

        if not best_text:
            continue

        tracks.append({
            "track_id": t.get("track_id"),
            "text": best_text,
            "ocr_score": round(float(best_score), 6),
            "first_frame": t.get("first_frame"),
            "last_frame": t.get("last_frame"),
            "frame_count": t.get("frame_count"),
            "detection_confidence_mean": t.get("detection_confidence", {}).get("mean", 0.0),
            "candidates": candidates[:max_candidates],
        })

    clean = {
        "video_path": result.get("video_path"),
        "weight_path": result.get("weight_path"),
        "created_at": result.get("created_at"),
        "track_count": len(tracks),
        "tracks": tracks,
    }
    clean_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    weight = resolve_weight(args.weight)
    args.conf = resolve_conf(args.conf)
    output_json = args.output_json or (args.output_dir / f"{args.video.stem}_results.json")
    output_video = args.output_video
    crop_root = args.output_dir / "crops" / args.video.stem
    output_json.parent.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weight))
    ocr = None if args.skip_ocr else build_ocr(args.ocr_lang)
    tracker = SimpleTracker(max_age=args.max_age)
    capture = open_video(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = (
        create_video_writer(output_video, fps / args.frame_stride, (width, height))
        if output_video is not None
        else None
    )

    frame_index = -1
    processed_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % args.frame_stride != 0:
                continue
            if args.max_frames is not None and processed_frames >= args.max_frames:
                break

            detections = read_frame_detections(model, frame, frame_index, args)
            assignments = tracker.update(detections, frame_index)
            for track, detection in assignments:
                maybe_add_sample(track, detection, frame, args)

            if writer is not None:
                active_tracks = sorted(tracker.tracks.values(), key=lambda item: item.track_id)
                writer.write(
                    draw_tracks(
                        frame,
                        active_tracks,
                        frame_index,
                        args.max_age,
                        args.draw_stale_tracks,
                    )
                )

            processed_frames += 1
            if processed_frames % 100 == 0:
                print(f"processed_frames={processed_frames} frame_index={frame_index} tracks={len(tracker.tracks)}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    export_tracks = [
        track
        for track in sorted(tracker.tracks.values(), key=lambda item: item.track_id)
        if should_export_track(track, args)
    ]
    tracks_json = [track_to_json(track, ocr, crop_root, args) for track in export_tracks]

    result = {
        "video_path": args.video.resolve().as_posix(),
        "weight_path": weight.resolve().as_posix(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "frame_stride": args.frame_stride,
            "max_age": args.max_age,
            "min_track_frames": args.min_track_frames,
            "max_crops_per_track": args.max_crops_per_track,
            "ocr_lang": args.ocr_lang,
            "ocr_variants": args.ocr_variants,
            "group_similarity": args.group_similarity,
            "skip_ocr": args.skip_ocr,
            "output_video": output_video.resolve().as_posix() if output_video else None,
            "draw_stale_tracks": args.draw_stale_tracks,
        },
        "frames_processed": processed_frames,
        "tracks": tracks_json,
    }

    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    clean_json = output_json.with_name(output_json.stem + "_clean" + output_json.suffix)
    write_clean_json(result, clean_json)

    print(f"saved_json={output_json.resolve().as_posix()}")
    print(f"saved_clean_json={clean_json.resolve().as_posix()}")
    if output_video is not None:
        print(f"saved_video={output_video.resolve().as_posix()}")
    print(f"tracks={len(tracks_json)}")


if __name__ == "__main__":
    main()
