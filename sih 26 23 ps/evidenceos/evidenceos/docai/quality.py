"""Quality analyzer (PRD §11.3.1) — decides preprocessing per page.

Metrics are cheap OpenCV/numpy heuristics: blur (variance of Laplacian),
noise (median-filter residual), contrast, skew (Hough on text edges), resolution
(effective DPI) and a handwriting probability heuristic. Clean digital PDFs are
never preprocessed (rule from the PRD).
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def analyze(gray: np.ndarray, dpi: int) -> dict:
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)

    lap_var = float(cv2.Laplacian(small, cv2.CV_64F).var())
    blur = float(np.clip(1.0 - math.log10(lap_var + 1.0) / 4.0, 0.0, 1.0))        # 0 sharp .. 1 blurry

    med = cv2.medianBlur(small, 3)
    noise = float(np.clip(np.mean(np.abs(small.astype(np.int16) - med.astype(np.int16))) / 25.0, 0.0, 1.0))

    contrast = float(np.clip(small.std() / 80.0, 0.0, 1.0))

    skew = estimate_skew(small)

    # ink statistics -> handwriting heuristic (irregular, connected strokes with high height variance)
    _, binv = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = float(binv.mean() / 255.0)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    hand = 0.0
    if n > 20:
        hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
        ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
        keep = (hs > 3) & (hs < small.shape[0] * 0.2)
        if keep.sum() > 20:
            hs, ws = hs[keep], ws[keep]
            cv_h = hs.std() / (hs.mean() + 1e-6)                # printed text: uniform heights
            aspect = float(np.mean(ws / (hs + 1e-6)))           # cursive strokes are wide
            hand = float(np.clip(0.5 * cv_h + 0.15 * max(0.0, aspect - 1.5), 0.0, 1.0))

    return {
        "blur": round(blur, 3),
        "noise": round(noise, 3),
        "contrast": round(contrast, 3),
        "skew": round(skew, 2),
        "resolution": int(dpi),
        "ink_ratio": round(ink, 4),
        "handwriting_probability": round(hand, 3),
        "actions": recommend(blur, noise, contrast, skew, dpi),
    }


def recommend(blur, noise, contrast, skew, dpi) -> list[str]:
    acts = []
    if blur > 0.6:
        acts.append("upscale")
    if noise > 0.5:
        acts.append("denoise")
    if contrast < 0.3:
        acts.append("enhance_contrast")
    if abs(skew) > 1.0:
        acts.append("deskew")
    if dpi < 200:
        acts.append("ocr_at_higher_dpi")
    return acts


def estimate_skew(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=120, minLineLength=gray.shape[1] // 5, maxLineGap=8)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(ang) < 20:                       # near-horizontal text/table lines
            angles.append(ang)
    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def preprocess(gray: np.ndarray, q: dict) -> np.ndarray:
    """Apply only the recommended actions; return possibly-corrected image."""
    img = gray
    acts = q.get("actions", [])
    if "deskew" in acts:
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), q["skew"], 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if "denoise" in acts:
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    if "enhance_contrast" in acts:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    return img
