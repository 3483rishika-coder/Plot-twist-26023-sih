"""OCR pipeline (PRD §11.3.3) — RapidOCR (PaddleOCR models on ONNX runtime, CPU friendly).

Returns line-level results with boxes + confidences.  A domain vocabulary pass
corrects frequent OCR confusions in coal-sector terms.  Handwriting is *not*
auto-recognized: regions with a high handwriting probability are flagged for
human review with the crop kept as evidence (PRD §11.3.4).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np

_engine = None
_lock = threading.Lock()

DOMAIN_VOCAB = [
    "Coal India Limited", "Eastern Coalfields", "Bharat Coking Coal", "Central Coalfields", "Northern Coalfields",
    "Western Coalfields", "South Eastern Coalfields", "Mahanadi Coalfields", "North Eastern Coalfields", "Singareni",
    "CMPDI", "CIL", "ECL", "BCCL", "CCL", "NCL", "WCL", "SECL", "MCL", "NEC", "SCCL", "NLCIL",
    "Production", "Despatch", "Dispatch", "Offtake", "Overburden", "Coking", "Non-coking", "Lignite", "Washery",
    "Opencast", "Underground", "Coalfield", "Raniganj", "Jharia", "Talcher", "Korba", "Singrauli", "Rajmahal",
    "Million Tonnes", "Lakh Tonnes", "Tonnes", "Directory", "Ministry of Coal", "Government of India",
]
_FIXES = [
    (re.compile(r"\bDespa[tl]ch\b"), "Despatch"), (re.compile(r"\bProducti[o0]n\b"), "Production"),
    (re.compile(r"\bMi[l1]{2}i[o0]n\b"), "Million"), (re.compile(r"\bT[o0]nnes\b"), "Tonnes"),
    (re.compile(r"\bC[o0]al\b"), "Coal"), (re.compile(r"\bCOALINDIA\b"), "COAL INDIA"),
    (re.compile(r"(?<=\d)[oO](?=\d|\b)"), "0"), (re.compile(r"(?<=\d)[lI](?=\d)"), "1"),
]


@dataclass
class OcrLine:
    bbox: list[float]     # [x0,y0,x1,y1] in pixel coords of the input image
    text: str
    confidence: float


def get_engine():
    """Lazy singleton over RapidOCR (PP-OCRv4 ONNX, CPU).  Supports both package generations:
    legacy ``rapidocr_onnxruntime`` (Python <3.13) and the current ``rapidocr`` (>=2.0, Python 3.13 OK)."""
    global _engine
    with _lock:
        if _engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # legacy package
            except ImportError:
                from rapidocr import RapidOCR  # current package (returns RapidOCROutput)
            _engine = RapidOCR()
    return _engine


def _iter_results(raw):
    """Normalise engine output to (box, text, conf) triples across RapidOCR versions."""
    if raw is None:
        return []
    if isinstance(raw, tuple):  # legacy: (result, elapse)
        raw = raw[0]
    if raw is None:
        return []
    if hasattr(raw, "boxes") and hasattr(raw, "txts"):  # rapidocr>=2 RapidOCROutput
        if raw.boxes is None or raw.txts is None:
            return []
        scores = raw.scores if raw.scores is not None else [1.0] * len(raw.txts)
        return list(zip(raw.boxes, raw.txts, scores))
    return list(raw)


def domain_correct(text: str) -> str:
    out = text
    for rx, rep in _FIXES:
        out = rx.sub(rep, out)
    return out


def run_ocr(gray: np.ndarray) -> list[OcrLine]:
    eng = get_engine()
    lines: list[OcrLine] = []
    for box, txt, conf in _iter_results(eng(gray)):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        t = domain_correct(str(txt).strip())
        if not t:
            continue
        lines.append(OcrLine([float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))], t, float(conf)))
    lines.sort(key=lambda l: (round(l.bbox[1] / 12), l.bbox[0]))
    return lines


def _garbage_ratio(text: str) -> float:
    """Share of OCR tokens that do not look like printed words/numbers ('Plo+', '-NF-111.', 'k01', single letters)."""
    toks = [t for t in re.split(r"\s+", (text or "").strip()) if t]
    if not toks:
        return 0.0
    bad = 0
    for t in toks:
        core = t.strip(".,;:()[]'\"")
        if not core:
            continue
        # Support English and Indic scripts (Devanagari \u0900-\u097F, Bengali \u0980-\u09FF, etc.)
        if re.fullmatch(r"[\u0900-\u0D7FA-Za-z]{2,}|\d[\d.,/-]*|[A-Z]\.|[\u0900-\u0D7FA-Za-z]+-[\u0900-\u0D7FA-Za-z]+|\d+(st|nd|rd|th)", core):
            continue
        bad += 1
    return bad / len(toks)


def handwriting_score(gray_crop: np.ndarray, ocr_conf: float, text: str = "", height_ratio: float = 1.0) -> float:
    """Heuristic handwriting / signature likelihood for one OCR line: low-to-medium OCR confidence, garbage-looking
    text, a line much taller than the page's printed text, irregular stroke components.  Printed stamps (high
    confidence, clean digits) score low; cursive annotations score high.  A trained classifier (TrOCR / CNN) can
    replace this function without touching the pipeline."""
    import cv2
    if gray_crop.size == 0:
        return 0.0
    _, binv = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    if n < 4:
        return 0.0
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    cv_h = hs.std() / (hs.mean() + 1e-6)
    stroke = min(cv_h, 1.5) / 1.5
    tall = float(np.clip((height_ratio - 1.5) / 2.0, 0.0, 1.0))
    s = 0.35 * (1.0 - ocr_conf) + 0.25 * _garbage_ratio(text) + 0.25 * tall + 0.15 * stroke
    if ocr_conf > 0.88:
        s *= 0.5
    return float(np.clip(s, 0.0, 1.0))
