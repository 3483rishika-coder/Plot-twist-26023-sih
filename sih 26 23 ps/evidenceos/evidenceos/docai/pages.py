"""Per-page Document AI: classify (digital/scanned) → quality → layout → OCR → tables → regions.

Output is a PageResult with all coordinates in PDF points, so evidence boxes are
resolution independent (the viewer scales them to whatever DPI it renders).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pymupdf

from .. import config
from . import layout, ocr, quality


@dataclass
class PageResult:
    page_number: int
    width: float
    height: float
    is_scanned: bool
    text: str
    regions: list[layout.RegionOut]
    tables: list[layout.TableOut]
    quality: dict
    ocr_confidence: Optional[float]
    meta: dict = field(default_factory=dict)


def render_gray(page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w).copy()


def is_scanned_page(page) -> bool:
    txt = page.get_text().strip()
    if len(txt) >= config.SCANNED_TEXT_THRESHOLD:
        return False
    return True


def analyze_page(doc: pymupdf.Document, pno: int, force_ocr: bool = False) -> PageResult:
    page = doc[pno]
    w, h = page.rect.width, page.rect.height
    scanned = force_ocr or is_scanned_page(page)
    hints: list[list[float]] = []
    hint_tables = []
    figure_boxes: list[list[float]] = []
    chart_boxes: list[list[float]] = []
    q: dict = {"digital": not scanned}
    ocr_conf = None
    font_sizes = None

    if not scanned:
        words = layout.words_from_page(page)
        font_sizes = layout.font_stats(page)
        try:
            hint_tables = page.find_tables(strategy="lines").tables
            hints = [list(t.bbox) for t in hint_tables]
        except Exception:
            hint_tables, hints = [], []
        figure_boxes = _image_boxes(page)
        chart_boxes = _chart_boxes(page, hints)
    else:
        dpi = config.OCR_DPI
        gray = render_gray(page, dpi)
        q = quality.analyze(gray, dpi)
        q["digital"] = False
        q["blank"] = bool(gray.std() < 3.0)
        proc = quality.preprocess(gray, q)
        lines = [] if q["blank"] else ocr.run_ocr(proc)
        scale = 72.0 / dpi
        words = layout.words_from_ocr(lines, scale)
        ocr_conf = float(np.mean([l.confidence for l in lines])) if lines else 0.0
        q["ocr_lines"] = len(lines)
        # handwriting / signature candidates: medium/low-confidence, garbage-looking, unusually tall lines;
        # neighbours of a strong candidate with similar height are pulled in (annotations span several lines)
        hw_boxes = []
        heights = [l.bbox[3] - l.bbox[1] for l in lines]
        med_h = float(np.median(heights)) if heights else 1.0
        scored = []
        for l in lines:
            x0, y0, x1, y1 = [int(v) for v in l.bbox]
            crop = proc[max(0, y0):y1, max(0, x0):x1]
            hr = (y1 - y0) / max(med_h, 1.0)
            sc = ocr.handwriting_score(crop, l.confidence, l.text, hr) if crop.size else 0.0
            scored.append((l, sc, hr))
        strong = [(l, sc, hr) for l, sc, hr in scored if sc >= 0.5]
        for l, sc, hr in scored:
            ok = sc >= 0.5
            if not ok and sc >= 0.33 and hr >= 2.0:
                cy = (l.bbox[1] + l.bbox[3]) / 2
                ok = any(abs(cy - (s_.bbox[1] + s_.bbox[3]) / 2) < 2.5 * (s_.bbox[3] - s_.bbox[1]) for s_, _, _ in strong)
            if ok:
                hw_boxes.append(([v * scale for v in l.bbox], l.text, l.confidence))
        q["handwriting_regions"] = len(hw_boxes)

    lines_all = layout.cluster_lines(words)
    boxes = _detect_tables_column_aware(words, lines_all, w, h, hints)
    tables: list[layout.TableOut] = []
    for b in boxes:
        hc, cells, hn = None, None, None
        for t in hint_tables:
            if layout._overlap_ratio(list(t.bbox), b) > 0.5:
                try:
                    cells = [list(c) for r in t.rows for c in r.cells if c]
                    hc = sorted({(round(c[0]), round(c[2])) for c in t.cells if c})
                    hn = t.col_count
                except Exception:
                    cells, hc, hn = None, None, None
        t = layout.build_table(words, b, hint_cols=hc, hint_cells=cells, hint_ncols=hn)
        if t and len(t.rows) >= 2:
            t.method = "wordgrid+ruled" if cells else ("ocrgrid" if scanned else "wordgrid")
            tables.append(t)

    regions = layout.build_regions(words, tables, w, h, figure_boxes, chart_boxes, font_sizes)
    if scanned:
        for bb, txt, conf in hw_boxes:
            regions.append(layout.RegionOut("handwriting", [round(v, 1) for v in bb], txt, round(conf, 3),
                                            reading_order=len(regions), meta={"needs_review": True}))
        for r in regions:
            if r.type in ("text", "title", "caption") and r.confidence < 0.6:
                r.meta["low_confidence"] = True

    body = [r for r in regions if r.type not in ("header", "footer", "page_number", "chart_label")]
    text = "\n\n".join(r.text for r in body if r.text)
    return PageResult(pno + 1, w, h, scanned, text, regions, tables, q, ocr_conf)


def _image_boxes(page) -> list[list[float]]:
    out = []
    try:
        pa = page.rect.width * page.rect.height
        for info in page.get_image_info():
            b = info.get("bbox")
            if not b:
                continue
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area > 0.015 * pa and area < 0.9 * pa:
                out.append([float(v) for v in b])
    except Exception:
        pass
    return out


def _detect_tables_column_aware(words, lines_all, w, h, hints) -> list[list[float]]:
    """Two-column pages (annual reports): small tables in one text column share their lines with prose in the
    other column, so detect tables per column; a full-width table is kept when ruled lines (PyMuPDF hint) or
    row structure show that it genuinely spans the gutter."""
    full = layout.detect_table_boxes(lines_all, w, h, hints)
    gutters = layout.detect_gutters(lines_all, w)
    if not gutters:
        return full
    seps = sorted((g0 + g1) / 2 for g0, g1 in gutters)
    keep: list[list[float]] = []
    for fb in full:
        seps_in = [sep for sep in seps if fb[0] < sep - 20 and fb[2] > sep + 20]
        if not seps_in:
            continue
        ruled_span = any(layout._overlap_ratio(hh, fb) > 0.5 and any(hh[0] < sep - 20 and hh[2] > sep + 20 for sep in seps_in) for hh in hints)
        if ruled_span:
            keep.append(fb); continue
        # unruled: most data rows must be short (non-prose) on both sides of the gutter
        rows = [l for l in lines_all if fb[1] - 1 <= l.cy <= fb[3] + 1 and l.value_count() >= 1]

        def _tab_both(l, sep):
            ws = l.sorted_words
            left = [x for x in ws if x.cx < sep]; right = [x for x in ws if x.cx >= sep]
            if not left or not right:
                return False
            return all(len(side) <= 6 or any(layout.is_numeric_token(x.text.strip("()%▲▼,")) for x in side) for side in (left, right)) \
                and not layout.Line(left).is_prose() and not layout.Line(right).is_prose()
        if rows and sum(1 for l in rows if any(_tab_both(l, sep) for sep in seps_in)) >= 0.7 * len(rows):
            keep.append(fb)
    # per-column detection on the words outside the kept wide tables
    rest = [wd for wd in words if not any(layout._inside(wd, kb) for kb in keep)]
    bounds = [0.0] + seps + [float(w)]
    out = list(keep)
    for a, b in zip(bounds, bounds[1:]):
        gw = [wd for wd in rest if a <= wd.cx < b]
        if len(gw) < 6:
            continue
        col_hints = [hh for hh in hints if hh[0] >= a - 5 and hh[2] <= b + 5]
        for pb in layout.detect_table_boxes(layout.cluster_lines(gw), w, h, col_hints):
            if not any(layout._overlap_ratio(pb, ob) > 0.3 for ob in out):
                out.append(pb)
    return sorted(out, key=lambda b: (b[1], b[0]))


def _chart_boxes(page, table_hints) -> list[list[float]]:
    """Cluster filled vector shapes (bars, pie slices) into chart regions."""
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    rects = []
    for d in drawings:
        r = d.get("rect")
        if r is None or d.get("fill") is None:
            continue
        if r.width * r.height < 4 or r.width < 1.5 or r.height < 1.5:
            continue
        if any(layout._overlap_ratio([r.x0, r.y0, r.x1, r.y1], hb) > 0.5 for hb in table_hints):
            continue
        rects.append([r.x0, r.y0, r.x1, r.y1])
    if len(rects) < 6:
        return []
    parent = list(range(len(rects)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    pad = 14
    for i in range(len(rects)):
        a = rects[i]
        for j in range(i + 1, len(rects)):
            b = rects[j]
            if a[0] - pad <= b[2] and b[0] - pad <= a[2] and a[1] - pad <= b[3] and b[1] - pad <= a[3]:
                parent[find(i)] = find(j)
    groups: dict[int, list[list[float]]] = {}
    for i, r in enumerate(rects):
        groups.setdefault(find(i), []).append(r)
    pa = page.rect.width * page.rect.height
    out = []
    for g in groups.values():
        if len(g) < 6:
            continue
        bb = [min(r[0] for r in g), min(r[1] for r in g), max(r[2] for r in g), max(r[3] for r in g)]
        area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        if 0.01 * pa < area < 0.8 * pa:
            out.append(bb)
    return out
