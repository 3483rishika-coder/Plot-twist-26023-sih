"""Layout detection + table intelligence (PRD §11.3.2, §11.3.5).

Works on *words with boxes* so the same algorithms serve digital PDFs (text
layer via PyMuPDF) and scanned pages (OCR lines split into pseudo-words).

Pipeline per page:
  words -> lines -> table candidates (ruled tables from PyMuPDF + borderless
  numeric-grid detection) -> word-grid table reconstruction (rows by y
  clustering, columns by robust x-projection of body rows, merged header cells
  propagated to spanned columns, wrapped labels re-attached) -> remaining text
  blocks -> region typing (title / caption / header / footer / page_number /
  figure / chart) -> XY-cut reading order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

import numpy as np

from .normalize import MONTHS, is_numeric_token

TITLE_LINE_RE = re.compile(r"^(table|statement|annexure|\d{1,2}(\.\d{1,2})?\.?\s+[A-Z])|\b(wise|summary|status|statement)\b", re.I)
CAPTION_RE = re.compile(r"^(table|statement|chart|fig\.?|figure|annexure|exhibit|graph)\b|^\d{1,2}(\.\d{1,2})?\.?\s+[A-Z]", re.I)
UNIT_HINT_RE = re.compile(
    r"\(?\s*(?:fig(?:ures?)?\.?\s*in|qty\.?\s*in|quantity\s*in|resources?\s*in|in|figures\s*in)?\s*"
    r"(million\s*tonnes?|lakh\s*tonnes?|thousand\s*tonnes?|'?000\s*tonnes?|th\.?\s*tonnes?|tonnes?|m\.?\s*cum|million\s*cum|mt|mte|rs\.?\s*crore|₹\s*crore|crore|%|nos\.?)\s*\)?",
    re.I)
AGG_RE = re.compile(r"^(grand\s+total|total|sub[- ]?total|all\s+india|cil\s+total|overall|india)\b", re.I)
PROSE_RE = re.compile(r"[a-z]{3,}[,.;:]?\s+[a-z]{2,}[,.;:]?\s+[a-z]{3,}[,.;:]?\s+[a-z]{3,}", re.I)


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    conf: float = 1.0
    block: int = 0

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def h(self):
        return self.y1 - self.y0


def value_like(tok: str, prev: Optional[str]) -> bool:
    """Numeric token that looks like a data value (not a header artefact like FY 25, (331), 2023- 24)."""
    t = tok.strip()
    core = t.strip("()%").replace(",", "")
    if not is_numeric_token(core):
        return False
    p = (prev or "").lower().rstrip("'-")
    if p in ("fy", "f.y.", "fy'", "year") or p[:3] in MONTHS:
        return False
    if re.fullmatch(r"\(\d{3}\)", t):           # (331) resource category codes
        return False
    if re.fullmatch(r"\d{2}", core) and (prev or "").endswith("-"):   # wrapped 2023- / 24
        return False
    if re.fullmatch(r"(19|20)\d{2}", core) and len(t) == 4:          # bare year -> header-ish
        return False
    if re.fullmatch(r"\d{1,2}\.", t) or re.fullmatch(r"\d{1,2}\.\d{1,2}\.?", t) and prev is None:   # '6.' / '9.1' heading numbers
        return False
    return True


@dataclass
class Line:
    words: list[Word]

    @property
    def bbox(self):
        return [min(w.x0 for w in self.words), min(w.y0 for w in self.words),
                max(w.x1 for w in self.words), max(w.y1 for w in self.words)]

    @property
    def sorted_words(self):
        return sorted(self.words, key=lambda w: w.x0)

    @property
    def text(self):
        return " ".join(w.text for w in self.sorted_words)

    @property
    def cy(self):
        b = self.bbox
        return (b[1] + b[3]) / 2

    @property
    def h(self):
        return median([w.h for w in self.words])

    def value_count(self):
        ws = self.sorted_words
        return sum(1 for i, w in enumerate(ws) if value_like(w.text, ws[i - 1].text if i else None))

    def text_count(self):
        ws = self.sorted_words
        return sum(1 for i, w in enumerate(ws) if not value_like(w.text, ws[i - 1].text if i else None) and re.search(r"[A-Za-z]{2,}", w.text))

    def big_gaps(self, thr):
        ws = self.sorted_words
        return sum(1 for a, b in zip(ws, ws[1:]) if b.x0 - a.x1 > thr)

    def strong_count(self):
        ws = self.sorted_words
        n = 0
        for i, w in enumerate(ws):
            if value_like(w.text, ws[i - 1].text if i else None):
                core = w.text.strip("()%▲▼").replace(",", "")
                if "." in core or len(re.sub(r"\D", "", core)) >= 3 or w.text.endswith("%"):
                    n += 1
        return n

    def is_prose(self):
        t = self.text
        ws = self.words
        if t.rstrip().endswith(":") and len(ws) >= 3:
            return True
        alpha = sum(1 for w in ws if re.fullmatch(r"[A-Za-z][A-Za-z'’\-]*[,.;:]?", w.text))
        if len(ws) >= 9 and alpha / len(ws) >= 0.6:
            return True
        return len(t) > 70 and bool(PROSE_RE.search(t)) and self.value_count() <= 1


@dataclass
class CellOut:
    row: int
    col: int
    bbox: list[float]
    text: str
    conf: float


@dataclass
class TableOut:
    bbox: list[float]
    headers: list[str]
    header_rows: int
    rows: list[list[str]]
    cells: list[CellOut]
    row_bboxes: list[list[float]]
    col_bounds: list[tuple[float, float]]
    caption: str = ""
    unit_hint: str = ""
    method: str = "wordgrid"
    confidence: float = 0.8
    meta: dict = field(default_factory=dict)


@dataclass
class RegionOut:
    type: str
    bbox: list[float]
    text: str
    confidence: float
    reading_order: int = 0
    meta: dict = field(default_factory=dict)
    table: Optional[TableOut] = None


# ---------------------------------------------------------------------------
# words & lines

def words_from_page(page) -> list[Word]:
    out = []
    try:
        raw = page.get_text("words")
    except Exception:
        return out
    for x0, y0, x1, y1, t, bno, lno, wno in raw:
        t = t.strip()
        if t:
            out.append(Word(x0, y0, x1, y1, t, 1.0, bno))
    return out


def font_stats(page) -> dict:
    sizes = {}
    allsz = []
    try:
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                for s in l["spans"]:
                    if s["text"].strip():
                        sizes[round(s["bbox"][1])] = s["size"]
                        allsz.append(s["size"])
    except Exception:
        pass
    sizes["median"] = median(allsz) if allsz else 10
    return sizes


def words_from_ocr(lines, scale: float) -> list[Word]:
    """Split OCR line boxes into pseudo-words proportionally to character length."""
    out = []
    for i, l in enumerate(lines):
        x0, y0, x1, y1 = [v * scale for v in l.bbox]
        toks = l.text.split()
        if not toks:
            continue
        total = sum(len(t) for t in toks) + (len(toks) - 1)
        cur = x0
        width = x1 - x0
        for t in toks:
            w = width * (len(t) / max(total, 1))
            out.append(Word(cur, y0, cur + w, y1, t, l.confidence, i))
            cur += w + width * (1 / max(total, 1))
    return out


def cluster_lines(words: list[Word], tol: Optional[float] = None) -> list[Line]:
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w.cy, w.x0))
    hmed = median([w.h for w in ws]) or 8
    tol = tol or max(2.5, hmed * 0.5)
    lines: list[Line] = []
    for w in ws:
        if lines and abs(lines[-1].cy - w.cy) <= tol:
            lines[-1].words.append(w)
        else:
            lines.append(Line([w]))
    return lines


# ---------------------------------------------------------------------------
# table detection

def _tabular(line: Line, gap_thr: float) -> bool:
    if line.is_prose():
        return False
    n = line.value_count()
    if n >= 2:
        return True
    if n >= 1 and line.text_count() >= 1 and line.big_gaps(gap_thr) >= 1 and len(line.words) <= 14:
        return True
    return False


def _short_label(line: Line) -> bool:
    return len(line.words) <= 8 and len(line.text) <= 60 and line.value_count() == 0 and not line.is_prose()


def detect_table_boxes(lines: list[Line], page_w: float, page_h: float, hints: list[list[float]]) -> list[list[float]]:
    """Table bboxes = borderless numeric grids ∪ ruled-table hints (merged when overlapping)."""
    if not lines:
        return [list(h) for h in hints]
    hmed = median([l.h for l in lines]) or 8
    gap_thr = hmed * 1.8
    cands: list[list[Line]] = []
    run: list[Line] = []
    prev: Optional[Line] = None

    def flush():
        nonlocal run
        tab = [l for l in run if _tabular(l, gap_thr)]
        good = [l for l in tab if l.value_count() >= 2 or (l.value_count() >= 1 and l.text_count() >= 1)]
        labelled = [l for l in tab if len(l.words) - l.value_count() >= 1]
        if tab:
            rb = [min(l.bbox[0] for l in tab), min(l.bbox[1] for l in tab), max(l.bbox[2] for l in tab), max(l.bbox[3] for l in tab)]
            ruled = any(_overlap_ratio(rb, h) > 0.3 for h in hints)
        else:
            ruled = False
        need = 2 if ruled else 3
        if len(tab) >= need and len(good) >= max(2 if ruled else 2, int(0.5 * len(tab))) and len(labelled) >= (1 if ruled else 2):
            cands.append(run[:])
        run = []

    for l in lines:
        if run and prev is not None and (l.bbox[1] - prev.bbox[3]) > hmed * 3.0:
            flush()
        if _tabular(l, gap_thr):
            run.append(l)
        elif run and _short_label(l):
            run.append(l)              # wrapped label / sub heading inside the table
        else:
            flush()
        prev = l
    flush()

    # split runs where the column structure changes around a label/caption line (stacked tables)
    split_cands: list[list[Line]] = []
    for run in cands:
        cur: list[Line] = []
        for k, l in enumerate(run):
            if cur and _short_label(l) and k + 1 < len(run):
                before = [x.value_count() for x in cur if _tabular(x, gap_thr)]
                after = [x.value_count() for x in run[k + 1:k + 6] if _tabular(x, gap_thr)]
                title_like = len(before) >= 2 and cur[-1].value_count() >= 1 and len(l.words) >= 3 and TITLE_LINE_RE.search(l.text)
                if (len(before) >= 2 and len(after) >= 2 and abs(median(before) - median(after)) >= 2) or title_like:
                    split_cands.append(cur); cur = []
            cur.append(l)
        if cur:
            split_cands.append(cur)
    cands = [c for c in split_cands if sum(1 for l in c if _tabular(l, gap_thr)) >= 3]
    # two small tables printed side by side share the same text lines: split on a wide empty vertical gutter
    side_split: list[list[Line]] = []
    for run in cands:
        side_split.extend(_split_side_by_side(run, hmed))
    cands = side_split

    boxes = []
    for run in cands:
        # trim trailing label-only lines
        while run and _short_label(run[-1]):
            run.pop()
        if not run:
            continue
        x0 = min(l.bbox[0] for l in run); x1 = max(l.bbox[2] for l in run)
        y0 = min(l.bbox[1] for l in run); y1 = max(l.bbox[3] for l in run)
        # header extension upwards
        above = [l for l in lines if l.bbox[3] <= y0 + 1 and l.bbox[3] > y0 - hmed * 9 and l.bbox[0] >= x0 - page_w * 0.12 and l.bbox[2] <= x1 + page_w * 0.12]
        above.sort(key=lambda l: -l.bbox[1])
        last_top = y0
        for l in above:
            # stop at prose, long lines, real data rows; header fragments such as '24 December 25)' are weakly
            # tabular (no strong numbers) and belong to the header block
            if last_top - l.bbox[3] > hmed * 2.2 or l.is_prose() or len(l.text) > 90 or (_tabular(l, gap_thr) and l.strong_count() >= 1):
                break
            y0 = min(y0, l.bbox[1]); x0 = min(x0, l.bbox[0]); x1 = max(x1, l.bbox[2])
            last_top = l.bbox[1]
            if l.text.rstrip().endswith(":"):
                break              # lead-in line ('... are given below:') closes the caption block
        # label lines to the left inside the vertical span
        for l in lines:
            if l.bbox[1] >= y0 - 1 and l.bbox[3] <= y1 + 1 and l.bbox[2] <= x0 + 2 and x0 - l.bbox[2] < hmed * 10 and _short_label(l):
                x0 = min(x0, l.bbox[0])
        boxes.append([x0 - 2, y0 - 2, x1 + 2, y1 + 2])

    allb = boxes + [_expand_hint(list(h), lines, hmed) for h in hints]
    merged = True
    while merged:
        merged = False
        out: list[list[float]] = []
        while allb:
            b = allb.pop()
            for i, o in enumerate(out):
                if _overlap_ratio(b, o) > 0.15:
                    out[i] = [min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3])]
                    merged = True
                    break
            else:
                out.append(b)
        allb = out
    return sorted(allb, key=lambda b: (b[1], b[0]))


def _split_side_by_side(run: list[Line], hmed: float) -> list[list[Line]]:
    """Split a run of lines into two tables when a wide empty vertical gutter separates two numeric grids."""
    words = [w for l in run for w in l.words]
    if len(words) < 8:
        return [run]
    x0 = min(w.x0 for w in words); x1 = max(w.x1 for w in words)
    W = x1 - x0
    if W < hmed * 18:
        return [run]
    n = int(W) + 2
    prof = np.zeros(n, dtype=np.int32)
    for w in words:
        a = int(max(0, w.x0 - x0)); b = int(min(n - 1, w.x1 - x0))
        prof[a:b + 1] += 1
    gaps = []
    i = 0
    while i < n:
        if prof[i] == 0:
            j = i
            while j < n and prof[j] == 0:
                j += 1
            if j - i >= hmed * 3 and i > 0.2 * W and j < 0.8 * W:
                gaps.append((x0 + i, x0 + j))
            i = j
        else:
            i += 1
    if not gaps:
        return [run]
    g = max(gaps, key=lambda g: g[1] - g[0])
    sep = (g[0] + g[1]) / 2
    left, right, inner_gaps = [], [], []
    for l in run:
        lw = [w for w in l.words if w.cx < sep]; rw = [w for w in l.words if w.cx >= sep]
        for side in (lw, rw):
            ws = sorted(side, key=lambda w: w.x0)
            if len(ws) >= 2:
                inner_gaps.append(max(b.x0 - a.x1 for a, b in zip(ws, ws[1:])))
        if lw:
            left.append(Line(lw))
        if rw:
            right.append(Line(rw))
    if inner_gaps and (g[1] - g[0]) < 2.0 * median(inner_gaps):
        return [run]

    def ok(ls):
        return sum(1 for l in ls if l.value_count() >= 1) >= 2 and sum(1 for l in ls if l.text_count() >= 1) >= 1
    if ok(left) and ok(right):
        return [left, right]
    return [run]


def _expand_hint(box, lines, hmed):
    """Ruled-table boxes from PyMuPDF sometimes miss unruled label/last columns: attach nearby words on the same rows."""
    x0, y0, x1, y1 = box
    changed = True
    while changed:
        changed = False
        for l in lines:
            if l.cy < y0 - 1 or l.cy > y1 + 1:
                continue
            ws = l.sorted_words
            inside = [w for w in ws if x0 - 1 <= w.cx <= x1 + 1]
            if not inside:
                continue
            if l.is_prose():
                continue
            for w in ws:
                if w.cx < x0 and x0 - w.x1 < hmed * 4:
                    x0 = min(x0, w.x0 - 1); changed = True
                elif w.cx > x1 and w.x0 - x1 < hmed * 4:
                    x1 = max(x1, w.x1 + 1); changed = True
    return [x0, y0, x1, y1]


def _overlap_ratio(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(area, 1e-6)


def _inside(w: Word, b, pad=1.0) -> bool:
    return b[0] - pad <= w.cx <= b[2] + pad and b[1] - pad <= w.cy <= b[3] + pad


# ---------------------------------------------------------------------------
# table reconstruction

def build_table(words: list[Word], bbox: list[float], hint_cols: Optional[list[tuple[float, float]]] = None,
                hint_cells: Optional[list[list[float]]] = None, hint_ncols: Optional[int] = None) -> Optional[TableOut]:
    tw = [w for w in words if _inside(w, bbox)]
    if len(tw) < 4:
        return None
    lines = cluster_lines(tw)
    if len(lines) < 2:
        return None
    hmed = median([l.h for l in lines]) or 8
    gap_thr = hmed * 1.2

    def is_body(l: Line) -> bool:
        if l.is_prose():
            return False
        vc, sc = l.value_count(), l.strong_count()
        return sc >= 2 or vc >= 3 or (sc >= 1 and l.text_count() >= 1 and l.big_gaps(gap_thr) >= 1)

    body_idx = [i for i, l in enumerate(lines) if is_body(l)]
    if not body_idx:
        return None
    first_body = body_idx[0]

    def _year_subheader(l: Line) -> bool:
        # '21 22 23 24 21 22 23 24' under 'Fatal Accidents | Fatalities': calendar-year sub-header, not data
        toks = [w.text.strip() for w in l.sorted_words]
        nums = [t for t in toks if re.fullmatch(r"(19|20)?\d{2}", t)]
        if len(nums) < 3 or len(nums) < len(toks) - 1 or l.text_count() > 0:
            return False
        return all(10 <= int(t[-2:]) <= 35 for t in nums)
    def _colnum_row(l: Line) -> bool:
        toks = [w.text.strip() for w in l.sorted_words]
        return len(toks) >= 3 and all(re.fullmatch(r"\(\d{1,2}\)|\d{1,2}", t) for t in toks) and any(t.startswith("(") for t in toks)
    while first_body < len(lines) - 1 and (_year_subheader(lines[first_body]) or _colnum_row(lines[first_body])):
        first_body += 1
    header_rows = min(first_body, 12)
    header_lines = lines[first_body - header_rows:first_body]
    body_lines = lines[first_body:]
    numeric_body = [l for l in body_lines if is_body(l)]

    # robust column separators: coverage fraction of body rows
    x0, x1 = bbox[0], bbox[2]
    W = int(max(10, x1 - x0)) + 2
    prof = np.zeros(W, dtype=np.int32)
    for l in numeric_body:
        for w in l.words:
            a = int(max(0, w.x0 - x0)); b = int(min(W - 1, w.x1 - x0))
            prof[a:b + 1] += 1
    thr = max(0, int(0.15 * len(numeric_body)))
    min_gap = max(3, int(hmed * 0.45))
    cols: list[tuple[float, float]] = []
    in_col, start, i = False, 0, 0
    while i < W:
        if prof[i] > thr and not in_col:
            in_col, start = True, i
        elif prof[i] <= thr and in_col:
            j = i
            while j < W and prof[j] <= thr:
                j += 1
            if j - i >= min_gap or j >= W:
                cols.append((x0 + start, x0 + i)); in_col = False
            i = j
            continue
        i += 1
    if in_col:
        cols.append((x0 + start, x0 + W))
    # adopt ruled-table hint columns if they are finer and don't cut through any word
    if hint_cols and len(hint_cols) > len(cols):
        ok = True
        for k in range(1, len(hint_cols)):
            sep = (hint_cols[k - 1][1] + hint_cols[k][0]) / 2
            if any(w.x0 < sep - 0.5 < w.x1 for l in numeric_body for w in l.words):
                ok = False; break
        if ok:
            cols = [(a, b) for a, b in hint_cols]
    cols = _split_dense_columns(cols, numeric_body, hmed)
    if len(cols) < 2:
        return None
    bounds = []
    for k, (a, b) in enumerate(cols):
        left = bbox[0] if k == 0 else (cols[k - 1][1] + a) / 2
        right = bbox[2] if k == len(cols) - 1 else (b + cols[k + 1][0]) / 2
        bounds.append((left, right))

    def col_of(w: Word) -> int:
        for k, (l, r) in enumerate(bounds):
            if l <= w.cx < r:
                return k
        return len(bounds) - 1 if w.cx >= bounds[-1][1] else 0

    # header labels (caption-like header lines are kept aside, not propagated to columns)
    headers = [""] * len(cols)
    tight = hmed * 0.7
    caption_lines: list[str] = []
    kept = []
    for hi, hl in enumerate(header_lines):
        groups_ = _group_cells(hl.words, tight)
        t = hl.text.strip()
        one_cell = len(groups_) == 1
        wide = (hl.bbox[2] - hl.bbox[0]) > 0.5 * (bbox[2] - bbox[0])
        capish = bool(CAPTION_RE.match(t)) or bool(re.fullmatch(r"[\[(].{0,45}[\])]", t)) or (t.isupper() and len(t.split()) >= 3) or t.endswith("-") or t.endswith(":")
        if one_cell and (capish or (wide and hi == 0)):
            caption_lines.append(t)
        elif one_cell and wide and UNIT_HINT_RE.search(t) and len(t) < 60:
            caption_lines.append(t)
        else:
            kept.append(hl)
    header_lines = kept
    col_centers = [(l + r) / 2 for l, r in bounds]
    col_w = median([b - a for a, b in cols]) if cols else 30
    body_top = body_lines[0].bbox[1] if body_lines else bbox[3]
    hcells = [c for c in (hint_cells or []) if (c[1] + c[3]) / 2 < body_top and c[3] - c[1] > 2]
    if hint_ncols is not None and hint_ncols < len(cols) - 1:
        hcells = []          # the ruled grid disagrees with the word grid: its merged header cells are not trustworthy
    if hcells and header_lines:
        # ruled table: header words -> ruled cell rect -> spanned columns (precise merged-cell semantics)
        groups: dict[int, list[Word]] = {}
        loose: list[Word] = []
        for hl in header_lines:
            for w in hl.words:
                for ci, c in enumerate(hcells):
                    vov = min(w.y1, c[3]) - max(w.y0, c[1])
                    if c[0] - 1 <= w.cx <= c[2] + 1 and vov >= 0.6 * max(w.h, 1e-3):
                        groups.setdefault(ci, []).append(w); break
                else:
                    loose.append(w)
        for ci, ws in sorted(groups.items(), key=lambda kv: (hcells[kv[0]][1], hcells[kv[0]][0])):
            c = hcells[ci]
            txt = " ".join(w.text for w in sorted(ws, key=lambda w: (round(w.y0 / max(hmed, 1)), w.x0)))
            spanned = [k for k, cc in enumerate(col_centers) if c[0] - 1 <= cc <= c[2] + 1]
            if not spanned:
                spanned = [col_of(ws[0])]
            for k in spanned:
                _append_header(headers, k, txt)
        header_word_lines = cluster_lines(loose) if loose else []
    else:
        header_word_lines = header_lines
    for hi_, hl in enumerate(header_word_lines):
        cells_in_line = _group_cells(hl.words, tight)
        below = [_group_cells(x.words, tight) for x in header_word_lines[hi_ + 1:hi_ + 3]]
        for cell in cells_in_line:
            cols_hit = sorted({col_of(w) for w in cell})
            cx0 = min(w.x0 for w in cell); cx1 = max(w.x1 for w in cell)
            # merged cell over sub-headers: overlaps 2+ cells of a lower header line and is centred over their union
            span_cols = None
            for sub_line in below:
                hit = [(min(w.x0 for w in g), max(w.x1 for w in g)) for g in sub_line
                       if min(cx1, max(w.x1 for w in g)) - max(cx0, min(w.x0 for w in g)) > 1.0]
                if len(hit) >= 2:
                    u0, u1 = min(h[0] for h in hit), max(h[1] for h in hit)
                    if abs((u0 + u1) / 2 - (cx0 + cx1) / 2) <= 0.35 * col_w:
                        ks = sorted({col_of(w) for g in sub_line for w in g if u0 - 1 <= w.cx <= u1 + 1})
                        if len(ks) >= 2:
                            span_cols = list(range(min(ks), max(ks) + 1))
                            break
            if span_cols:
                txt = " ".join(w.text for w in sorted(cell, key=lambda w: w.x0))
                for k in span_cols:
                    _append_header(headers, k, txt)
                continue
            if len(cols_hit) > 1 and _words_centered(cell, bounds, col_of):
                groups = {}
                for w in cell:
                    groups.setdefault(col_of(w), []).append(w)
                for k, ws in groups.items():
                    _append_header(headers, k, " ".join(w.text for w in sorted(ws, key=lambda w: w.x0)))
                continue
            txt = " ".join(w.text for w in sorted(cell, key=lambda w: w.x0))
            spanned = [k for k, (l, r) in enumerate(bounds) if min(cx1, r) - max(cx0, l) > 0.3 * min(cx1 - cx0 + 1e-3, r - l)]
            if len(cols_hit) > 1 or (spanned and len(spanned) > 1):
                # merged header over several columns: bounded Voronoi partition among cells of this line
                others = [(min(w.x0 for w in oc), max(w.x1 for w in oc)) for oc in cells_in_line if oc is not cell]
                ext = 0.9 * col_w
                spanned = []
                for k, cc in enumerate(col_centers):
                    if cc < cx0 - ext or cc > cx1 + ext:
                        continue
                    d_self = 0 if cx0 <= cc <= cx1 else min(abs(cc - cx0), abs(cc - cx1))
                    d_oth = min([0 if o0 <= cc <= o1 else min(abs(cc - o0), abs(cc - o1)) for o0, o1 in others], default=1e9)
                    if d_self <= d_oth:
                        spanned.append(k)
            if not spanned:
                spanned = [col_of(cell[0])]
            for k in spanned:
                _append_header(headers, k, txt)

    rows: list[list[str]] = []
    row_words: list[list[list[Word]]] = []
    for l in body_lines:
        rw = [[] for _ in cols]
        for w in l.sorted_words:
            rw[col_of(w)].append(w)
        row_words.append(rw)
    row_words = _attach_label_rows(row_words, is_body_words=lambda rw: _rw_is_body(rw, gap_thr), hmed=hmed)

    cells_out: list[CellOut] = []
    row_bboxes = []
    for ri, rw in enumerate(row_words):
        row = []
        allw = [w for ws in rw for w in ws]
        for k, ws in enumerate(rw):
            txt = " ".join(w.text for w in sorted(ws, key=lambda w: (round(w.y0 / max(hmed, 1)), w.x0)))
            row.append(txt)
            if ws:
                b = [min(w.x0 for w in ws), min(w.y0 for w in ws), max(w.x1 for w in ws), max(w.y1 for w in ws)]
                cells_out.append(CellOut(ri, k, [round(v, 1) for v in b], txt, float(np.mean([w.conf for w in ws]))))
        rows.append(row)
        row_bboxes.append([round(min(w.x0 for w in allw), 1), round(min(w.y0 for w in allw), 1), round(max(w.x1 for w in allw), 1), round(max(w.y1 for w in allw), 1)])

    n_numeric = sum(1 for r in rows for c in r if c and is_numeric_token(c.strip("()%▲▼ ")))
    n_cells = sum(1 for r in rows for c in r if c)
    conf = 0.55 + 0.35 * (n_numeric / max(n_cells, 1)) + (0.1 if any(headers) else 0.0)
    conf = float(min(conf, 0.98)) * float(np.mean([w.conf for w in tw]))
    t = TableOut([round(v, 1) for v in bbox], headers, header_rows, rows, cells_out, row_bboxes, bounds,
                 confidence=round(conf, 3), meta={"n_lines": len(lines), "caption_lines": caption_lines})
    if caption_lines:
        t.caption = " ".join(caption_lines)[:300]
        for cl in caption_lines:
            m = UNIT_HINT_RE.search(cl)
            if m and not t.unit_hint and len(m.group(0)) < 40 and m.group(1).strip() != "%":
                t.unit_hint = m.group(1)
    return t


def _split_dense_columns(cols, numeric_body, hmed):
    """If a projection column contains 2+ numeric words in the same row (in 2+ rows), split it by right-edge clusters."""
    out = []
    for (a, b) in cols:
        multi = 0
        edges = []
        for l in numeric_body:
            ws = [w for w in l.sorted_words if a - 1 <= w.cx <= b + 1]
            nums = [w for i, w in enumerate(ws) if value_like(w.text, ws[i - 1].text if i else None)]
            if len(nums) >= 2:
                multi += 1
            edges.extend(w.x1 for w in nums)
        if multi >= 2 and edges:
            edges.sort()
            clusters = [[edges[0]]]
            for e in edges[1:]:
                if e - clusters[-1][-1] <= max(2.0, hmed * 0.35):
                    clusters[-1].append(e)
                else:
                    clusters.append([e])
            clusters = [c for c in clusters if len(c) >= 2]
            if len(clusters) >= 2:
                rights = [median(c) for c in clusters]
                lefts = [a] + [(rights[i] + rights[i + 1]) / 2 - (rights[i + 1] - rights[i]) / 2 + 0.5 for i in range(len(rights) - 1)]
                for i, r in enumerate(rights):
                    left = a if i == 0 else rights[i - 1] + 1
                    out.append((left, r if i < len(rights) - 1 else b))
                continue
        out.append((a, b))
    return out


def _append_header(headers, k, txt):
    if txt and txt not in headers[k]:
        headers[k] = (headers[k] + " " + txt).strip()


def _words_centered(cell: list[Word], bounds, col_of) -> bool:
    """True when each word sits near the centre of its own column (separate header cells, not a merged span)."""
    seen = set()
    for w in cell:
        k = col_of(w)
        if k in seen:
            return False
        seen.add(k)
        l, r = bounds[k]
        if abs(w.cx - (l + r) / 2) > 0.3 * (r - l):
            return False
    return True


def _group_cells(words: list[Word], gap_thr: float) -> list[list[Word]]:
    ws = sorted(words, key=lambda w: w.x0)
    groups: list[list[Word]] = []
    for w in ws:
        if groups and w.x0 - groups[-1][-1].x1 <= gap_thr:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def _rw_is_body(rw: list[list[Word]], gap_thr: float) -> bool:
    allw = sorted([w for ws in rw for w in ws], key=lambda w: w.x0)
    if not allw:
        return False
    l = Line(allw)
    vc, sc = l.value_count(), l.strong_count()
    return sc >= 2 or vc >= 3 or (sc >= 1 and l.text_count() >= 1)


def _attach_label_rows(row_words, is_body_words, hmed):
    """Merge label-only rows (wrapped entity names) into the *nearest* numeric row (by vertical distance)."""
    n = len(row_words)
    if n < 2:
        return row_words
    body = [is_body_words(rw) for rw in row_words]

    def ycen(rw):
        ws = [w for x in rw for w in x]
        return (min(w.y0 for w in ws) + max(w.y1 for w in ws)) / 2 if ws else None

    def is_label_row(rw):
        ws = sorted([w for x in rw for w in x], key=lambda w: w.x0)
        if not ws or len(ws) > 6:
            return False
        if any(value_like(w.text, ws[i - 1].text if i else None) for i, w in enumerate(ws)):
            return False
        return len(ws) - 1 == 0 or (ws[-1].x1 - ws[0].x0) < 6 * hmed * 2.5

    merged = [[list(ws) for ws in rw] for rw in row_words]
    drop = set()
    body_idx = [i for i in range(n) if body[i]]
    if not body_idx:
        return row_words
    for i in range(n):
        if body[i] or not is_label_row(row_words[i]):
            continue
        yc = ycen(row_words[i])
        j = min(body_idx, key=lambda k: abs(ycen(row_words[k]) - yc))
        if abs(ycen(row_words[j]) - yc) <= hmed * 1.9:
            ws = sorted([w for x in row_words[i] for w in x], key=lambda w: w.x0)
            # target column = column of the leftmost label word, or first column holding text in the body row
            k = next((ci for ci, cw in enumerate(row_words[i]) if cw), 0)
            merged[j][k] = merged[j][k] + ws
            drop.add(i)
    return [merged[i] for i in range(n) if i not in drop]


# ---------------------------------------------------------------------------
# regions

def build_regions(words: list[Word], tables: list[TableOut], page_w: float, page_h: float,
                  figure_boxes: list[list[float]], chart_boxes: list[list[float]], font_sizes: Optional[dict] = None) -> list[RegionOut]:
    regions: list[RegionOut] = []
    tboxes = [t.bbox for t in tables]
    free = [w for w in words if not any(_inside(w, b) for b in tboxes)]
    lines = cluster_lines(free)
    hmed = median([l.h for l in lines]) if lines else 10
    # split lines at column gutters (multi-column pages) so blocks form per column
    gutters = detect_gutters(lines, page_w)
    split: list[Line] = []
    for l in lines:
        ws = l.sorted_words
        cur = [ws[0]]
        for a, b in zip(ws, ws[1:]):
            gap = b.x0 - a.x1
            crosses = any(a.x1 <= g0 + 1 and b.x0 >= g1 - 1 for g0, g1 in gutters)
            if gap > hmed * 3.0 or (crosses and gap >= 4):
                split.append(Line(cur)); cur = [b]
            else:
                cur.append(b)
        split.append(Line(cur))
    lines = split

    blocks: list[list[Line]] = []
    for l in sorted(lines, key=lambda l: l.bbox[1]):
        placed = False
        for blk in blocks:
            lb = blk[-1].bbox; cb = l.bbox
            if -2 <= cb[1] - lb[3] <= hmed * 0.9 and min(cb[2], lb[2]) - max(cb[0], lb[0]) > 0.3 * min(cb[2] - cb[0], lb[2] - lb[0]):
                blk.append(l); placed = True; break
        if not placed:
            blocks.append([l])

    for blk in blocks:
        bb = [min(l.bbox[0] for l in blk), min(l.bbox[1] for l in blk), max(l.bbox[2] for l in blk), max(l.bbox[3] for l in blk)]
        text = "\n".join(l.text for l in blk)
        conf = float(np.mean([w.conf for l in blk for w in l.words]))
        t1 = text.strip()
        rtype = "text"
        if (bb[3] < page_h * 0.075 or bb[1] > page_h * 0.91) and len(t1) < 90:
            rtype = "page_number" if re.fullmatch(r"\d{1,4}", t1) else ("header" if bb[3] < page_h * 0.075 else "footer")
        elif CAPTION_RE.match(t1) and len(t1) < 160:
            rtype = "caption"
        elif len(blk) <= 2 and len(t1) < 90 and (t1.isupper() or (font_sizes and _line_size(blk[0], font_sizes) > 1.25 * font_sizes.get("median", 10))):
            rtype = "title"
        if any(_overlap_ratio(bb, cb) > 0.6 for cb in chart_boxes):
            rtype = "chart_label"
        regions.append(RegionOut(rtype, [round(v, 1) for v in bb], text, round(conf, 3)))

    for t in tables:
        regions.append(RegionOut("table", t.bbox, "", t.confidence, table=t))
    for fb in figure_boxes:
        regions.append(RegionOut("figure", [round(v, 1) for v in fb], "", 0.9))
    for cb in chart_boxes:
        regions.append(RegionOut("chart", [round(v, 1) for v in cb], "", 0.8))

    for t in tables:
        cap, unit = "", ""
        cands = [r for r in regions if r.type in ("caption", "title", "text") and r.bbox[3] <= t.bbox[1] + 2 and t.bbox[1] - r.bbox[3] < hmed * 6 and _hoverlap(r.bbox, t.bbox) > 0.2]
        cands.sort(key=lambda r: -r.bbox[3])
        for r in cands[:3]:
            if not cap and (r.type in ("caption", "title") or len(r.text) < 120):
                cap = r.text.replace("\n", " ")
            m = UNIT_HINT_RE.search(r.text)
            if m and not unit and len(m.group(0)) < 40 and m.group(1).strip() != "%":
                unit = m.group(1)
        below = [r for r in regions if r.type in ("caption", "title") and r.bbox[1] >= t.bbox[3] - 2 and r.bbox[1] - t.bbox[3] < hmed * 3]
        if not cap and below:
            cap = below[0].text.replace("\n", " ")
        if not unit:
            for h in t.headers:
                m = UNIT_HINT_RE.search(h or "")
                if m and m.group(1).strip() != "%":
                    unit = m.group(1); break
        if cap and not t.caption:
            t.caption = cap[:300]
        elif cap and t.caption and cap not in t.caption:
            t.caption = (cap + " " + t.caption)[:300]
        if unit and not t.unit_hint:
            t.unit_hint = unit
    for r in regions:
        if r.table is not None:
            r.text = table_text(r.table)

    order = xy_cut_order([r.bbox for r in regions], page_w, page_h)
    for r, o in zip(regions, order):
        r.reading_order = o
    regions.sort(key=lambda r: r.reading_order)
    return regions


def detect_gutters(lines: list[Line], page_w: float) -> list[tuple[float, float]]:
    """Find vertical gutters: x-intervals where nearly every line that spans them has a gap."""
    W = int(page_w) + 1
    span = np.zeros(W, dtype=np.int32)
    gap = np.zeros(W, dtype=np.int32)
    n_multi = 0
    for l in lines:
        ws = l.sorted_words
        if len(ws) < 2:
            continue
        n_multi += 1
        lx0, lx1 = int(ws[0].x0), int(ws[-1].x1)
        span[max(0, lx0):min(W, lx1) + 1] += 1
        for a, b in zip(ws, ws[1:]):
            if b.x0 - a.x1 >= 4:
                gap[max(0, int(a.x1)):min(W, int(b.x0)) + 1] += 1
    if n_multi < 6:
        return []
    frac = np.where(span >= 5, gap / np.maximum(span, 1), 0.0)
    out = []
    x = int(page_w * 0.2)
    xmax = int(page_w * 0.8)
    while x < xmax:
        if frac[x] >= 0.85:
            x0 = x
            while x < xmax and frac[x] >= 0.85:
                x += 1
            if x - x0 >= 5:
                # a text-column gutter is a column edge: many lines start right after it (left-aligned prose);
                # the gap between a table's label and value columns is not (values start much further right)
                starts = 0
                for l in lines:
                    ws = l.sorted_words
                    if any(x - 3 <= wd.x0 <= x + 6 for wd in ws) and any(wd.x1 <= x0 + 1 for wd in ws):
                        starts += 1
                    elif ws and x - 3 <= ws[0].x0 <= x + 6:
                        starts += 1
                if starts >= 6:
                    out.append((float(x0), float(x)))
        x += 1
    return out


def _line_size(line: Line, font_sizes: dict) -> float:
    return font_sizes.get(round(line.bbox[1]), font_sizes.get("median", 10))


def _hoverlap(a, b) -> float:
    ov = min(a[2], b[2]) - max(a[0], b[0])
    return ov / max(min(a[2] - a[0], b[2] - b[0]), 1e-6)


def table_text(t: TableOut) -> str:
    parts = []
    if t.caption:
        parts.append(t.caption)
    parts.append(" | ".join(h for h in t.headers))
    for r in t.rows:
        parts.append(" | ".join(r))
    return "\n".join(parts)


def xy_cut_order(boxes: list[list[float]], page_w: float, page_h: float) -> list[int]:
    idx = list(range(len(boxes)))
    order: list[int] = []

    def rec(ids):
        if len(ids) <= 1:
            order.extend(ids); return
        xs = sorted((boxes[i][0], boxes[i][2]) for i in ids)
        vgap = max(_gaps(xs), key=lambda g: g[1] - g[0], default=None)
        ys = sorted((boxes[i][1], boxes[i][3]) for i in ids)
        hgap = max(_gaps(ys), key=lambda g: g[1] - g[0], default=None)
        if vgap and (vgap[1] - vgap[0]) > 12 and (not hgap or (hgap[1] - hgap[0]) < 30):
            left = [i for i in ids if boxes[i][2] <= vgap[0] + 1]; right = [i for i in ids if i not in left]
            if left and right:
                rec(left); rec(right); return
        if hgap and (hgap[1] - hgap[0]) > 2:
            top = [i for i in ids if boxes[i][3] <= hgap[0] + 1]; bottom = [i for i in ids if i not in top]
            if top and bottom:
                rec(top); rec(bottom); return
        order.extend(sorted(ids, key=lambda i: (boxes[i][1], boxes[i][0])))

    rec(idx)
    rank = {i: k for k, i in enumerate(order)}
    return [rank[i] for i in idx]


def _gaps(intervals):
    intervals = sorted(intervals)
    gaps = []
    cur_end = intervals[0][1]
    for a, b in intervals[1:]:
        if a > cur_end:
            gaps.append((cur_end, a))
        cur_end = max(cur_end, b)
    return gaps


def is_aggregate_label(label: str) -> bool:
    return bool(AGG_RE.match((label or "").strip()))
