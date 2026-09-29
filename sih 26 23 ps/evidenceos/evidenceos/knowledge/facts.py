"""Fact extraction (PRD §11.4.2): tables and narrative sentences → subject-predicate-value facts with evidence.

Rules: no evidence → no fact; original value/unit always retained; per-dimension confidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..docai.normalize import (MONTHS, Period, fy_for_month, fy_key, normalize_unit, parse_number, parse_period)
from ..docai.layout import is_aggregate_label
from ..db import Entity
from .entities import Resolution, Resolver, norm_key

# --------------------------------------------------------------------------- header semantics

PREDICATE_PATTERNS = [
    ("stripping_ratio", re.compile(r"\bstripping\s+ratio\b|\bstrip\.?\s*ratio\b", re.I)),
    ("overburden", re.compile(r"\b(obr?\.?|overburden|o\.b\.|ob removal|obr\.)\b|\bover\b.{0,12}\bburden\b|\bburden\b.{0,12}\bover\b", re.I)),
    ("dispatch", re.compile(r"\b(despatch(?:es|ed)?|dispatch(?:es|ed)?|offtake|off-take|supply|supplies|loading)\b", re.I)),
    ("production", re.compile(r"\b(prod\.?|production|produced|output|raw coal)\b", re.I)),
    ("resources_measured", re.compile(r"\b(measured|proved)\b", re.I)),
    ("resources_indicated", re.compile(r"\bindicated\b", re.I)),
    ("resources_inferred", re.compile(r"\binferred\b", re.I)),
    ("resources_total", re.compile(r"\b(total resources?|resources?|reserves?|grand total)\b", re.I)),
    ("generation", re.compile(r"\bgeneration\b", re.I)),
    ("stock", re.compile(r"\b(stock|inventory)\b", re.I)),
    ("import", re.compile(r"\bimports?\b", re.I)),
    ("consumption", re.compile(r"\bconsumption\b", re.I)),
]
PCT_ACH_RE = re.compile(r"\b(achmt\.?|achiev\w*|ach\.?)\b|against\s+(the\s+)?target|%\s*of\s+target|\bsatisfaction\b", re.I)
PCT_GROWTH_RE = re.compile(r"\bgrowth\b", re.I)
PCT_SHARE_RE = re.compile(r"\b(%\s*share|share)\b", re.I)
TARGET_RE = re.compile(r"\btarget(s|ed)?\b", re.I)
ACTUAL_RE = re.compile(r"\bactual\b", re.I)
PROJECTED_RE = re.compile(r"\b(projected|projection|estimated|estimate|anticipated)\b", re.I)
ENTITY_COL_RE = re.compile(r"\b(subs\.?|subsidiary|subsidiaries|company|companies|mines?|state|states|sectors?|coalfield|particulars|name|source|item|category|type|depth range|year|companies)\b", re.I)
SERIAL_RE = re.compile(r"^\s*(sl|sr|s)\.?\s*no\.?\s*$", re.I)
YEAR_ROW_RE = re.compile(r"^(19|20)\d{2}\s*-\s*\d{2,4}$")
QUARTER_ROW_RE = re.compile(r"^(?:(20\d{2})\s*-\s*\d{2,4}\s*)?(1st|2nd|3rd|4th|first|second|third|fourth|q\s*[1-4])\s*(quarter|qtr\.?)?\s*$", re.I)
MONTH_ROW_RE = re.compile(r"^(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t|tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\.?\s*['’]?\s*(\d{2,4})?$", re.I)
HEADER_NOISE_RE = re.compile(r"\b(quantity|qty\.?|production|prod\.?|despatch|dispatch|offtake|share|month|growth|actual|target|in mt|mt)\b|\(\d+\)|\(%\)|%", re.I)
MONTH_IN_HEADER_RE = re.compile(r"\b(during|upto|up to|for|till)\s+([a-z]{3,9})\.?\s*'?\s*(\d{2,4})?", re.I)

TABLE_TOPIC_PREFIX = [
    ("lignite_", re.compile(r"\blignite\b", re.I)),
    ("opencast_", re.compile(r"\bopen\s*-?\s*cast\b|\bopencast\b|\bOC\b", re.I)),
    ("underground_", re.compile(r"\bunder\s*-?\s*ground\b|\bunderground\b|\bUG\b", re.I)),
    ("noncoking_", re.compile(r"\bnon\s*-?\s*coking\b", re.I)),
    ("coking_", re.compile(r"(?<!non[- ])(?<!non)\bcoking\b", re.I)),
    ("washed_", re.compile(r"\bwashed\b", re.I)),
]
SAFETY_RE = re.compile(r"\b(fatal\w*|injur\w*|accident\w*|casualt\w*|per\s+(?:million|mt|lakh|3\s*lakh)|rate\s+per|man[- ]?shifts?)\b", re.I)
SAFETY_PREDS = [("fatality_rate", re.compile(r"fatal\w*.*(rate|per)|(rate|per).*fatal", re.I)), ("serious_injury_rate", re.compile(r"serious.*(rate|per)|injur\w*.*(rate|per)", re.I)),
                ("fatal_accidents", re.compile(r"fatal\s+accident", re.I)), ("serious_accidents", re.compile(r"serious\s+accident", re.I)),
                ("fatalities", re.compile(r"fatal", re.I)), ("serious_injuries", re.compile(r"serious|injur", re.I)), ("accidents", re.compile(r"accident", re.I))]
SKIP_LABEL_RE = re.compile(r"^(top\s*\d+|others?\b.*|rest\b.*|balance\b.*|particulars|misc\.?)$|\bothers$", re.I)


@dataclass
class ColumnSpec:
    index: int
    header: str
    predicate: Optional[str] = None
    qualifier: str = "actual"
    period: Optional[Period] = None
    unit: Optional[str] = None
    unit_mult: float = 1.0
    unit_original: Optional[str] = None
    kind: str = "value"            # value | percent | ignore | dimension | period
    clean: bool = True
    dimension_label: Optional[str] = None   # sub-category carried by the column header (grade, mine type, coalfield…)
    period_explicit: bool = False           # period parsed from the column's own header (vs caption/document fallback)


@dataclass
class FactDraft:
    subject_text: str
    subject_entity_id: Optional[str]
    subject_type: str
    predicate: str
    qualifier: str
    period: Optional[str]
    period_scope: Optional[str]
    value: float
    unit: str
    display_value: str
    original_value: str
    original_unit: Optional[str]
    bbox: Optional[list]
    context: str
    extraction_method: str
    row_index: Optional[int] = None
    col_index: Optional[int] = None
    cell_id: Optional[str] = None
    table_id: Optional[str] = None
    region_id: Optional[str] = None
    is_aggregate: bool = False
    aggregate_level: int = 0
    confidence: dict = field(default_factory=dict)
    parent_entity_id: Optional[str] = None
    dimension_entity_id: Optional[str] = None
    dimension_text: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass
class DocContext:
    document_id: str
    doc_code: str
    doc_type: str
    period: Optional[str]          # e.g. FY2024-25 or 2025-03
    default_unit: str = "million tonnes"
    title: str = ""


def clean_header(h: str) -> str:
    h = (h or "").replace("’", "'").replace("\u2013", "-")
    h = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", h)      # 'Project- ed' -> 'Projected' (but keep 'Jan- Mar')
    h = re.sub(r"(\d{4})-\s+(\d{2})\b", r"\1-\2", h)         # '2023- 24' -> '2023-24'
    return " ".join(h.split())


def parse_header_period(header: str, caption: str, doc_ctx: DocContext) -> Optional[Period]:
    """Combine group headers like 'Production during Mar' + 'FY 25' into a single period."""
    h = clean_header(header)
    cp = parse_period(caption or "")
    if cp and cp.scope in ("ytd", "range") and not MONTH_IN_HEADER_RE.search(h):
        hp = parse_period(h)
        if hp is None or (hp.scope == "fy" and hp.fy == cp.fy) or hp.scope == "year":
            return cp
    fy = None
    ms4 = list(re.finditer(r"\bfy\s*'?-?\s*(20\d{2})\s*-\s*(\d{2})\b", h, re.I))
    ms = list(re.finditer(r"\bfy\s*'?-?\s*(\d{2,4})\b", h, re.I))
    if ms4:
        fy = fy_key(int(ms4[-1].group(1)))
    elif ms:
        m = ms[-1]
        y = int(m.group(1)); y = y + 2000 if y < 100 else y
        fy = fy_key(y - 1)
    m2 = re.search(r"\b(20\d{2})\s*-\s*(\d{2})\b", h)
    if m2 and not fy:
        fy = fy_key(int(m2.group(1)))
    mm = MONTH_IN_HEADER_RE.search(h)
    if re.search(r"\bannual\b", h, re.I) and not re.search(r"\b(upto|up to|till)\b", h, re.I):
        fyk = fy
        if not fyk and mm and mm.group(2).lower()[:3] in MONTHS:
            mon = MONTHS[mm.group(2).lower()[:3]]
            yy = mm.group(3)
            if yy:
                y = int(yy); y = y + 2000 if y < 100 else y
                fyk = fy_for_month(y, mon)
        if not fyk:
            p0 = parse_period(doc_ctx.period or "")
            fyk = p0.fy if p0 else None
        if fyk:
            return Period(fyk, "fy", fyk, fy=fyk, year=int(fyk[2:6]) + 1)
    if mm and mm.group(2).lower()[:3] in MONTHS:
        kind, mon, yy = mm.group(1).lower(), MONTHS[mm.group(2).lower()[:3]], mm.group(3)
        if yy:
            y = int(yy); y = y + 2000 if y < 100 else y
        elif fy:
            sy = int(fy[2:6]); y = sy if mon >= 4 else sy + 1
        else:
            p = parse_period(doc_ctx.period or "")
            y = p.year if p else None
            if y and p and p.scope == "fy":
                sy = int(p.fy[2:6]); y = sy if mon >= 4 else sy + 1
        if y:
            if kind in ("during", "for"):
                return Period(f"{y:04d}-{mon:02d}", "month", f"{mm.group(2).title()[:3]}'{y % 100:02d}", fy=fy_for_month(y, mon), year=y)
            fyk = fy_for_month(y, mon)
            if mon == 3:
                return Period(fyk, "fy", fyk, fy=fyk, year=y)
            return Period(f"{fyk}:ytd:{y:04d}-{mon:02d}", "ytd", f"{fyk} (upto {mm.group(2).title()[:3]}'{y % 100:02d})", fy=fyk, year=y)
    p = parse_period(h)
    if p and (p.scope != "year" or re.search(r"\b(19|20)\d{2}\b", h)):
        return p
    m3 = re.search(r"\b(1[5-9]|2\d|30)\s*$", h)
    if m3 and not fy and not mm and re.search(r"[A-Za-z]{3,}", h):
        y = 2000 + int(m3.group(1))
        return Period(f"Y{y}", "year", str(y), fy=None, year=y)
    # 'as on' captions (resources)
    cp = parse_period(caption or "")
    if cp and cp.scope == "asof":
        return cp
    if p:
        return p
    return None


def lead_in_sentence(intro: str) -> str:
    """Last sentence of the paragraph printed just above a table, when it introduces the table."""
    t = " ".join((intro or "").split())
    if not t:
        return ""
    lines_ = [x.strip() for x in (intro or "").split("\n") if x.strip()]
    for i in range(len(lines_) - 1, max(-1, len(lines_) - 4), -1):
        if re.match(r"^(table|statement|annexure|chart|exhibit)\s*[-–:]?\s*[\dIVX]+", lines_[i], re.I):
            return " ".join(lines_[i:])[:260]          # 'Table 3.11 : Company Wise Production of Coal in last Three Years'
    parts = re.split(r"(?<=[.:])\s+(?=[A-Z0-9(])", t)
    last = parts[-1].strip() if parts else ""
    if len(last) < 15 and len(parts) >= 2:
        last = (parts[-2] + " " + last).strip()
    if len(last) > 260:
        last = last[-260:]
    if re.search(r"(below|as under|as follows|given|table|status|details?|is|are)\s*[:\-–]*\s*$", last, re.I) or last.endswith(":"):
        return last
    if re.match(r"^(table|statement|annexure|chart|exhibit)\s*[-–:]?\s*[\dIVX]+(\.\d+)?\s*[a-z]?\s*[:\-–.]", last, re.I) or re.match(r"^\d{1,2}(\.\d{1,2})?\.?\s+[A-Z][^.]{5,120}$", last):
        return last                                    # 'Table 3.11 : Company Wise Production of Coal in last Three Years'
    return ""


ASOF_RE = re.compile(r"as\s*on\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4})", re.I)


def classify_columns(headers: list[str], rows: list[list[str]], caption: str, unit_hint: str, doc_ctx: DocContext,
                     intro: str = "", page_text: str = "") -> tuple[list[ColumnSpec], Optional[int], Optional[int], Optional[str]]:
    """Return column specs, entity column index, parent (subsidiary) column index, and default table period label."""
    headers = [clean_header(h) for h in headers]
    n = len(headers)
    specs: list[ColumnSpec] = []
    cap = clean_header(caption or "")
    lead = lead_in_sentence(intro)
    if lead and (parse_period(cap) is None or len(cap) < 25):
        cap = (cap + " " + clean_header(lead)).strip()      # e.g. 'Sector-wise off-take from CIL during Jan'24-Nov'24 ... as below:'
    if re.search(r"resource|reserve|inventory", (cap + " " + " ".join(headers)).lower()) and not ASOF_RE.search(cap):
        # inventory tables are dated 'as on 1 April YYYY' once per page/section, not in every caption
        m_asof = ASOF_RE.search(page_text or "") or ASOF_RE.search(intro or "")
        if m_asof:
            cap = (cap + " " + m_asof.group(0)).strip()
    cap = _effective_caption(cap, headers, rows)
    cap_l = cap.lower()
    topic_l = (cap + " " + " ".join(h or "" for h in headers)).lower()
    safety = bool(SAFETY_RE.search(topic_l))
    rakes = bool(re.search(r"\brakes?\b", topic_l))
    # table-level predicate from caption
    table_pred = None
    for pred, rx in PREDICATE_PATTERNS:
        if rx.search(cap_l):
            table_pred = pred; break
    prefix = ""
    cap_hits = [pf for pf, rx in TABLE_TOPIC_PREFIX if rx.search(cap_l)]
    if len(cap_hits) == 1 and not (cap_hits[0] == "lignite_" and re.search(r"\bcoal\b", cap_l)):
        prefix = cap_hits[0]                                  # 'Lignite production by company' → every column is lignite
    if not prefix:
        hdr_hits = {pf for h in headers for pf, rx in TABLE_TOPIC_PREFIX if rx.search((h or "").lower())}
        if len(hdr_hits) == 1 and not any(re.search(r"\braw\s+coal\b|\btotal\s+coal\b|\ball\s+coal\b", (h or "").lower()) for h in headers):
            prefix = next(iter(hdr_hits))                     # only 'Coking Coal Production' group headers → whole table is coking
    table_prefix = prefix
    # dominant predicate named anywhere in the header row ('Projected Production', 'Dispatch-CIL'): fallback for
    # neighbouring 'Actual' / 'FY 2024-25' columns of the same table
    dominant_pred = None
    if table_pred is None:
        for h in headers:
            for pred, rx in PREDICATE_PATTERNS:
                if rx.search((h or "").lower()):
                    dominant_pred = pred; break
            if dominant_pred:
                break
    # unit
    t_unit, t_mult, t_orig = normalize_unit(unit_hint) if unit_hint else (None, 1.0, None)
    if not t_unit:
        m = re.search(r"\((?:in\s*)?(million tonnes?|lakh tonnes?|tonnes?|m\.?\s*cum|mt|rs\.? crore|crore|mw)\)", cap_l)
        if m:
            t_unit, t_mult, t_orig = normalize_unit(m.group(1))
    # entity columns: prefer header keywords, else first text column (skipping serial no.)
    text_ratio = []
    for c in range(n):
        vals = [r[c] for r in rows if c < len(r) and r[c]]
        nums = sum(1 for v in vals if parse_number(v.strip("▲▼ ").rstrip("%")) is not None)
        text_ratio.append(1 - nums / max(len(vals), 1) if vals else 0)
    ent_cols = [c for c in range(n) if ENTITY_COL_RE.search(headers[c] or "") and text_ratio[c] > 0.5 and not SERIAL_RE.match(headers[c] or "")]
    if not ent_cols:
        ent_cols = [c for c in range(n) if text_ratio[c] > 0.6 and not SERIAL_RE.match(headers[c] or "x")]
    entity_col = ent_cols[0] if ent_cols else None
    parent_col = None
    if len(ent_cols) >= 2:
        # subsidiary + mines → subject = finer level (mine), parent = coarser
        hs = [(headers[c] or "").lower() for c in ent_cols]
        if any("mine" in h for h in hs):
            entity_col = next(c for c in ent_cols if "mine" in (headers[c] or "").lower())
            parent_col = next((c for c in ent_cols if c != entity_col and re.search(r"subs|company", (headers[c] or "").lower())), None)
    cap_period = parse_period(cap) if cap else None
    doc_period = parse_period(doc_ctx.period) if doc_ctx.period else None

    unit_col = next((c for c in range(n) if re.match(r"^\s*units?\s*$", headers[c] or "", re.I)), None)
    for c in range(n):
        h = (headers[c] or "").strip()
        spec = ColumnSpec(index=c, header=h)
        if c == entity_col or c == parent_col or c == unit_col or SERIAL_RE.match(h) or (not h and text_ratio[c] > 0.8):
            spec.kind = "ignore"; specs.append(spec); continue
        _vals = [r[c].strip() for r in rows if c < len(r) and r[c] and r[c].strip()]
        if _vals and sum(1 for v in _vals if re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{4}", v)) >= 0.5 * len(_vals):
            spec.kind = "period"; specs.append(spec); continue      # 'As on 01-04-2024' column → per-row as-on period
        if _vals and sum(1 for v in _vals if re.fullmatch(r"\d{1,4}\s*-\s*\d{1,4}|>\s*\d+|<\s*\d+|\d+\s*&\s*above", v)) >= 0.5 * len(_vals):
            spec.kind = "dimension"; specs.append(spec); continue   # depth ranges such as '0-300' → fact dimension
        if re.search(r"\bdepth\b", h, re.I) and not re.search(r"resource|reserve|tonne|mt", h, re.I):
            spec.kind = "dimension"; specs.append(spec); continue
        hl = h.lower()
        pred = None
        prefix = table_prefix
        if re.search(r"\braw\s+coal\b|\ball\s+coal\b|\btotal\s+coal\b", hl):
            prefix = ""                                    # 'Raw Coal Quantity' next to 'Lignite Quantity' = total coal
        for pf, rx in TABLE_TOPIC_PREFIX:
            if rx.search(hl):
                prefix = pf; break                         # '2022-23 Coking' / 'Non-Coking' / 'Total' columns of one table
        col_vals = [r[c] for r in rows if c < len(r) and r[c] and r[c].strip()]
        arrow_col = bool(col_vals) and sum(1 for v in col_vals if v.strip()[:1] in "▲▼\uf035\uf036" or "%" in v) >= 0.5 * len(col_vals)
        if arrow_col and not (PCT_ACH_RE.search(hl) or PCT_GROWTH_RE.search(hl) or PCT_SHARE_RE.search(hl)):
            hl = (hl + " growth (%)").strip()             # data-driven: ▲/▼-marked columns are growth columns
        if safety:
            for p_, rx in SAFETY_PREDS:
                if rx.search(hl) or rx.search(cap_l):
                    pred = p_; break
            if pred is None:
                spec.kind = "ignore"; spec.clean = False; specs.append(spec); continue
            spec.predicate = pred; spec.kind = "value"; spec.unit = "count" if not pred.endswith("_rate") else "rate"; spec.unit_mult = 1.0
            spec.unit_original = "count" if not pred.endswith("_rate") else "per million tonnes"
            hp = parse_header_period(h, cap, doc_ctx); spec.period_explicit = hp is not None
            spec.period = hp or cap_period or doc_period
            specs.append(spec); continue
        if rakes:
            spec.predicate = "rake_loading" if not (PCT_ACH_RE.search(hl) or PCT_GROWTH_RE.search(hl) or "%" in hl) else None
            if spec.predicate is None:
                spec.kind = "ignore"; specs.append(spec); continue
            spec.kind = "value"; spec.unit = "rakes_per_day"; spec.unit_mult = 1.0; spec.unit_original = "rakes/day"
            spec.qualifier = "target" if re.search(r"plan|target", hl) else "actual"
            hp = parse_header_period(h, cap, doc_ctx); spec.period_explicit = hp is not None
            spec.period = hp or cap_period or doc_period
            specs.append(spec); continue
        for p, rx in PREDICATE_PATTERNS:
            if rx.search(hl):
                pred = p; break
        if PCT_ACH_RE.search(hl) or PCT_GROWTH_RE.search(hl) or PCT_SHARE_RE.search(hl) or "%" in hl:
            base = prefix + (pred or table_pred or dominant_pred or "production")
            if PCT_GROWTH_RE.search(hl) and not PCT_ACH_RE.search(hl):
                spec.predicate = f"{base}_growth_pct"
            elif PCT_ACH_RE.search(hl) and not PCT_GROWTH_RE.search(hl):
                spec.predicate = f"{base}_achievement_pct"
            elif PCT_SHARE_RE.search(hl):
                spec.predicate = f"{base}_share_pct"
            else:
                spec.predicate = f"{base}_ambiguous_pct"; spec.clean = False
            spec.kind = "percent"; spec.unit = "percent"; spec.unit_mult = 1.0
            hp = parse_header_period(h, cap, doc_ctx); spec.period_explicit = hp is not None
            spec.period = hp or cap_period or doc_period
            specs.append(spec); continue
        if pred is None:
            pred = table_pred or dominant_pred
        if pred is None:
            # generic numeric column in a table with a known topic, e.g. 'Coking Prime' under a resources caption
            if re.search(r"resource|reserve|inventory", cap_l):
                pred = "resources_total"
            elif YEAR_ROW_RE.match((rows[0][entity_col] if entity_col is not None and rows and entity_col < len(rows[0]) else "").strip()) and (ACTUAL_RE.search(hl) or TARGET_RE.search(hl)):
                pred = "production" if "prod" in cap_l else None
        if pred is None:
            spec.kind = "ignore"; spec.clean = False; specs.append(spec); continue
        if TARGET_RE.search(hl):
            spec.qualifier = "target"
        elif PROJECTED_RE.search(hl):
            spec.qualifier = "projected"
        elif "provisional" in cap_l or "(p)" in hl or "(prov" in hl or "provisional" in (doc_ctx.title or "").lower():
            spec.qualifier = "provisional"
        else:
            spec.qualifier = "actual"
        # unit: header → table → doc default
        if pred.endswith("_ratio"):
            spec.unit, spec.unit_mult, spec.unit_original = "ratio", 1.0, "ratio"
            hp = parse_header_period(h, cap, doc_ctx); spec.period_explicit = hp is not None
            spec.period = hp or cap_period or doc_period
            spec.predicate = pred
            specs.append(spec); continue
        m = re.search(r"\(([^)]*(?:mt|tonne|cum|crore|mw|metre|nos)[^)]*)\)", hl)
        u, mult, orig = normalize_unit(m.group(1)) if m else (None, 1.0, None)
        if not u and pred == "overburden":
            u, mult, orig = "cubic_metres", 1e6, "M.Cum (overburden is reported in million cubic metres)"
        if not u:
            u, mult, orig = t_unit, t_mult, t_orig
        if not u:
            if pred == "overburden":
                u, mult, orig = "cubic_metres", 1e6, "M.Cum (assumed)"
            else:
                u, mult, orig = normalize_unit(doc_ctx.default_unit)
                orig = f"{doc_ctx.default_unit} (document default)"
                spec.clean = False
        spec.unit, spec.unit_mult, spec.unit_original = u, mult, orig
        hp = parse_header_period(h, cap, doc_ctx); spec.period_explicit = hp is not None
        spec.period = hp or cap_period or doc_period
        if spec.period is None:
            spec.clean = False
        spec.predicate = prefix + pred if prefix else pred
        specs.append(spec)
    # group titles such as 'Company in 2023-24 and 2024-25' repeated over every column: assign the listed fiscal
    # years to the non-empty value columns in order
    _assign_listed_fys(specs, rows)
    # infer missing/partial periods from neighbours (wrapped '2023- / 24')
    _infer_periods(specs)
    _propagate_block_periods(specs)
    _fix_percent_periods(specs)
    _label_subcategory_columns(specs)
    # chart-legend pseudo tables: the only label column sits to the right of the numbers
    numeric_cols = [sp.index for sp in specs if sp.kind not in ("ignore", "dimension")]
    if entity_col is not None and numeric_cols and entity_col > max(numeric_cols) and parent_col is None:
        return specs, None, None, None
    return specs, entity_col, parent_col, (cap_period.key if cap_period else None)


UPTO_HDR_RE = re.compile(r"\b(?:upto|up to|till)\s*'?\s*([A-Za-z]{3,9})\.?\s*'?\s*(\d{2,4})?\)?", re.I)


def _effective_caption(cap: str, headers: list[str], rows: list[list[str]]) -> str:
    """Propagate an 'upto <month>' (year-to-date) qualifier that sits in a header cell instead of the caption.

    Only done when the table refers to a single fiscal year (otherwise a YTD column header must not
    contaminate the full-year columns next to it)."""
    cp = parse_period(cap) if cap else None
    if cp and cp.scope in ("ytd", "range", "asof", "month"):
        return cap
    fys = set()
    for h in headers:
        hl = (h or "").lower()
        for m in re.finditer(r"\b(20\d{2})\s*-\s*(\d{2})\b", hl):
            fys.add(m.group(1))
        for m in re.finditer(r"\bfy\s*'?-?\s*(\d{2})\b", hl):
            fys.add(str(2000 + int(m.group(1)) - 1))
    if len(fys) > 1:
        return cap
    for h in headers:
        m = UPTO_HDR_RE.search(h or "")
        if m and m.group(1)[:3].lower() in MONTHS:
            frag = m.group(0)
            if not m.group(2) and fys:
                # 'upto Dec' without a year: attach the table's fiscal year so the parser can place the month
                frag = f"{next(iter(fys))}-{(int(next(iter(fys))) + 1) % 100:02d} ({frag})"
            return (cap + " " + frag).strip()
    return cap


def scope_entity_from_caption(caption: str, headers: list[str], resolver: Resolver):
    """Company scope for sector/type tables, e.g. 'Sector wise Raw Coal Dispatch-CIL' → CIL; default India."""
    txt = clean_header(caption or "") + " " + " ".join(headers or [])
    txt = re.sub(r"\bCoal India Limited\b", "CIL", txt, flags=re.I)
    txt = re.sub(r"\bSingareni Collieries\b[^,.;]*", "SCCL", txt, flags=re.I)
    found = []
    for name in ("SCCL", "NLCIL", "ECL", "BCCL", "CCL", "NCL", "WCL", "SECL", "MCL", "NEC", "CIL"):
        if re.search(rf"(?<![A-Za-z]){name}(?![A-Za-z])", txt):
            found.append(name)
    if re.search(r"\b(all india|india'?s|country)\b", txt, re.I):
        found.append("India")
    if len(found) == 1 or (found and all(f == found[0] for f in found)):
        r = resolver.resolve(found[0], create=False)
        if r.entity is not None:
            return r.entity
    return resolver.resolve("India", create=False).entity


def _fix_percent_periods(specs: list[ColumnSpec]):
    """Achievement % = actual / target of the nearest actual column to the left; growth % belongs to the later
    of the two value columns just before it (current vs previous year)."""
    for i, sp in enumerate(specs):
        if sp.kind != "percent":
            continue
        left = [x for x in specs[:i] if x.kind == "value" and x.period is not None]
        if not left:
            continue
        if "achievement" in (sp.predicate or ""):
            cands = [x for x in left if x.qualifier in ("actual", "provisional")] or left
            best = cands[-1]
        else:
            cands = left[-2:]
            best = max(cands, key=lambda x: ((x.period.year or 0), len(x.period.key)))
        own_month = MONTH_IN_HEADER_RE.search(sp.header or "") or re.search(r"\bfy\s*'?\d{2}|\b20\d{2}\s*-\s*\d{2}", sp.header or "", re.I)
        if sp.period is None or sp.period.scope == "year" or not own_month or (best.period.year or 0) > (sp.period.year or 0):
            sp.period = best.period


def _assign_listed_fys(specs: list[ColumnSpec], rows: list[list[str]]):
    vals = [sp for sp in specs if sp.kind == "value"]
    if len(vals) < 2:
        return
    lists = []
    for sp in vals:
        clause = re.split(r"\.\s+(?=[A-Z])", sp.header)[-1]          # drop stray sentence fragments before the title
        fys = re.findall(r"\b(20\d{2})\s*-\s*\d{2}\b", clause)
        lists.append(tuple(dict.fromkeys(fys)))
    if len(set(lists)) != 1 or len(lists[0]) < 2:
        return
    fys = lists[0]                                                   # keep the printed order (current year may come first)
    nonempty = [sp for sp in vals if any(sp.index < len(r) and r[sp.index].strip() for r in rows)]
    if len(nonempty) != len(fys):
        return
    mm = re.search(r"\b(?:upto|up to|till)\s+([A-Za-z]{3,9})\b", vals[0].header, re.I)
    mon = MONTHS.get(mm.group(1).lower()[:3]) if mm else None
    for sp, sy in zip(nonempty, fys):
        sy = int(sy)
        if mon and mon != 3:
            y = sy if mon >= 4 else sy + 1
            sp.period = Period(f"{fy_key(sy)}:ytd:{y:04d}-{mon:02d}", "ytd", f"{fy_key(sy)} (upto {mm.group(1).title()[:3]}'{y % 100:02d})", fy=fy_key(sy), year=y)
        else:
            sp.period = Period(fy_key(sy), "fy", fy_key(sy), fy=fy_key(sy), year=sy + 1)
        sp.period_explicit = True
    for sp in vals:
        if sp not in nonempty:
            sp.kind = "ignore"


def _propagate_block_periods(specs: list[ColumnSpec]):
    """'2022-23' printed once over a 3-column block (Open Cast | Under Ground | Total) but attached to one column:
    when period-bearing columns are regularly spaced, every column of a block shares that block's period."""
    V = [sp for sp in specs if sp.kind in ("value", "percent")]
    P = [j for j, sp in enumerate(V) if sp.period is not None and sp.period_explicit]
    if len(P) < 2 or len(P) == len(V):
        return
    k = P[1] - P[0]
    if k < 2 or any(b - a != k for a, b in zip(P, P[1:])):
        return
    totals = [j for j, sp in enumerate(V) if re.search(r"\btotal\b", sp.header, re.I)]
    block_start = (totals[0] - k + 1) if totals and totals[0] >= P[0] - k + 1 else P[0]
    for j, sp in enumerate(V):
        if sp.period_explicit:
            continue
        b = (j - block_start) // k if j >= block_start else 0
        if 0 <= b < len(P) and abs(j - P[b]) < k:
            sp.period = V[P[b]].period; sp.period_explicit = True


def _label_subcategory_columns(specs: list[ColumnSpec]):
    """Several value columns with the same predicate/period/qualifier (grade-wise, coalfield-wise…) differ only by a
    sub-category in the header: keep the 'Total' column as the plain fact, tag the others with a dimension label."""
    groups: dict[tuple, list[ColumnSpec]] = {}
    for sp in specs:
        if sp.kind == "value" and sp.predicate and sp.period is not None:
            groups.setdefault((sp.predicate, sp.period.key, sp.qualifier, sp.unit), []).append(sp)
    for key, cols in groups.items():
        if len(cols) < 2:
            continue
        common = set.intersection(*[set(re.findall(r"[A-Za-z][A-Za-z\-]+", sp.header.lower())) for sp in cols]) if len(cols) > 1 else set()
        for sp in cols:
            if re.search(r"\btotal\b|\ball\b", sp.header, re.I):
                continue
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-/]*", sp.header) if w.lower() not in common and not re.fullmatch(r"(?i)fy|quantity|qty|production|prod|despatch|dispatch|actual|target|in|mt|of|the|and", w)]
            label = " ".join(words).strip()
            label = re.sub(r"\b(19|20)\d{2}\s*-\s*\d{2}\b|\(\d+\)", "", label).strip(" -")
            if label:
                sp.dimension_label = label[:40]


def _infer_periods(specs: list[ColumnSpec]):
    known = [s.period for s in specs if s.period]
    for s in specs:
        if s.kind == "ignore" or s.period is not None:
            continue
        m = re.match(r"^\s*(\d{2})\b", s.header)
        if m and known:
            frag = int(m.group(1))
            for k in known:
                if k.fy:
                    sy = int(k.fy[2:6])
                    for cand in (sy - 1, sy - 2, sy + 1):
                        if (cand + 1) % 100 == frag:
                            s.period = Period(fy_key(cand), "fy", fy_key(cand), fy=fy_key(cand), year=cand + 1)
                            break
                if s.period:
                    break
        if s.period is None:
            nxt = next((k for k in known), None)
            if nxt is not None and re.search(r"actual|target", s.header, re.I):
                s.period = nxt; s.clean = False


def extract_cell_value(text: str) -> tuple[Optional[float], str, bool]:
    """Return (number, original_text, clean). Handles '▼ 1.45' (negative), '97.48%', '1.39 (till Nov’24)'."""
    t = (text or "").strip()
    if not t:
        return None, t, True
    neg = "▼" in t
    core = re.sub(r"[▲▼]", "", t).strip()
    v = parse_number(core.rstrip("%"))
    clean = True
    if v is None:
        toks = re.findall(r"[-+]?\(?\d[\d,]*(?:\.\d+)?\)?%?", core)
        nums = [parse_number(x.rstrip("%")) for x in toks]
        nums = [x for x in nums if x is not None]
        if not nums:
            return None, t, True
        v = nums[-1] if re.search(r"[A-Za-z]{3,}", core.split(toks[-1])[0]) else nums[0]
        clean = False
    if neg and v > 0:
        v = -v
    return v, t, clean


def _repeat_period(headers: list[str]) -> Optional[int]:
    """Length p of a header block repeated across the table ('State | As on | … | State | As on | …'), else None."""
    hs = [clean_header(h).lower() for h in headers]
    n = len(hs)
    for p_ in range(3, n // 2 + 1):
        if n % p_ == 0 and all(hs[i] == hs[i % p_] for i in range(n)) and any(hs[i] for i in range(p_)):
            return p_
    return None


def facts_from_table(headers, rows, cells_by_pos, caption, unit_hint, doc_ctx: DocContext, resolver: Resolver,
                     table_conf: float, region_conf: float, table_id: str, region_id: str, intro: str = "", page_text: str = "") -> list[FactDraft]:
    if re.match(r"^\s*(chart|fig\.?|figure|graph)\b", caption or "", re.I):
        return []                                   # chart axis/data labels are marked as figures, not mined as facts
    p_ = _repeat_period(headers)
    if p_:
        # side-by-side halves of one long table printed next to each other: process block by block
        out_all: list[FactDraft] = []
        for b in range(len(headers) // p_):
            cols = list(range(b * p_, (b + 1) * p_))
            sub_rows = [[(r[c] if c < len(r) else "") for c in cols] for r in rows]
            sub_cells = {(ri, c - b * p_): v for (ri, c), v in cells_by_pos.items() if b * p_ <= c < (b + 1) * p_}
            for d in facts_from_table(headers[b * p_:(b + 1) * p_], sub_rows, sub_cells, caption, unit_hint, doc_ctx, resolver, table_conf, region_conf,
                                      table_id, region_id, intro=intro, page_text=page_text):
                d.col_index = (d.col_index or 0) + b * p_
                out_all.append(d)
        return out_all
    specs, entity_col, parent_col, table_period = classify_columns(headers, rows, caption, unit_hint, doc_ctx, intro=intro, page_text=page_text)
    lead = lead_in_sentence(intro)
    if lead and (parse_period(clean_header(caption or "")) is None or len(caption or "") < 25):
        caption = ((caption or "") + " " + lead).strip()
    if entity_col is None:
        return []
    out: list[FactDraft] = []
    value_specs = [s for s in specs if s.kind not in ("ignore", "dimension", "period") and s.predicate]
    dim_cols = [s.index for s in specs if s.kind == "dimension"]
    period_col = next((s.index for s in specs if s.kind == "period"), None)
    if not value_specs:
        return []
    if period_col is not None:
        rows = _fill_merged_labels(rows, entity_col, period_col)
    parent_cache: dict[str, object] = {}
    unit_col = next((c for c in range(len(headers)) if re.match(r"^\s*units?\s*$", headers[c] or "", re.I)), None)
    sector_table = _row_entity_type(headers[entity_col], "", caption) == "sector"
    if sector_table and not re.search(r"sector", headers[entity_col] or "", re.I):
        # caption mentions a sector but the rows may still be companies ('Company-wise rakes for power sector')
        labels = [(r[entity_col] if entity_col < len(r) else "").strip() for r in rows]
        labels = [l for l in labels if l and not is_aggregate_label(l)][:6]
        hits = sum(1 for l in labels if (resolver.resolve(l, expected_type="sector", create=False).entity or Entity()).entity_type == "sector")
        sector_table = bool(labels) and hits >= max(1, len(labels) // 2)
    scope_ent = scope_entity_from_caption(caption, headers, resolver) if sector_table else None
    if sector_table:
        for r in rows:
            lab = (r[entity_col] if entity_col < len(r) else "").strip()
            r_ = resolver.resolve(lab, create=False) if lab and not is_aggregate_label(lab) else None
            if r_ is not None and r_.entity is not None and r_.method == "exact" and r_.entity.entity_type in ("company", "subsidiary"):
                scope_ent = r_.entity                     # a total row labelled with the producer ('SCCL') names the scope
                break
    # company-wise tables restricted to one consuming sector ('rakes for power sector', 'despatch to power') → dimension
    fixed_dim = None
    if not sector_table:
        mcap = re.search(r"\b(?:for|to)\s+(power|steel|cement|sponge iron|fertili[sz]er|non[- ]?regulated)\s+(?:sector|utilities|plants|houses)\b", (caption or ""), re.I)
        if mcap:
            fixed_dim = resolver.resolve(mcap.group(1).title(), expected_type="sector", create=False).entity
    col_entity: dict[int, object] = {}
    if any(YEAR_ROW_RE.match((r[entity_col] if entity_col < len(r) else "").strip()) or MONTH_ROW_RE.match((r[entity_col] if entity_col < len(r) else "").strip()) for r in rows[:6]):
        for sp in value_specs:
            name = re.sub(r"all\s+india\s*\(month\)", " ", sp.header, flags=re.I)
            name = HEADER_NOISE_RE.sub(" ", re.sub(r"\b(19|20)\d{2}\s*-\s*\d{2}\b|\bfy\s*'?\d{2,4}\b", " ", name, flags=re.I))
            name = " ".join(name.split()).strip(" -:")
            if 2 <= len(name) <= 40:
                r_ = resolver.resolve(name, create=False)
                if r_.entity is not None and r_.confidence >= 0.95 and r_.entity.entity_type in ("company", "subsidiary", "country", "organization", "state"):
                    col_entity[sp.index] = r_.entity
        if len(col_entity) >= 2:
            # entity-per-column layout: columns whose header names no known producer are skipped (never mis-attributed)
            value_specs = [sp for sp in value_specs if sp.index in col_entity]
            for sp in value_specs:
                sp.dimension_label = None
    # state-wise AND company-wise matrices: the state name (a merged cell) shares the label column with the companies
    # of its block; blocks end at a 'Total' row.  Companies get dimension=state, the block total becomes the state's figure.
    state_by_row: dict[int, object] = {}
    if re.search(r"state\s*wise", (caption or "") + " " + " ".join(headers), re.I) and re.search(r"company|compan", (headers[entity_col] or "") + " " + (caption or ""), re.I):
        state_names = sorted([e for e in resolver.entities.values() if e.entity_type == "state" and not (e.meta or {}).get("auto_created")], key=lambda e: -len(e.canonical_name))
        block: list[int] = []
        block_state = None
        for ri, r in enumerate(rows):
            lab = (r[entity_col] if entity_col < len(r) else "").strip()
            for st in state_names:
                m_st = re.search(rf"(?<![A-Za-z]){re.escape(st.canonical_name)}(?![A-Za-z])", lab, re.I)
                if m_st:
                    block_state = st
                    rows[ri][entity_col] = (lab[:m_st.start()] + lab[m_st.end():]).strip(" -,") or lab
                    break
            block.append(ri)
            if is_aggregate_label(lab):
                if block_state is not None:
                    for i in block:
                        state_by_row[i] = block_state
                block, block_state = [], None
    # label-only rows ('Gondwana Coalfields', 'Washery Grade-IV') are group headers: they qualify the rows below
    group_by_row: dict[int, str] = {}
    cur_group = None
    n_in_group = 0
    for ri, r in enumerate(rows):
        lab = (r[entity_col] if entity_col < len(r) else "").strip()
        has_vals = any(sp.index < len(r) and extract_cell_value(r[sp.index])[0] is not None for sp in value_specs)
        if lab and not has_vals and not is_aggregate_label(lab) and len(lab) >= 4 and len(lab) <= 40 and not YEAR_ROW_RE.match(lab) and not MONTH_ROW_RE.match(lab) and not QUARTER_ROW_RE.match(lab):
            g_ent = resolver.resolve(lab, create=False).entity
            if g_ent is None or g_ent.entity_type not in ("company", "subsidiary", "mine", "state", "country", "organization", "sector"):
                cur_group, n_in_group = (g_ent.canonical_name if g_ent is not None else lab), 0
                continue
        if cur_group and has_vals:
            group_by_row[ri] = cur_group; n_in_group += 1
        if is_aggregate_label(lab):
            cur_group = None
    current_fy: Optional[int] = None
    for ri, row in enumerate(rows):
        label = (row[entity_col] if entity_col < len(row) else "").strip()
        if not label or len(label) > 80:
            continue
        mfy = re.match(r"^(20\d{2})\s*-\s*\d{2,4}", label)
        if mfy:
            current_fy = int(mfy.group(1))
        # skip note / summary rows and numeric-looking labels (chart data labels such as '7.92, 8%')
        if re.match(r"^(no\.? of|mines produced|note|source|\*|figures?)", label, re.I):
            continue
        if len(re.sub(r"[^A-Za-z]", "", label)) < 2 or re.fullmatch(r"[\d.,%\s▲▼-]+", label):
            continue
        if any(re.search(r"\d[\d.,]*,\s*\d+%", (row[c] or "")) for c in range(len(row)) if c != entity_col):
            continue
        if SKIP_LABEL_RE.match(label) and not re.search(r"sector", (headers[entity_col] or "") + " " + (caption or ""), re.I):
            continue
        parent_ent = None
        if parent_col is not None:
            plabel = (row[parent_col] if parent_col < len(row) else "").strip()
            if plabel:
                if plabel not in parent_cache:
                    parent_cache[plabel] = resolver.resolve(plabel, expected_type="subsidiary", create=False).entity
                parent_ent = parent_cache[plabel]
        agg = is_aggregate_label(label)
        if sector_table and scope_ent is not None and not agg and norm_key(label) in {norm_key(scope_ent.canonical_name)} | set(resolver.aliases_of(scope_ent)):
            agg = True                                       # 'SCCL' row at the bottom of SCCL's sector table = its total
        level = 2 if re.match(r"^grand\s+total|^all\s+india", label, re.I) else (1 if agg else 0)
        etype = "mine" if parent_col is not None else _row_entity_type(headers[entity_col], label, caption)
        partial_total = bool(agg and etype == "mine")     # total of the mines listed, not a company figure
        mrow = MONTH_ROW_RE.match(label)
        qrow = QUARTER_ROW_RE.match(label)
        key_period = None
        if period_col is not None and period_col < len(row) and row[period_col].strip():
            key_period = parse_period("as on " + row[period_col].strip())
        if key_period is not None:
            subj_res = resolver.resolve(label, expected_type=etype, parent_hint=parent_ent, create=(not agg and etype in ("mine", "organization", "sector", "company")),
                                        source=f"auto:{doc_ctx.doc_code}") if not agg else _caption_subject(caption, doc_ctx, resolver)
            row_period = key_period
        elif qrow:
            sy = int(qrow.group(1)) if qrow.group(1) else current_fy
            if sy is None:
                continue
            qn = {"1": 1, "2": 2, "3": 3, "4": 4, "f": 1, "s": 2, "t": 3}[qrow.group(2).lower().replace("q", "").strip()[:1]] if not qrow.group(2).lower().startswith(("fo",)) else 4
            if qrow.group(2).lower().startswith("fourth"):
                qn = 4
            m1 = 4 + 3 * (qn - 1); m2 = m1 + 2
            y1 = sy if m1 <= 12 else sy + 1; y2 = sy if m2 <= 12 else sy + 1
            m1 = m1 if m1 <= 12 else m1 - 12; m2 = m2 if m2 <= 12 else m2 - 12
            fyk = fy_key(sy)
            subj_res = _caption_subject(caption, doc_ctx, resolver)
            row_period = Period(f"{fyk}:range:{y1:04d}-{m1:02d}..{y2:04d}-{m2:02d}", "range", f"Q{qn} {fyk}", fy=fyk, year=y2)
        elif YEAR_ROW_RE.match(label) or mrow:
            # period rows (years / months): subject = column entity if the header names one, else the caption's entity
            subj_res = _caption_subject(caption, doc_ctx, resolver)
            if mrow:
                mon = MONTHS[label[:3].lower()]
                yy = mrow.group(len(mrow.groups()))
                fy_ctx = parse_period(caption or "") or parse_period(doc_ctx.period or "")
                if yy:
                    y = int(yy); y = y + 2000 if y < 100 else y
                elif fy_ctx and fy_ctx.fy:
                    sy = int(fy_ctx.fy[2:6]); y = sy if mon >= 4 else sy + 1
                else:
                    y = None
                row_period = Period(f"{y:04d}-{mon:02d}", "month", f"{label[:3].title()}'{y % 100:02d}", fy=fy_for_month(y, mon), year=y) if y else None
                if row_period is None:
                    continue
            else:
                row_period = parse_period(label)
        elif agg and sector_table and scope_ent is not None:
            # 'Total' / 'Total Despatch' of a sector table = the scope's overall figure (no entity lookup needed)
            subj_res = Resolution(scope_ent, 1.0, "scope", [])
            row_period = None
        elif etype == "item" and not agg:
            continue                                         # commodity / mode / grade rows: no producer subject
        else:
            subj_res = resolver.resolve(label, expected_type=etype, parent_hint=parent_ent, create=(not agg and etype in ("mine", "organization", "sector", "company")),
                                        source=f"auto:{doc_ctx.doc_code}")
            row_period = None
        if subj_res.entity is None:
            continue
        ent = subj_res.entity
        row_ent = ent
        dimension_ent = fixed_dim
        if sector_table and scope_ent is not None and ent.entity_type == "sector" and not ent.is_aggregate:
            dimension_ent, ent = ent, scope_ent          # subject = producer scope (India/CIL/SCCL), dimension = consuming sector
        elif sector_table and scope_ent is not None and agg:
            ent = scope_ent                              # 'Total' of a sector table = the scope's overall figure
        row_unit = None
        if unit_col is not None and unit_col < len(row) and row[unit_col].strip():
            ru, rmult, rorig = normalize_unit(row[unit_col])
            if ru:
                row_unit = (ru, rmult, rorig)
        if ri in state_by_row:
            if agg and not re.match(r"^(grand\s+total|all\s+india)", label, re.I):
                ent = state_by_row[ri]                    # block total = the state's own figure
                level, agg = 1, True
            elif not agg:
                dimension_ent = state_by_row[ri]          # company within a state block
        if dimension_ent is not None and not agg:
            level = 0                                    # a sector/dimension row is a component, never an aggregate
        elif row_ent.entity_type in ("company", "country") and row_ent.is_aggregate:
            level = max(level, 2 if row_ent.canonical_name == "India" else 1)
            agg = True
        elif ent.canonical_name == "India" and level < 2 and not agg:
            level = 2
        for spec in value_specs:
            c = spec.index
            if c >= len(row):
                continue
            v, orig, clean = extract_cell_value(row[c])
            if v is None:
                continue
            per = row_period or spec.period
            cell = cells_by_pos.get((ri, c))
            ent_c = col_entity.get(c) if row_period is not None else None
            subj_ent = ent_c if ent_c is not None else ent
            dim_text = dimension_ent.canonical_name if dimension_ent is not None else None
            if dim_text is None and spec.dimension_label:
                dim_text = spec.dimension_label
            if dim_text is None and ri in group_by_row and not agg:
                dim_text = group_by_row[ri]
            if dim_text is None and dim_cols:
                dvals = [row[dc].strip() for dc in dim_cols if dc < len(row) and row[dc].strip()]
                dim_text = " / ".join(dvals) if dvals else None
            row_agg = agg or (ent_c is not None and ent_c.canonical_name == "India")
            row_level = level if ent_c is None else (2 if ent_c.canonical_name == "India" else (1 if ent_c.is_aggregate else 0))
            if spec.kind == "percent":
                value, unit = v, "percent"
                display = f"{v:g}"
            elif row_unit is not None:
                value, unit = v * row_unit[1], row_unit[0]
                display = f"{v:g}"
            else:
                value, unit = v * spec.unit_mult, spec.unit
                display = f"{v:g}"
            conf = {
                "ocr": round(cell["conf"], 3) if cell else 1.0,
                "layout": round(region_conf, 3),
                "structure": round(table_conf * (1.0 if (spec.clean and clean) else 0.8), 3),
                "entity_matching": round(subj_res.confidence, 3),
                "validation": 1.0,
                "source_agreement": 0.8,
            }
            out.append(FactDraft(
                subject_text=(ent_c.canonical_name if ent_c is not None else label), subject_entity_id=subj_ent.entity_id, subject_type=subj_ent.entity_type,
                predicate=spec.predicate, qualifier=spec.qualifier,
                period=per.key if per else None, period_scope=per.scope if per else None,
                value=value, unit=unit, display_value=display, original_value=orig, original_unit=spec.unit_original,
                bbox=cell["bbox"] if cell else None, context=f"{caption} | {spec.header}".strip(" |"),
                extraction_method="table", row_index=ri, col_index=c, cell_id=cell["cell_id"] if cell else None,
                table_id=table_id, region_id=region_id, is_aggregate=row_agg, aggregate_level=row_level, confidence=conf,
                parent_entity_id=parent_ent.entity_id if parent_ent is not None else subj_ent.parent_entity_id,
                dimension_entity_id=dimension_ent.entity_id if dimension_ent is not None else None,
                dimension_text=dim_text,
                meta={"resolution": subj_res.method, "candidates": subj_res.candidates[:3], "header": spec.header,
                      "period_label": per.label if per else None, "row_unit": row_unit[2] if row_unit else None,
                      **({"partial_total": True} if partial_total else {})},
            ))
    _remap_totals(out, resolver)
    return out


def _fill_merged_labels(rows: list[list[str]], entity_col: int, key_col: int) -> list[list[str]]:
    """Tables keyed by a date column repeat the key sequence per entity while the entity label (a merged cell)
    is printed once, on the middle row of its block: copy the block's label to every row of the block."""
    out = [list(r) for r in rows]
    blocks: list[list[int]] = []
    prev = None
    for i, r in enumerate(out):
        k = r[key_col].strip() if key_col < len(r) else ""
        key = tuple(reversed(k.replace(".", "-").replace("/", "-").split("-"))) if k else None
        if key is None:
            blocks.append([i]); prev = None; continue
        if prev is not None and key > prev and blocks:
            blocks[-1].append(i)
        else:
            blocks.append([i])
        prev = key
    for b in blocks:
        labels = [out[i][entity_col].strip() for i in b if entity_col < len(out[i]) and out[i][entity_col].strip()]
        if len(labels) == 1:
            for i in b:
                if entity_col < len(out[i]):
                    out[i][entity_col] = labels[0]
    return out


def _remap_totals(drafts: list[FactDraft], resolver: Resolver) -> None:
    """'Total' in a table listing only CIL subsidiaries means CIL, not all-India."""
    cil = resolver.resolve("CIL", create=False).entity
    india = resolver.resolve("India", create=False).entity
    if cil is None or india is None:
        return
    subjects = {d.subject_entity_id for d in drafts}
    if cil.entity_id in subjects:
        return
    comps = [d for d in drafts if d.subject_entity_id != india.entity_id]
    if not comps:
        return
    ents = {resolver.entities[d.subject_entity_id] for d in comps if d.subject_entity_id in resolver.entities}
    if ents and all(e.entity_type == "subsidiary" and e.parent_entity_id == cil.entity_id for e in ents):
        for d in drafts:
            if d.subject_entity_id == india.entity_id and re.match(r"^(total|grand total)", d.subject_text, re.I):
                d.subject_entity_id = cil.entity_id; d.subject_type = "company"; d.aggregate_level = 1; d.is_aggregate = True
                d.meta["remapped"] = "Total→CIL (all rows are CIL subsidiaries)"


def _row_entity_type(header: str, label: str, caption: str) -> str:
    h = (header or "").lower()
    if re.search(r"\b(product|item|particulars|commodity|category|grade|mode|type of|head)\b", h):
        return "item"                                   # commodities / transport modes: never auto-created as entities
    if re.search(r"state", h):
        return "state"
    if re.search(r"sector", h) or re.search(r"despatch to|dispatch to|sector", (caption or "").lower()):
        return "sector"
    if re.search(r"mine", h):
        return "mine"
    if re.search(r"subs|company|companies", h):
        return "company"
    if re.search(r"depth", h):
        return "depth_range"
    return "organization"


def _caption_subject(caption: str, doc_ctx: DocContext, resolver: Resolver):
    cap = caption or ""
    for name in ("CIL", "SCCL", "NLCIL", "NLC", "ECL", "BCCL", "CCL", "NCL", "WCL", "SECL", "MCL", "NEC", "CMPDI"):
        if re.search(rf"\b{name}\b", cap):
            return resolver.resolve(name, create=False)
    return resolver.resolve("India", create=False)


# --------------------------------------------------------------------------- narrative facts

NARRATIVE_PATTERNS = [
    # India’s coal production increased by 1.59% to 118.54 MT from 116.68 MT during Mar'25 as compared to Mar'24.
    (re.compile(r"(India[’']?s|CIL|SCCL|Captives?/Others)\s+(?:coal\s+)?(production|despatch|dispatch|offtake)\s+(?:has\s+)?(increased|decreased|grew|declined|registered a growth|registered a negative growth)\b.{0,60}?\bto\s+([\d,]+\.?\d*)\s*MT\s+from\s+([\d,]+\.?\d*)\s*MT\s+during\s+([A-Za-z]{3,9}[’']\s?\d{2}|[A-Za-z]{3,9}\s+\d{4})", re.I),
     "growth_to_from"),
    # During 2024-25, actual Raw Coal Production is 1047.52 Million Tonnes (MT) against the Annual production Target of 1080.20 MT
    (re.compile(r"During\s+(20\d{2}-\d{2}),\s*actual\s+Raw\s+Coal\s+(Production|dispatch(?:ed)?)\s+(?:is|was)\s+([\d,]+\.?\d*)\s*(?:Million\s+Tonnes?|MT).{0,80}?(?:Target\s+of\s+([\d,]+\.?\d*)\s*MT)?", re.I),
     "fy_actual_target"),
    # SCCL & Captives/Others registered a growth of 22.08 % & 15.20% by producing 8.91 MT & 23.82 MT
    (re.compile(r"\b(CIL|SCCL|NLCIL|Captives?/Others)\s*&\s*(CIL|SCCL|NLCIL|Captives?/Others)\s+registered\s+a\s+(?:negative\s+)?growth\s+of\s+[\d.]+\s*%\s*&\s*[\d.]+\s*%\s+by\s+producing\s+([\d,]+\.?\d*)\s*MT\s*&\s*([\d,]+\.?\d*)\s*MT", re.I),
     "pair_month_prod"),
    # The Power utilities despatch has increased by 6.25% to 78.46 MT during Mar'25 as compared to 73.84 MT during Mar'24.
    (re.compile(r"(Power\s+utilities|Power\s+sector|CPP|Steel|Cement|Sponge\s+Iron)\s+(despatch|dispatch|supply)\s+has\s+(increased|decreased)\s+by\s+[\d.]+%\s+to\s+([\d,]+\.?\d*)\s*MT\s+during\s+([A-Za-z]{3,9}[’']\s?\d{2})\s+as\s+compared\s+to\s+([\d,]+\.?\d*)\s*MT\s+during\s+([A-Za-z]{3,9}[’']\s?\d{2})", re.I),
     "sector_growth"),
    # SCCL & Captives/Others registered a growth of 22.08 % & 15.20% by producing 8.91 MT & 23.82 MT ... Whereas CIL registered a negative growth of 3.13% by producing 85.81 MT
    (re.compile(r"\b(CIL|SCCL|NLCIL)\s+registered\s+a\s+(?:negative\s+)?growth\s+of\s+[\d.]+\s*%\s+by\s+producing\s+([\d,]+\.?\d*)\s*MT", re.I),
     "company_month_prod"),
    # A total of 3,89,421.34 Mt of geological resources of coal ... as on 01.04.2024 / inventory ... is 389421.34 MT
    (re.compile(r"(?:total\s+of|resources?\s+(?:of\s+coal\s+)?(?:as\s+on|stands?\s+at)\s*[\d.]*\s*(?:is|at)?)\s*([\d,]+\.\d+)\s*(?:Mt|MT|million\s+tonnes?)\b[^.]*?(?:as\s+on\s+(\d{1,2}\.\d{1,2}\.\d{4}))?", re.I),
     "coal_resources"),
    (re.compile(r"inventory\s+of\s+Geological\s+Resources\s+of\s+Indian\s+Coal\s+as\s+on\s+(\d{1,2}\.\d{1,2}\.\d{4})[^.]*?\bis\s+([\d,]+\.\d+)\s*(?:MT|Mt|million\s+tonnes?)", re.I),
     "coal_resources_inv"),
    (re.compile(r"India[’']?s\s+total\s+geological\s+coal\s+resource\s+stands\s+at\s+([\d,]+\.\d+)\s*million\s+tonnes", re.I),
     "coal_resources_plain"),
    # Lignite reserves ... estimated at around 47370.54 million Tonne (as on 01.04.2025)
    (re.compile(r"Lignite\s+reserves?\s+in\s+the\s+country\s+are\s+estimated\s+at\s+around\s+([\d,]+\.?\d*)\s*million\s+Tonnes?\s*\(as\s+on\s+(\d{1,2}\.\d{1,2}\.\d{4})\)", re.I),
     "lignite_resources"),
]


def facts_from_text(text: str, region_bbox, region_id: str, doc_ctx: DocContext, resolver: Resolver, region_conf: float) -> list[FactDraft]:
    out: list[FactDraft] = []
    t = " ".join((text or "").split())
    for rx, kind in NARRATIVE_PATTERNS:
        for m in rx.finditer(t):
            sent = _sentence_around(t, m.start(), m.end())
            try:
                out.extend(_narrative_fact(kind, m, sent, region_bbox, region_id, doc_ctx, resolver, region_conf))
            except Exception:
                continue
    return out


def _sentence_around(t: str, a: int, b: int) -> str:
    s = t.rfind(". ", 0, a); s = 0 if s < 0 else s + 2
    e = t.find(". ", b); e = len(t) if e < 0 else e + 1
    return t[s:e].strip()


def _mk(subj_name, pred, qual, per: Optional[Period], val, unit_raw, sent, bbox, region_id, resolver, region_conf, doc_ctx, extra=None):
    res = resolver.resolve(subj_name, create=False)
    if res.entity is None or per is None:
        return None
    unit, mult, orig = normalize_unit(unit_raw)
    if not unit:
        unit, mult, orig = "tonnes", 1e6, unit_raw
    ent = res.entity
    return FactDraft(subject_text=subj_name, subject_entity_id=ent.entity_id, subject_type=ent.entity_type, predicate=pred, qualifier=qual,
                     period=per.key, period_scope=per.scope, value=val * mult, unit=unit, display_value=f"{val:g}", original_value=f"{val:g} {unit_raw}",
                     original_unit=orig, bbox=bbox, context=sent[:400], extraction_method="narrative", region_id=region_id,
                     is_aggregate=bool(ent.is_aggregate), aggregate_level=2 if ent.canonical_name == "India" else (1 if ent.is_aggregate else 0),
                     confidence={"ocr": round(region_conf, 3), "layout": round(region_conf, 3), "structure": 0.85, "entity_matching": round(res.confidence, 3), "validation": 1.0, "source_agreement": 0.8},
                     parent_entity_id=ent.parent_entity_id, meta={"pattern": extra or pred, "period_label": per.label})


def _narrative_fact(kind, m, sent, bbox, region_id, doc_ctx, resolver, rc) -> list[FactDraft]:
    facts = []
    if kind == "growth_to_from":
        subj = m.group(1); pred = "dispatch" if m.group(2).lower() in ("despatch", "dispatch", "offtake") else "production"
        subj = "India" if "india" in subj.lower() else subj.replace("Captives/Others", "Captive and Others")
        cur = parse_number(m.group(4)); prev = parse_number(m.group(5)); per = parse_period(m.group(6))
        if per:
            facts.append(_mk(subj, pred, "provisional" if "provisional" in doc_ctx.title.lower() else "actual", per, cur, "MT", sent, bbox, region_id, resolver, rc, doc_ctx, kind))
            if per.scope == "month":
                y, mo = per.year - 1, int(per.key[-2:])
                pp = Period(f"{y:04d}-{mo:02d}", "month", f"{per.label[:-2]}{y % 100:02d}", fy=fy_for_month(y, mo), year=y)
                facts.append(_mk(subj, pred, "actual", pp, prev, "MT", sent, bbox, region_id, resolver, rc, doc_ctx, kind + "_prev"))
    elif kind == "fy_actual_target":
        per = parse_period(m.group(1)); pred = "production" if m.group(2).lower().startswith("prod") else "dispatch"
        facts.append(_mk("India", pred, "actual", per, parse_number(m.group(3)), "MT", sent, bbox, region_id, resolver, rc, doc_ctx, kind))
        if m.group(4):
            facts.append(_mk("India", pred, "target", per, parse_number(m.group(4)), "MT", sent, bbox, region_id, resolver, rc, doc_ctx, kind + "_target"))
    elif kind == "sector_growth":
        subj = m.group(1); per = parse_period(m.group(5)); per2 = parse_period(m.group(7))
        facts.append(_mk(subj, "dispatch", "provisional" if "provisional" in doc_ctx.title.lower() else "actual", per, parse_number(m.group(4)), "MT", sent, bbox, region_id, resolver, rc, doc_ctx, kind))
        facts.append(_mk(subj, "dispatch", "actual", per2, parse_number(m.group(6)), "MT", sent, bbox, region_id, resolver, rc, doc_ctx, kind + "_prev"))
    elif kind == "company_month_prod":
        per = parse_period(doc_ctx.period or "")
        if per and per.scope == "month":
            facts.append(_mk(m.group(1), "production", "provisional", per, parse_number(m.group(2)), "MT", sent, bbox, region_id, resolver, rc, doc_ctx, kind))
    elif kind == "pair_month_prod":
        per = parse_period(doc_ctx.period or "")
        if per and per.scope == "month":
            for name, val in ((m.group(1), m.group(3)), (m.group(2), m.group(4))):
                name = "Captive and Others" if "captive" in name.lower() else name
                facts.append(_mk(name, "production", "provisional", per, parse_number(val), "MT", sent, bbox, region_id, resolver, rc, doc_ctx, kind))
    elif kind in ("coal_resources", "coal_resources_inv", "coal_resources_plain"):
        if kind == "coal_resources_inv":
            date, val = m.group(1), parse_number(m.group(2))
        elif kind == "coal_resources":
            val, date = parse_number(m.group(1)), m.group(2)
        else:
            val, date = parse_number(m.group(1)), None
        if not date:
            dm = re.search(r"as\s+on\s+(\d{1,2}\.\d{1,2}\.\d{4})", sent)
            date = dm.group(1) if dm else None
        per = parse_period(f"as on {date}") if date else None
        if val and val > 1000 and per:
            facts.append(_mk("India", "resources_total", "actual", per, val, "million tonnes", sent, bbox, region_id, resolver, rc, doc_ctx, kind))
    elif kind == "lignite_resources":
        per = parse_period(f"as on {m.group(2)}")
        facts.append(_mk("India", "lignite_resources_total", "actual", per, parse_number(m.group(1)), "million tonnes", sent, bbox, region_id, resolver, rc, doc_ctx, kind))
    return [f for f in facts if f is not None]
