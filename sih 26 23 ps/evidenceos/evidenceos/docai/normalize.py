"""Normalization engine (PRD §11.3.7): numbers, units, fiscal periods, dates.

Rule: never discard the original representation — callers store both the
canonical value and the original string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTHS.update({"sept": 9, "june": 6, "july": 7})
FY_START_MONTH = 4  # Indian fiscal year April–March

# canonical units and multipliers to base unit
UNIT_TABLE = {
    # mass -> tonnes
    "tonne": ("tonnes", 1.0), "tonnes": ("tonnes", 1.0), "te": ("tonnes", 1.0), "t": ("tonnes", 1.0),
    "th. tonnes": ("tonnes", 1e3), "thousand tonnes": ("tonnes", 1e3), "'000 tonnes": ("tonnes", 1e3), "000 tonnes": ("tonnes", 1e3),
    "lakh tonnes": ("tonnes", 1e5), "lakh tonne": ("tonnes", 1e5),
    "million tonnes": ("tonnes", 1e6), "million tonne": ("tonnes", 1e6), "mt": ("tonnes", 1e6), "mte": ("tonnes", 1e6),
    "mn t": ("tonnes", 1e6), "mtpa": ("tonnes", 1e6), "mty": ("tonnes", 1e6), "million te": ("tonnes", 1e6),
    "crore tonnes": ("tonnes", 1e7), "billion tonnes": ("tonnes", 1e9), "bt": ("tonnes", 1e9),
    # volume -> cubic metres (overburden)
    "cum": ("cubic_metres", 1.0), "m3": ("cubic_metres", 1.0), "cu.m": ("cubic_metres", 1.0),
    "m.cum": ("cubic_metres", 1e6), "mcum": ("cubic_metres", 1e6), "million cum": ("cubic_metres", 1e6), "mm3": ("cubic_metres", 1e6),
    "lakh cum": ("cubic_metres", 1e5),
    # money -> INR
    "rs. crore": ("inr", 1e7), "rs crore": ("inr", 1e7), "crore": ("inr", 1e7), "₹ crore": ("inr", 1e7), "rs. lakh": ("inr", 1e5),
    # misc
    "%": ("percent", 1.0), "percent": ("percent", 1.0), "per cent": ("percent", 1.0),
    "nos": ("count", 1.0), "no.": ("count", 1.0), "number": ("count", 1.0), "nos.": ("count", 1.0),
    "mw": ("megawatt", 1.0), "km": ("kilometres", 1.0), "ha": ("hectares", 1.0), "m": ("metres", 1.0),
    "lakh metre": ("metres", 1e5), "lakh metres": ("metres", 1e5), "metre": ("metres", 1.0), "metres": ("metres", 1.0),
}

DISPLAY_UNITS = {  # canonical unit -> preferred display (label, divisor)
    "tonnes": ("MT", 1e6),
    "cubic_metres": ("M.Cum", 1e6),
    "inr": ("Rs. crore", 1e7),
    "percent": ("%", 1.0),
    "count": ("nos", 1.0),
}

NUM_RE = re.compile(r"[-+−]?\(?\d[\d,]*(?:\.\d+)?\)?")


def parse_number(s: str) -> Optional[float]:
    """Parse numbers incl. Indian grouping (3,89,421.34), unicode minus, (123) negatives, trailing %."""
    if s is None:
        return None
    t = str(s).strip().replace("\u2212", "-").replace("\u00a0", " ")
    t = re.sub(r"[▲▼↑↓*#]", "", t).strip()
    if not t or t in {"-", "--", "—", "NA", "N.A.", "N/A", "nil", "Nil", "NIL", "..", "…"}:
        return None
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").rstrip("%").strip()
    if not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+", t):
        return None
    try:
        v = float(t.replace(",", ""))
    except ValueError:
        return None
    return -v if neg else v


def is_numeric_token(s: str) -> bool:
    return parse_number(s) is not None


def normalize_unit(raw: Optional[str], context: str = "") -> tuple[Optional[str], float, Optional[str]]:
    """Return (canonical_unit, multiplier_to_base, original_unit_string)."""
    if not raw:
        return None, 1.0, None
    u = raw.strip().strip("()[]").strip().lower()
    u = u.replace("fig. in", "").replace("figures in", "").replace("qty. in", "").replace("in ", "").strip()
    u = u.replace("tonne)", "tonne").replace("tonnes)", "tonnes")
    if u in UNIT_TABLE:
        canon, mult = UNIT_TABLE[u]
        return canon, mult, raw.strip()
    # relaxed matching
    for key in sorted(UNIT_TABLE, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", u):
            canon, mult = UNIT_TABLE[key]
            return canon, mult, raw.strip()
    return None, 1.0, raw.strip()


def to_display(value: Optional[float], unit: Optional[str], decimals: int = 2) -> str:
    if value is None:
        return "—"
    if unit in DISPLAY_UNITS:
        label, div = DISPLAY_UNITS[unit]
        v = value / div
        return f"{v:,.{decimals}f} {label}".rstrip()
    return f"{value:,.{decimals}f} {unit or ''}".strip()


@dataclass
class Period:
    key: str          # FY2024-25 | 2025-03 | asof:2024-04-01 | Y2024
    scope: str        # fy | month | ytd | asof | year
    label: str
    fy: Optional[str] = None      # fiscal year key for grouping
    year: Optional[int] = None    # calendar year of the reference


def fy_key(start_year: int) -> str:
    return f"FY{start_year}-{(start_year + 1) % 100:02d}"


def fy_for_month(year: int, month: int) -> str:
    return fy_key(year if month >= FY_START_MONTH else year - 1)


def _two_digit_year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_period(text: str, doc_period: Optional[str] = None) -> Optional[Period]:
    """Parse fiscal-year / month / as-on expressions found in headers, captions, sentences.

    Examples: 'FY 25', 'FY2024-25', '2024-25', '2024- 25', "Mar'25", 'Upto Mar'25', 'during March 2025',
    'as on 01.04.2024', '2025-26 (upto December 25)', 'Apr-Dec 2024', '2023-24 to 2024-25' (first only)
    """
    if not text:
        return None
    t = " ".join(str(text).replace("\u2019", "'").replace("`", "'").replace("\u2013", "-").replace("\u2014", "-").split())
    tl = t.lower()

    # ISO month key produced by the system itself (document period '2025-03')
    m = re.fullmatch(r"(20\d{2})-(0[1-9]|1[0-2])", tl)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return Period(f"{y:04d}-{mo:02d}", "month", f"{MONTH_NAMES[mo - 1] if 'MONTH_NAMES' in globals() else mo}'{y % 100:02d}", fy=fy_for_month(y, mo), year=y)

    # month range: (Jan-Mar'25), Apr'24-Dec'24, Apr-Dec 2024  -> scope 'range'
    m = re.search(r"\b([a-z]{3,9})\s*'?\s*(\d{2,4})?\s*(?:-|to)\s*([a-z]{3,9})\s*'?\s*(\d{2,4})\b", tl)
    if m and m.group(1)[:3] in MONTHS and m.group(3)[:3] in MONTHS and m.group(1)[:3] != "apr":
        m1, m2 = MONTHS[m.group(1)[:3]], MONTHS[m.group(3)[:3]]
        y2 = _two_digit_year(m.group(4)); y1 = _two_digit_year(m.group(2)) if m.group(2) else (y2 if m1 <= m2 else y2 - 1)
        fy = fy_for_month(y2, m2)
        return Period(f"{fy}:range:{y1:04d}-{m1:02d}..{y2:04d}-{m2:02d}", "range", f"{m.group(1).title()[:3]}'{y1 % 100:02d}-{m.group(3).title()[:3]}'{y2 % 100:02d}", fy=fy, year=y2)

    # as on dd.mm.yyyy / 1st April 2024
    m = re.search(r"as\s*on\s*(\d{1,2})[./-](\d{1,2})[./-](\d{4})", tl)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return Period(f"asof:{y:04d}-{mo:02d}-{d:02d}", "asof", f"as on {d:02d}.{mo:02d}.{y}", fy=fy_for_month(y, mo), year=y)
    m = re.search(r"as\s*on\s*(\d{1,2})(?:st|nd|rd|th)?\s*([a-z]+),?\s*(\d{4})", tl)
    if m and m.group(2)[:3] in MONTHS:
        d, mo, y = int(m.group(1)), MONTHS[m.group(2)[:3]], int(m.group(3))
        return Period(f"asof:{y:04d}-{mo:02d}-{d:02d}", "asof", f"as on {d:02d}.{mo:02d}.{y}", fy=fy_for_month(y, mo), year=y)

    # 2025-26 (upto December 25) / 2024-25 (upto Dec'24)
    m = re.search(r"(20\d{2})\s*-\s*(\d{2})\s*\(?\s*(?:upto|up to|till)\s*([a-z]{3,9})[^\d]*(\d{2,4})?", tl)
    if m:
        sy = int(m.group(1)); mo = MONTHS.get(m.group(3)[:3])
        if mo:
            y = sy if mo >= FY_START_MONTH else sy + 1
            return Period(f"{fy_key(sy)}:ytd:{y:04d}-{mo:02d}", "ytd", f"{fy_key(sy)} (upto {m.group(3).title()}'{y % 100:02d})", fy=fy_key(sy), year=y)

    # upto Mar'25 / Upto Mar 2025 / Apr-Mar 2024-25 / Apr'24 to Mar'25
    m = re.search(r"(?:upto|up to|till|cumulative)\s*([a-z]{3,9})\.?\s*'?\s*(\d{2,4})", tl)
    if m and m.group(1)[:3] in MONTHS:
        mo = MONTHS[m.group(1)[:3]]; y = _two_digit_year(m.group(2))
        fy = fy_for_month(y, mo)
        if mo == 3:   # April–March cumulative = full fiscal year
            return Period(fy, "fy", fy, fy=fy, year=y)
        return Period(f"{fy}:ytd:{y:04d}-{mo:02d}", "ytd", f"{fy} (upto {m.group(1).title()}'{y % 100:02d})", fy=fy, year=y)
    m = re.search(r"apr(?:il)?\s*'?\s*(\d{2,4})?\s*(?:-|to|–)\s*([a-z]{3,9})\s*'?\s*(\d{2,4})", tl)
    if m and m.group(2)[:3] in MONTHS:
        mo = MONTHS[m.group(2)[:3]]; y = _two_digit_year(m.group(3))
        fy = fy_for_month(y, mo)
        if mo == 3:
            return Period(fy, "fy", fy, fy=fy, year=y)
        return Period(f"{fy}:ytd:{y:04d}-{mo:02d}", "ytd", f"{fy} (upto {m.group(2).title()}'{y % 100:02d})", fy=fy, year=y)

    # FY 25 / FY25 / FY-25 / FY 2025 / FY 2024-25
    m = re.search(r"\bfy\s*'?-?\s*(20\d{2})\s*-\s*(\d{2})\b", tl)
    if m:
        sy = int(m.group(1)); return Period(fy_key(sy), "fy", fy_key(sy), fy=fy_key(sy), year=sy + 1)
    m = re.search(r"\bfy\s*'?-?\s*(\d{2,4})\b", tl)
    if m:
        y = _two_digit_year(m.group(1)); sy = y - 1
        return Period(fy_key(sy), "fy", fy_key(sy), fy=fy_key(sy), year=y)

    # month'yy or month yyyy or month-yy (Mar'25, March 2025, Mar-25, Mar,2025)
    m = re.search(r"\b([a-z]{3,9})\.?\s*[',\- ]\s*'?(\d{2}|\d{4})\b", tl)
    if m and m.group(1)[:3] in MONTHS and m.group(1) not in ("during", "upto"):
        mo = MONTHS[m.group(1)[:3]]; y = _two_digit_year(m.group(2))
        return Period(f"{y:04d}-{mo:02d}", "month", f"{m.group(1).title()[:3]}'{y % 100:02d}", fy=fy_for_month(y, mo), year=y)

    # 2024-25 / 2024 - 25 / 2024-2025
    m = re.search(r"\b(20\d{2}|19\d{2})\s*-\s*(\d{2}|\d{4})\b", tl)
    if m:
        sy = int(m.group(1)); ey = int(m.group(2)); ey = ey if ey > 100 else (sy // 100) * 100 + ey
        if ey == sy + 1:
            return Period(fy_key(sy), "fy", fy_key(sy), fy=fy_key(sy), year=ey)
    # calendar year
    m = re.search(r"\b(19\d{2}|20\d{2})\b", tl)
    if m:
        y = int(m.group(1)); return Period(f"Y{y}", "year", str(y), fy=None, year=y)
    if doc_period:
        return parse_period(doc_period)
    return None


def normalize_text(s: str) -> str:
    s = (s or "").lower().replace("\u2019", "'")
    s = re.sub(r"[^a-z0-9&/ ]+", " ", s)
    return " ".join(s.split())
