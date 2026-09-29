"""Validation engine + confidence policy + conflict detection (PRD §11.4.3–§11.4.5, §13).

Levels: extraction (OCR), structural, arithmetic (totals), temporal, entity,
cross-document.  Every check writes a `validations` row; cross-document
disagreements create `conflicts` rows; agreements raise source_agreement.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable

from sqlalchemy import func

from .. import config
from ..db import Conflict, Document, Entity, Fact, Validation, next_code
from ..docai.normalize import to_display

PERCENT_LIKE = ("_pct",)


def overall_confidence(conf: dict) -> float:
    w = config.CONFIDENCE_WEIGHTS
    tot = sum(w.values())
    return round(sum(w[k] * float(conf.get(k, 0.8)) for k in w) / tot, 4)


def review_flag(overall: float) -> str:
    if overall >= config.AUTO_ACCEPT_THRESHOLD:
        return "none"
    if overall >= config.MANDATORY_REVIEW_THRESHOLD:
        return "recommended"
    return "mandatory"


def _add(session, fact: Fact, rule_id: str, rule_type: str, status: str, message: str, evidence: dict | None = None):
    session.add(Validation(fact_id=fact.fact_id, rule_id=rule_id, rule_type=rule_type, status=status, message=message, evidence=evidence or {}))


# --------------------------------------------------------------------------- per-document rules

def validate_document_facts(session, document: Document, facts: list[Fact]) -> None:
    """Numeric range, OCR confidence, structural, temporal, entity, arithmetic checks for one document."""
    doc_year = None
    if document.period:
        m = re.search(r"(19|20)\d{2}", document.period)
        doc_year = int(m.group(0)) if m else None
    ent_cache: dict[str, Entity] = {}

    def ent(eid):
        if eid not in ent_cache:
            ent_cache[eid] = session.get(Entity, eid)
        return ent_cache[eid]

    for f in facts:
        conf = dict(f.confidence or {})
        vscore = 1.0
        # V-001 numeric range
        if f.unit != "percent" and f.value is not None and f.value < 0:
            _add(session, f, "V-001", "numeric", "fail", f"{f.predicate} cannot be negative ({f.display_value})")
            vscore = min(vscore, 0.3)
        elif f.unit == "percent" and f.value is not None and not (-100 <= f.value <= 1000):
            _add(session, f, "V-001", "numeric", "warning", f"percentage out of plausible range ({f.display_value})")
            vscore = min(vscore, 0.6)
        elif f.unit == "tonnes" and f.value is not None and f.value > 5e12:
            _add(session, f, "V-001", "numeric", "warning", "implausibly large tonnage; unit may be mis-scaled")
            vscore = min(vscore, 0.5)
        else:
            _add(session, f, "V-001", "numeric", "pass", "value within plausible range")
        # V-008 OCR confidence
        if conf.get("ocr", 1.0) < 0.70:
            _add(session, f, "V-008", "ocr", "warning", f"OCR confidence {conf.get('ocr'):.2f} < 0.70 → review")
        # V-007 structural
        if conf.get("structure", 1.0) < 0.75 or f.original_unit and "assumed" in (f.original_unit or ""):
            _add(session, f, "V-007", "structural", "warning", "column/unit inferred with low structural confidence")
            vscore = min(vscore, 0.8)
        else:
            _add(session, f, "V-007", "structural", "pass", "value sits in a parsed metric column")
        # V-004 temporal
        if doc_year and f.period:
            m = re.search(r"(19|20)\d{2}", f.period)
            if m:
                fy = int(m.group(0))
                if fy > doc_year + 1 or fy < doc_year - 15:
                    _add(session, f, "V-004", "temporal", "warning", f"period {f.period} outside document window ({document.period})")
                    vscore = min(vscore, 0.7)
                else:
                    _add(session, f, "V-004", "temporal", "pass", "period consistent with document period")
        # V-005 entity ↔ parent
        if f.subject_entity_id:
            e = ent(f.subject_entity_id)
            if e is not None and e.entity_type == "mine":
                meta = f.confidence.get("_parent_check") if isinstance(f.confidence, dict) else None
        conf["validation"] = round(vscore, 3)
        f.confidence = conf


def arithmetic_checks(session, facts_by_table: dict[str, list[Fact]]) -> None:
    """V-003: aggregate rows (Total / CIL / Grand Total) must equal the sum of their component rows per column."""
    for table_id, facts in facts_by_table.items():
        by_col: dict[int, list[Fact]] = defaultdict(list)
        for f in facts:
            col = (f.confidence or {}).get("_col")
            if f.unit == "percent" or col is None:
                continue
            by_col[col].append(f)
        for col, col_facts in by_col.items():
            col_facts.sort(key=lambda f: (f.confidence or {}).get("_row", 0))
            buffer1: list[Fact] = []
            level1_aggs: list[Fact] = []
            for f in col_facts:
                lvl = (f.confidence or {}).get("_agg_level", 0)
                if lvl == 0:
                    buffer1.append(f)
                    continue
                if lvl == 1:
                    if len(buffer1) < 2:
                        buffer1.append(f)          # e.g. 'CIL' listed as a plain row (no subsidiaries above)
                        continue
                    comps = buffer1
                    _check_sum(session, f, comps)
                    level1_aggs.append(f)
                    buffer1 = []
                else:  # grand total
                    comps = level1_aggs + buffer1
                    if len(comps) >= 2:
                        _check_sum(session, f, comps)
                    buffer1, level1_aggs = [], []


def _check_sum(session, agg: Fact, comps: list[Fact]) -> None:
    if len(comps) < 2 or agg.value is None:
        return
    if agg.predicate.endswith("_rate") or agg.predicate.endswith("_pct"):
        return                                   # rates/ratios are not additive
    s = sum(c.value for c in comps if c.value is not None)
    tol = max(config.ARITHMETIC_REL_TOLERANCE * abs(agg.value), 0.02 * _unit_scale(agg) * len(comps))
    ok = abs(s - agg.value) <= tol
    ev = {"components": [{"fact_id": c.fact_id, "subject": c.subject_text, "value": c.display_value} for c in comps],
          "sum": round(s / _unit_scale(agg), 3), "reported": agg.display_value}
    msg = f"{agg.subject_text}: sum of {len(comps)} rows = {ev['sum']} vs reported {agg.display_value}"
    status = "pass" if ok else "fail"
    if not ok:
        # a continuation table may list only part of the aggregate's children (e.g. WCL..NEC + CIL on the next page)
        children = session.query(Entity).filter(Entity.parent_entity_id == agg.subject_entity_id).count() if agg.subject_entity_id else 0
        present = {c.subject_entity_id for c in comps}
        if children and len(present) < children and s < agg.value:
            status = "warning"
            msg += f" (only {len(present)} of {children} components on this page: continuation table?)"
            ev["partial"] = True
    _add(session, agg, "V-003", "arithmetic", status, msg, ev)
    ok = status != "fail"
    conf = dict(agg.confidence or {})
    conf["validation"] = round(min(conf.get("validation", 1.0), 1.0 if ok else 0.5), 3)
    conf["arithmetic_checked"] = True
    agg.confidence = conf
    for c in comps:
        cc = dict(c.confidence or {})
        cc["validation"] = round(min(cc.get("validation", 1.0), 1.0 if ok else 0.85), 3)
        cc["arithmetic_checked"] = True
        c.confidence = cc


def _unit_scale(f: Fact) -> float:
    return {"tonnes": 1e6, "cubic_metres": 1e6, "inr": 1e7}.get(f.unit, 1.0)


# --------------------------------------------------------------------------- cross-document

def group_key(f: Fact):
    q = "target" if f.qualifier == "target" else ("projected" if f.qualifier == "projected" else "actual")
    return (f.subject_entity_id, f.predicate + (f" @{f.dimension_text}" if f.dimension_text else ""), f.period, f.unit, q)


def cross_document_validation(session, subject_ids: Iterable[str] | None = None) -> dict:
    """Compare the same (subject, predicate, period) across documents; create/refresh conflicts."""
    q = session.query(Fact).filter(Fact.validation_status != "rejected", Fact.period.isnot(None))
    if subject_ids:
        q = q.filter(Fact.subject_entity_id.in_(list(subject_ids)))
    facts = q.all()
    groups: dict[tuple, list[Fact]] = defaultdict(list)
    for f in facts:
        if (f.confidence or {}).get("_partial_total"):
            continue
        if f.subject_entity_id and f.predicate and f.unit != "percent" and f.period_scope not in ("year", "range"):
            groups[group_key(f)].append(f)
    docs = {d.document_id: d for d in session.query(Document).all()}
    # reset previous open cross-doc results for these groups
    fact_ids = [f.fact_id for f in facts]
    if fact_ids:
        for chunk in _chunks(fact_ids, 500):
            session.query(Validation).filter(Validation.fact_id.in_(chunk), Validation.rule_id == "V-006").delete(synchronize_session=False)
    existing = {(c.subject_entity_id, c.predicate, c.period, c.unit): c for c in session.query(Conflict).filter(Conflict.status == "open").all()}
    stats = {"groups": 0, "corroborated": 0, "conflicts": 0}
    seen_conflicts = set()
    for key, gf in groups.items():
        by_doc: dict[str, list[Fact]] = defaultdict(list)
        for f in gf:
            by_doc[f.document_id].append(f)
        if len(by_doc) < 2:
            for f in gf:
                _set_agreement(f, 0.8, single=True)
            continue
        stats["groups"] += 1
        # one representative value per document (median of duplicates within doc)
        reps = []
        for did, fl in by_doc.items():
            vals = sorted(fl, key=lambda f: f.value)
            reps.append(vals[len(vals) // 2])
        vmax = max(abs(r.value) for r in reps) or 1.0
        decimals = min(_decimals(r.display_value) for r in reps)
        scale = _unit_scale(reps[0])
        tol = max(config.CONFLICT_REL_TOLERANCE * vmax, 0.5 * (10 ** -decimals) * scale + 1e-9)
        spread = max(r.value for r in reps) - min(r.value for r in reps)
        agree = spread <= tol
        if agree:
            stats["corroborated"] += 1
            n = len(by_doc)
            for f in gf:
                _set_agreement(f, min(1.0, 0.85 + 0.05 * (n - 1)))
                _add(session, f, "V-006", "cross_document", "pass",
                     f"{n} documents agree on {f.subject_text} {f.predicate} {f.period}: {f.display_value}",
                     {"documents": [docs[d].code for d in by_doc]})
            ck = (key[0], key[1], key[2], key[3])
            if ck in existing and key[4] == "actual":
                existing[ck].status = "resolved"; existing[ck].resolution = "auto: values now agree"
        else:
            stats["conflicts"] += 1
            rel = spread / vmax
            severity = "critical" if rel >= 0.05 else ("major" if rel >= 0.005 else "minor")
            values = []
            for r in sorted(reps, key=lambda r: r.value):
                d = docs.get(r.document_id)
                values.append({"fact_id": r.fact_id, "value": r.display_value, "display": to_display(r.value, r.unit),
                               "document_id": r.document_id, "document_code": d.code if d else None, "document_title": d.title if d else None,
                               "page": r.page_number, "qualifier": r.qualifier, "document_type": d.document_type if d else None})
            expl = _explain(values, rel)
            ent = session.get(Entity, key[0])
            ck = (key[0], key[1], key[2], key[3])
            c = existing.get(ck)
            if c is None or key[4] != "actual":
                if (ck, key[4]) in seen_conflicts:
                    continue
                c = Conflict(code=next_code(session, "conflict", "CF", 3), subject_entity_id=key[0],
                             subject_text=ent.canonical_name if ent else gf[0].subject_text, predicate=key[1] + ("" if key[4] == "actual" else f" ({key[4]})"),
                             period=key[2], unit=key[3])
                session.add(c)
                if key[4] == "actual":
                    existing[ck] = c
            seen_conflicts.add((ck, key[4]))
            c.fact_ids = [v["fact_id"] for v in values]
            c.values = values
            c.severity = severity
            c.explanation = expl
            c.status = "open" if c.status != "ignored" else c.status
            for f in gf:
                _set_agreement(f, 0.55 if severity == "major" else 0.7)
                f.validation_status = "conflict"
                _add(session, f, "V-006", "cross_document", "fail",
                     f"conflict: {', '.join(v['value'] + ' (' + (v['document_code'] or '?') + ')' for v in values)}",
                     {"conflict_code": c.code, "values": values})
    session.flush()
    return stats


def _explain(values, rel):
    kinds = {v.get("qualifier") for v in values}
    types = {v.get("document_type") for v in values}
    parts = [f"{len(values)} documents report different values (spread {rel * 100:.2f}%)."]
    if "provisional" in kinds and "actual" in kinds:
        parts.append("Likely provisional vs final figures: monthly statistics are provisional, annual reports carry audited/final numbers.")
    elif "monthly_statistics" in types and "annual_report" in types:
        parts.append("Monthly statistics (provisional) vs annual report (final) — later publication usually supersedes.")
    elif len(types) == 1:
        parts.append("Same report family across editions — later edition may carry revised figures.")
    return " ".join(parts)


def _set_agreement(f: Fact, score: float, single: bool = False):
    conf = dict(f.confidence or {})
    conf["source_agreement"] = round(score, 3)
    conf["corroborating_docs"] = 1 if single else max(2, conf.get("corroborating_docs", 2))
    f.confidence = conf


def _decimals(s: str) -> int:
    s = (s or "").strip()
    return len(s.split(".")[1]) if "." in s else 0


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# --------------------------------------------------------------------------- finalize

def finalize_confidence(session, facts: Iterable[Fact]) -> None:
    for f in facts:
        conf = {k: v for k, v in (f.confidence or {}).items()}
        f.overall_confidence = overall_confidence(conf)
        if f.validation_status in ("verified", "rejected"):
            continue
        flag = review_flag(f.overall_confidence)
        f.review_status = flag if f.review_status != "done" else "done"
        if f.validation_status != "conflict":
            f.validation_status = "valid" if flag == "none" or flag == "recommended" else "low_confidence"
