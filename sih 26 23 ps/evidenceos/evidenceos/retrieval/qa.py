"""Query planning + evidence-grounded answering (PRD §11.5, §12).

Routes: numeric fact (structured SQL over validated facts) · comparison · conflict · topic · report · explanatory (hybrid RAG).
Every factual answer carries citations (document, page, region/table/cell, bbox).  If evidence is insufficient the
system abstains ("Not enough evidence").  An optional OpenAI-compatible LLM can *phrase* explanatory answers, but a
fail-closed citation registry guarantees it cannot cite anything that was not retrieved.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Optional

import httpx

from .. import config
from ..db import Conflict, Document, Entity, EntityAlias, Fact, Page, Region, Table
from ..docai.normalize import MONTHS, parse_period, to_display
from ..knowledge.entities import JUNK_LABEL_RE, norm_key, mine_key
from . import index as retrieval_index

PRED_WORDS = [
    ("overburden", re.compile(r"\b(overburden|ob removal|obr|ob)\b", re.I)),
    ("dispatch", re.compile(r"\b(despatch(?:ed|es)?|dispatch(?:ed|es)?|offtake|off-take|supply|supplies|supplied|sent to)\b", re.I)),
    ("lignite_resources_total", re.compile(r"\blignite\b.*\b(resource|reserve)s?\b|\b(resource|reserve)s?\b.*\blignite\b", re.I)),
    ("resources_measured", re.compile(r"\b(measured|proved)\b.*\b(resource|reserve)s?|\b(resource|reserve)s?\b.*\b(measured|proved)\b", re.I)),
    ("resources_indicated", re.compile(r"\bindicated\b", re.I)),
    ("resources_inferred", re.compile(r"\binferred\b", re.I)),
    ("resources_total", re.compile(r"\b(resource|reserve|inventory)s?\b", re.I)),
    ("production", re.compile(r"\b(production|produce[ds]?|output|raw coal)\b", re.I)),
    ("generation", re.compile(r"\b(generation|generated)\b", re.I)),
]
TARGET_RE = re.compile(r"\btarget", re.I)
GROWTH_RE = re.compile(r"\bgrowth\b", re.I)
ACH_RE = re.compile(r"\b(achievement|achieved|achmt)\b", re.I)
CONFLICT_RE = re.compile(r"\b(conflict|conflicting|discrepanc|mismatch|inconsisten|disagree|differ)", re.I)
REPORT_RE = re.compile(r"\b(generate|create|prepare|draft)\b.*\breport\b|\breport\b.*\b(generate|create|prepare)\b", re.I)
TOPIC_RE = re.compile(r"\b(topic|topics|word ?cloud|themes?|keywords?|most common|frequent terms)\b", re.I)
COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference between|higher|lower|more than|less than)\b", re.I)
WHY_RE = re.compile(r"\b(why|explain|reason|because|how does|what is the|describe|what are|which)\b", re.I)


class Planner:
    def __init__(self, session):
        self.session = session
        self.aliases: list[tuple[str, str, str]] = []
        for a, e in session.query(EntityAlias, Entity).join(Entity, Entity.entity_id == EntityAlias.entity_id).all():
            if a.alias_norm and len(a.alias_norm) >= 2:
                self.aliases.append((a.alias_norm, e.entity_id, e.entity_type))
        self.aliases.sort(key=lambda x: -len(x[0]))
        self.entities = {e.entity_id: e for e in session.query(Entity).all()}

    SUBJECT_TYPES = {"company", "subsidiary", "mine", "state", "sector", "country", "organization", "coalfield", "project", "area"}

    def find_entities(self, q: str) -> list[Entity]:
        qn = " " + norm_key(q) + " "
        found: list[Entity] = []
        used: list[tuple[int, int]] = []
        for alias, eid, etype in self.aliases:
            if etype not in self.SUBJECT_TYPES or len(alias) < 3 or JUNK_LABEL_RE.match(alias):
                continue
            if alias in ("total", "others", "india", "overall", "country", "grand total", "misc") and len(alias) < 8:
                # only accept generic aggregates when explicitly asked for all-India
                if alias == "india" and re.search(r"\b(india|all india|country|national)\b", q, re.I):
                    pass
                else:
                    continue
            pat = " " + alias + " "
            pos = qn.find(pat)
            if pos < 0:
                continue
            span = (pos + 1, pos + 1 + len(alias))          # exclude the delimiting spaces
            if any(s < span[1] and span[0] < e for s, e in used):
                continue
            used.append(span)
            e = self.entities[eid]
            if e not in found:
                found.append(e)
        return found

    def classify(self, q: str) -> str:
        if REPORT_RE.search(q):
            return "report"
        if CONFLICT_RE.search(q):
            return "conflict"
        if TOPIC_RE.search(q):
            return "topic"
        ents = self.find_entities(q)
        pred = detect_predicate(q, self, ents)
        if COMPARE_RE.search(q) and pred and (len(ents) >= 2 or len(_period_filters(q)) >= 2):
            return "comparison"
        if pred and (ents or parse_period(q)) and not re.match(r"^\s*(why|explain|how does|describe)\b", q, re.I):
            return "numeric"
        if WHY_RE.search(q) or not pred:
            return "explanatory"
        return "explanatory"


PREFIX_WORDS = [("lignite_", re.compile(r"\blignite\b", re.I)), ("noncoking_", re.compile(r"\bnon[- ]?coking\b", re.I)),
                ("coking_", re.compile(r"(?<!non[- ])(?<!non)\bcoking\b", re.I)), ("washed_", re.compile(r"\bwashed\b", re.I))]


def detect_predicate(q: str, planner: Optional["Planner"] = None, ents: Optional[list] = None) -> Optional[str]:
    # entity names must not drive the metric ('Bharat Coking Coal' is a company, not a coking-coal question)
    q_wo = q
    if planner is not None and ents:
        ids = {e.entity_id for e in ents}
        names = sorted({e.canonical_name for e in ents} | {a for a, eid, _t in planner.aliases if eid in ids}, key=len, reverse=True)
        for n in names:
            q_wo = re.sub(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", " ", q_wo, flags=re.I)
    for p, rx in PRED_WORDS:
        if rx.search(q_wo):
            base = p
            if not base.startswith("lignite_"):
                for pf, prx in PREFIX_WORDS:
                    if prx.search(q_wo):
                        base = pf + base
                        break
            if GROWTH_RE.search(q):
                return f"{base}_growth_pct"
            if ACH_RE.search(q):
                return f"{base}_achievement_pct"
            return base
    return None


def _acl_docs(session, user) -> set[str]:
    role = getattr(user, "role", "viewer") or "viewer"
    max_lvl = config.ROLE_MAX_ACL.get(role, 0)
    return {d.document_id for d in session.query(Document).all() if config.ACL_ORDER.get(d.acl_level or "public", 0) <= max_lvl}


# --------------------------------------------------------------------------- citations

def citation_for_fact(session, f: Fact, doc: Document) -> dict:
    return {
        "fact_id": f.fact_id, "fact_code": f.code, "document_id": f.document_id, "document_code": doc.code, "document_title": doc.title,
        "source": doc.source, "page": f.page_number, "page_id": f.page_id, "region_id": f.region_id, "table_id": f.table_id, "cell_id": f.cell_id,
        "bbox": f.bbox, "value": to_display(f.value, f.unit), "original_value": f.original_value, "qualifier": f.qualifier,
        "period": f.period, "confidence": f.overall_confidence, "validation_status": f.validation_status,
        "crop_url": f"/api/v1/facts/{f.fact_id}/crop", "viewer_url": f"/#/viewer/{f.document_id}/{f.page_number}?region={f.region_id or ''}&fact={f.fact_id}",
        "context": (f.context or "")[:200],
    }


def citation_for_chunk(session, ch: dict) -> dict:
    doc = session.get(Document, ch["document_id"])
    reg = session.get(Region, ch["region_id"]) if ch.get("region_id") else None
    bbox = reg.bbox if reg else None
    return {
        "chunk_id": ch["chunk_id"], "level": ch["level"], "document_id": ch["document_id"], "document_code": doc.code if doc else None,
        "document_title": doc.title if doc else None, "source": doc.source if doc else None, "page": ch.get("page_number"), "page_id": ch.get("page_id"),
        "region_id": ch.get("region_id"), "table_id": ch.get("table_id"), "fact_id": ch.get("fact_id"), "bbox": bbox,
        "snippet": ch["text"][:300], "score": round(ch["score"], 4),
        "viewer_url": f"/#/viewer/{ch['document_id']}/{ch.get('page_number') or 1}?region={ch.get('region_id') or ''}",
    }


# --------------------------------------------------------------------------- structured route

def _period_filters(q: str) -> list[tuple[str, str]]:
    """All periods mentioned in the question ('FY 2023-24 and FY 2024-25', 'March 2025 vs March 2024')."""
    ql = q.lower().replace("’", "'")
    frags = re.split(r"\band\b|,|\bvs\.?\b|\bversus\b|\bwith\b|\bagainst\b|\bto\b(?=\s+(?:fy|20\d{2}|[a-z]{3,9}\s*'?\d{2}))", ql)
    out: list[tuple[str, str]] = []
    for frag in frags + [ql]:
        pk, sc = _period_filter(frag)
        if pk and (pk, sc) not in out:
            out.append((pk, sc))
    # the whole-question parse comes last; drop it when fragments already produced periods
    if len(out) > 1:
        whole = _period_filter(ql)
        if whole in out and any(o != whole for o in out):
            out = [o for o in out if o != whole] + ([whole] if sum(1 for f in frags if _period_filter(f) == whole) else [])
    return out


def _period_filter(q: str) -> tuple[Optional[str], Optional[str]]:
    """Return (period_key, scope) requested in the question, if any."""
    ql = q.lower().replace("’", "'")
    m = re.search(r"\b(in|for|during|of)\s+(20\d{2})\b(?!\s*-)", ql)
    p = parse_period(ql)
    if p is None:
        return None, None
    if p.scope == "year" and m:
        # bare year: prefer fiscal year starting that year for production-type metrics
        y = int(m.group(2))
        return f"FY{y}-{(y + 1) % 100:02d}", "fy_or_year"
    if p.scope == "month":
        return p.key, "month"
    if p.scope == "asof":
        return p.key, "asof"
    if p.scope == "fy":
        return p.key, "fy"
    if p.scope == "ytd":
        return p.key, "ytd"
    return p.key, p.scope


def _match_period(f: Fact, period: str, scope: str) -> bool:
    if not f.period:
        return False
    if scope == "fy_or_year":
        y = period[2:6]
        return f.period == period or f.period.startswith("Y" + y) or f.period.startswith(period)
    if scope == "asof":
        return f.period == period or f.period.startswith("asof:" + period[5:9])
    return f.period == period


KNOWN_WORDS = re.compile(r"^(what|which|how|compare|comparison|coal|india|fy|mt|production|dispatch|despatch|during|upto|target|total|the|and|for|"
                         r"of|in|to|vs|versus|between|was|were|is|are|did|much|many|from|by|on|as|at|with|sector|sectors|resources|reserves|"
                         r"lignite|coking|non|raw|offtake|per|rate|growth|q[1-4]|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
                         r"january|february|march|april|june|july|august|september|october|november|december|year|years|month|quarter|"
                         r"mine|mines|company|companies|state|states|all|list|give|show|tell|me|about|please|latest|current|recent|"
                         r"overburden|ob|removal|stripping|ratio|measured|indicated|inferred|proved|geological|inventory|open|cast|underground|"
                         r"power|steel|cement|sponge|iron|others|captive|commercial|utilities|plants|figure|figures|value|values|number|numbers|"
                         r"data|table|report|reports|document|documents|disagree|conflict|conflicts|discrepancy|difference|differ|higher|lower|"
                         r"than|more|less|increase|decrease|increased|decreased|explain|why|reason|reasons|trend|trends|topic|topics|summary|"
                         r"summarise|summarize|compared|comparison|according|across|each|every|both|its|their|has|have|had|be|been|it|this|that|"
                         r"these|those|there|any|some|most|top|last|first|second|third|next|previous|new|old|over|under|up|down|out|into|"
                         r"tonnes|tonne|million|lakh|crore|billion|thousand|rupees|rs|inr|amount|quantity|qty|share|percent|percentage|"
                         r"achievement|achieved|actual|actuals|provisional|final|estimated|projected|annual|monthly|yearly|daily|average|"
                         r"cumulative|since|till|until|before|after|through|via|use|used|using|make|made|get|got|find|found|know|see|say|"
                         r"said|do|does|done|can|could|would|should|will|shall|may|might|must|not|no|yes|also|only|even|just|so|if|then|"
                         r"else|when|where|who|whom|whose|while|because|although|though|whether|either|neither|nor|but|or|a|an|s|am|pm)$", re.I)


def _unknown_terms(q: str, planner: Planner) -> list[str]:
    """Proper-noun-like tokens of the question that match no known entity alias (e.g. 'NMDC', 'Mahagenco')."""
    qn = " " + norm_key(q) + " "
    out = []
    for tok in re.findall(r"[A-Za-z][A-Za-z&./'-]{2,}", q):
        tok = re.sub(r"['’]s$", "", tok)
        low = tok.lower().strip(".,'")
        if KNOWN_WORDS.match(low) or not (tok[:1].isupper() or tok.isupper()):
            continue
        if any((" " + a + " ") in qn for a, _e, _t in planner.aliases if low in a.split() or a == low):
            continue
        out.append(tok.strip(".,'"))
    return out


def answer_numeric(session, q: str, planner: Planner, allowed: set[str], filters: dict) -> dict:
    ents = planner.find_entities(q)
    pred = detect_predicate(q, planner, ents)
    unknown = _unknown_terms(q, planner) if not ents else []
    if unknown:
        return {"answer": None, "missing_evidence": True, "reason": f"unknown entity: {', '.join(unknown)}", "unknown_terms": unknown}
    qual = "target" if TARGET_RE.search(q) else None
    periods = _period_filters(q)
    if filters.get("year") and not periods:
        y = int(filters["year"]); periods = [(f"FY{y}-{(y + 1) % 100:02d}", "fy_or_year")]
    if not ents and re.search(r"\b(india|all india|country|national|total)\b", q, re.I):
        ents = [e for e in planner.entities.values() if e.canonical_name == "India"]
    if not pred:
        return {"answer": None, "missing_evidence": True, "reason": "no metric detected"}
    fq = session.query(Fact).filter(Fact.document_id.in_(list(allowed)), Fact.validation_status != "rejected")
    fq = fq.filter(Fact.predicate == pred) if not pred.startswith("resources_total") else fq.filter(Fact.predicate.in_([pred, "resources_total"]))
    # sectors act as a dimension (dispatch *to* a sector), other entities as subjects
    sector_ents = [e for e in ents if e.entity_type == "sector"]
    subj_ents = [e for e in ents if e.entity_type != "sector"]
    if sector_ents:
        fq = fq.filter(Fact.dimension_entity_id.in_([e.entity_id for e in sector_ents]))
        if not subj_ents:
            subj_ents = [e for e in planner.entities.values() if e.canonical_name == "India"]
    else:
        fq = fq.filter(Fact.dimension_text.is_(None))
    if subj_ents:
        fq = fq.filter(Fact.subject_entity_id.in_([e.entity_id for e in subj_ents]))
    ents = subj_ents
    if qual == "target":
        fq = fq.filter(Fact.qualifier == "target")
    else:
        fq = fq.filter(Fact.qualifier.in_(["actual", "provisional"]))
    all_facts = [f for f in fq.all() if not (f.confidence or {}).get("_partial_total")]   # totals of listed mines are not company figures
    note = None
    if periods:
        facts = []
        for pk, sc in periods:
            sel = [f for f in all_facts if _match_period(f, pk, sc)]
            exact = [f for f in sel if f.period == pk]
            facts.extend(exact or sel)
        if not facts and all_facts and ents:
            # entity + metric exist but not for the requested period(s): show what is available instead of abstaining
            avail = sorted({f.period for f in all_facts if f.period and f.period_scope in ("fy", "ytd", "month", "asof")}, reverse=True)[:4]
            facts = [f for f in all_facts if f.period in avail]
            note = (f"No validated fact for {', '.join(p for p, _ in periods)}; showing the closest available periods "
                    f"({', '.join(avail)}). Treat as partial evidence.")
    else:
        facts = all_facts
        if len(ents) <= 1 and facts:
            # no period asked: prefer full fiscal years / latest values, keep the answer short
            keep = sorted({f.period for f in facts if f.period and f.period_scope in ("fy", "asof")}, reverse=True)[:3]
            facts = [f for f in facts if f.period in keep] or facts
    if not facts:
        return {"answer": None, "missing_evidence": True, "reason": "no validated fact matches", "entities": [e.canonical_name for e in ents],
                "predicate": pred, "period": [p for p, _ in periods]}
    docs = {d.document_id: d for d in session.query(Document).filter(Document.document_id.in_({f.document_id for f in facts})).all()}
    # group by subject+dimension+period, collect distinct values
    groups: dict[tuple, list[Fact]] = defaultdict(list)
    for f in facts:
        groups[(f.subject_entity_id, f.dimension_text, f.period)].append(f)
    lines, cits, conflicts = [], [], []
    overall = []
    summary: list[tuple[str, str, float, str]] = []
    for (sid, dim, per), fl in sorted(groups.items(), key=lambda kv: (kv[0][2] or "", planner.entities.get(kv[0][0]).canonical_name if planner.entities.get(kv[0][0]) else "")):
        ent = planner.entities.get(sid)
        name = ent.canonical_name if ent else fl[0].subject_text
        if dim:
            name += f" → {dim}"
        by_val: dict[str, list[Fact]] = defaultdict(list)
        for f in fl:
            by_val[to_display(f.value, f.unit)].append(f)
        best = max(fl, key=lambda f: (f.overall_confidence or 0, f.qualifier == "actual"))
        if len(by_val) == 1:
            val = next(iter(by_val))
            srcs = sorted({docs[f.document_id].code for f in fl})
            lines.append(f"{name} — {pred.replace('_', ' ')} {per}: **{val}** (source{'s' if len(srcs) > 1 else ''}: {', '.join(srcs)}"
                         f"{'; corroborated by ' + str(len(srcs)) + ' documents' if len(srcs) > 1 else ''})")
            summary.append((name, per, best.value, best.unit))
        else:
            parts = []
            ranked = sorted(by_val.items(), key=lambda kv: (-len({f.document_id for f in kv[1]}), -len(kv[1]), -max(f.overall_confidence or 0 for f in kv[1])))
            for val, ffs in ranked[:3]:
                srcs = sorted({f"{docs[f.document_id].code} p{f.page_number} ({f.qualifier})" for f in ffs})
                parts.append(f"{val} [{'; '.join(srcs)}]")
            more = f" (+{len(ranked) - 3} other value{'s' if len(ranked) - 3 != 1 else ''} in the corpus — see citations)" if len(ranked) > 3 else ""
            lines.append(f"{name} — {pred.replace('_', ' ')} {per}: ⚠ conflicting values: " + " vs ".join(parts) + more)
            cf = session.query(Conflict).filter(Conflict.subject_entity_id == sid, Conflict.period == per, Conflict.status == "open").first()
            if cf:
                conflicts.append({"conflict_id": cf.conflict_id, "code": cf.code, "subject": cf.subject_text, "predicate": cf.predicate, "period": cf.period,
                                  "values": cf.values, "severity": cf.severity, "explanation": cf.explanation})
            # for comparisons use the latest (final) publication's value
            latest = max(fl, key=lambda f: (docs[f.document_id].document_date or "", f.qualifier == "actual"))
            summary.append((name, per, latest.value, latest.unit))
        for f in sorted(fl, key=lambda f: (f.document_id, f.page_number)):
            cits.append(citation_for_fact(session, f, docs[f.document_id]))
        overall.append(best.overall_confidence or 0.5)
    if len(summary) == 2 and summary[0][3] == summary[1][3] and summary[0][2] and summary[1][2]:
        (n1, p1, v1, u), (n2, p2, v2, _) = summary
        diff = v2 - v1
        pct = diff / v1 * 100 if v1 else 0.0
        lines.append(f"Difference ({n2} {p2} vs {n1} {p1}): {to_display(diff, u)} ({pct:+.2f}%)")
    if note:
        lines.append(f"ℹ {note}")
    conf = round(sum(overall) / len(overall), 3) if overall else 0.0
    if conflicts:
        conf = round(min(conf, 0.75), 3)
    if note:
        conf = round(min(conf, 0.6), 3)
    return {"answer": "\n".join(lines), "citations": cits, "confidence": conf, "conflicts": conflicts, "missing_evidence": False,
            "partial": bool(note), "entities": [e.canonical_name for e in ents], "predicate": pred, "period": [p for p, _ in periods]}


# --------------------------------------------------------------------------- conflict route

def answer_conflicts(session, q: str, planner: Planner, allowed: set[str]) -> dict:
    ents = planner.find_entities(q)
    pred = detect_predicate(q)
    cq = session.query(Conflict).filter(Conflict.status == "open")
    if ents:
        cq = cq.filter(Conflict.subject_entity_id.in_([e.entity_id for e in ents]))
    if pred:
        cq = cq.filter(Conflict.predicate.like(pred + "%"))
    periods = [p for p, _ in _period_filters(q)]
    if periods:
        cq = cq.filter(Conflict.period.in_(periods))
    cfs = [c for c in cq.order_by(Conflict.severity.desc(), Conflict.created_at.desc()).all()
           if all(v.get("document_id") in allowed for v in (c.values or []))]
    if not cfs:
        return {"answer": "No open conflicts found for that scope. All cross-document comparisons agree within tolerance.",
                "citations": [], "confidence": 0.9, "conflicts": [], "missing_evidence": False}
    lines, cits, out = [], [], []
    for c in cfs[:12]:
        vals = ", ".join(f"{v['value']} ({v['document_code']} p{v['page']}, {v.get('qualifier')})" for v in c.values)
        lines.append(f"{c.code} [{c.severity}] {c.subject_text} — {c.predicate} {c.period}: {vals}. {c.explanation}")
        out.append({"conflict_id": c.conflict_id, "code": c.code, "subject": c.subject_text, "predicate": c.predicate, "period": c.period,
                    "values": c.values, "severity": c.severity, "explanation": c.explanation})
        for fid in c.fact_ids or []:
            f = session.get(Fact, fid)
            if f:
                cits.append(citation_for_fact(session, f, session.get(Document, f.document_id)))
    head = f"{len(cfs)} open conflict{'s' if len(cfs) != 1 else ''} found" + (f" (showing {min(12, len(cfs))})" if len(cfs) > 12 else "") + ":"
    return {"answer": head + "\n" + "\n".join(lines), "citations": cits, "confidence": 0.9, "conflicts": out, "missing_evidence": False}


# --------------------------------------------------------------------------- RAG route

SYSTEM_PROMPT = """You are EvidenceOS, an evidence-first assistant for CMPDI/Coal India.
Answer ONLY from the numbered evidence blocks. Every sentence with a number or claim must end with a citation like [E2].
If the evidence does not contain the answer, reply exactly: NOT ENOUGH EVIDENCE.
Do not invent numbers. Keep units as written (MT = million tonnes). Mention conflicting values if evidence disagrees."""


def call_llm(question: str, evidence: list[str]) -> Optional[str]:
    from .. import cache

    # Check cache for identical query + evidence fingerprint
    import hashlib
    blocks = "\n\n".join(f"[E{i + 1}] {e}" for i, e in enumerate(evidence))
    cache_key = "llm:" + hashlib.sha256(f"{config.LLM_MODEL}:{question}:{blocks[:2000]}".encode()).hexdigest()
    cached = cache.cache_get(cache_key)
    if cached:
        return cached

    # Ollama offline / air-gapped provider support
    if config.LLM_PROVIDER == "ollama" or "11434" in (config.LLM_BASE_URL or "") or "ollama" in (config.LLM_BASE_URL or ""):
        base = (config.LLM_BASE_URL or "http://localhost:11434").rstrip("/")
        # Try native Ollama client first if installed
        try:
            import ollama
            client = ollama.Client(host=base)
            res = client.chat(
                model=config.LLM_MODEL or "llama3:8b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Evidence:\n{blocks}\n\nQuestion: {question}"},
                ],
                options={"temperature": 0.0},
            )
            content = res["message"]["content"]
            if content:
                cache.cache_set(cache_key, content, ttl_seconds=3600)
                return content
        except Exception:
            pass

        # Try native Ollama HTTP endpoint (/api/chat)
        try:
            r = httpx.post(
                f"{base}/api/chat",
                json={
                    "model": config.LLM_MODEL or "llama3:8b",
                    "stream": False,
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Evidence:\n{blocks}\n\nQuestion: {question}"},
                    ],
                },
                timeout=45.0,
            )
            if r.status_code == 200:
                content = r.json().get("message", {}).get("content", "")
                if content:
                    cache.cache_set(cache_key, content, ttl_seconds=3600)
                    return content
        except Exception:
            pass

    # OpenAI / vLLM / Generic OpenAI-compatible endpoint
    if not config.LLM_BASE_URL and not config.LLM_API_KEY:
        return None
    base = config.LLM_BASE_URL or "https://api.openai.com/v1"
    headers = {}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    try:
        url = base.rstrip("/") + ("/chat/completions" if not base.endswith("/v1") else "/chat/completions")
        if not url.endswith("/chat/completions"):
            url = base.rstrip("/") + "/v1/chat/completions"
        r = httpx.post(
            url,
            headers=headers,
            json={
                "model": config.LLM_MODEL or "llama3:8b",
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Evidence:\n{blocks}\n\nQuestion: {question}"},
                ],
            },
            timeout=45.0,
        )
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            if content:
                cache.cache_set(cache_key, content, ttl_seconds=3600)
                return content
        return None
    except Exception:
        return None


def _extractive(question: str, chunks: list[dict]) -> str:
    toks = [t for t in re.findall(r"[a-z0-9]{3,}", question.lower()) if t not in retrieval_index.STOP]
    scored = []
    for i, ch in enumerate(chunks):
        for sent in re.split(r"(?<=[.;])\s+|\n", ch["text"]):
            s = sent.strip()
            if len(s) < 30 or len(s) > 400:
                continue
            if s.count(" | ") >= 3 and not re.search(r"\d", s):
                continue                                   # table header rows carry no answerable content
            ov = sum(1 for t in toks if t in s.lower())
            if ov:
                scored.append((ov + (0.5 if ch["level"] == "fact" else 0), i, s))
    scored.sort(key=lambda x: -x[0])
    seen, lines = set(), []
    for ov, i, s in scored:
        key = s[:60]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{s} [E{i + 1}]")
        if len(lines) >= 4:
            break
    return "\n".join(lines)


def answer_rag(session, q: str, allowed: set[str], filters: dict, required_terms: Optional[list[str]] = None) -> dict:
    hits = retrieval_index.search(session, q, top_k=config.RETRIEVAL_TOP_K, allowed_doc_ids=allowed, filters=filters)
    hits = hits[: config.EVIDENCE_TOP_K]
    if not hits:
        return {"answer": "NOT ENOUGH EVIDENCE — no relevant passages were retrieved from the authorised corpus.", "citations": [],
                "confidence": 0.0, "conflicts": [], "missing_evidence": True, "mode": "abstain"}
    missing_terms = [t for t in (required_terms or []) if not any(t.lower() in (h["text"] or "").lower() for h in hits)]
    if missing_terms:
        return {"answer": f"NOT ENOUGH EVIDENCE — the authorised corpus contains no passage mentioning {', '.join(missing_terms)}. "
                          f"Showing the closest (non-answering) evidence for transparency.",
                "citations": [citation_for_chunk(session, h) for h in hits[:3]], "confidence": 0.1, "conflicts": [], "missing_evidence": True,
                "mode": "abstain", "unknown_terms": missing_terms}
    evidence = [f"({session.get(Document, h['document_id']).code}, p{h.get('page_number')}) {h['text'][:900]}" for h in hits]
    llm = call_llm(q, evidence)
    mode = "llm"
    if llm is None:
        llm = _extractive(q, hits)
        mode = "extractive (no LLM configured)"
    if not llm or "NOT ENOUGH EVIDENCE" in llm.upper():
        return {"answer": "NOT ENOUGH EVIDENCE — the retrieved passages do not answer the question. Showing the closest evidence instead.",
                "citations": [citation_for_chunk(session, h) for h in hits], "confidence": 0.2, "conflicts": [], "missing_evidence": True, "mode": mode}
    # fail-closed citation registry: only [E#] that exist survive; answers without any valid citation are downgraded
    valid = set(range(1, len(hits) + 1))
    cited = {int(x) for x in re.findall(r"\[E(\d+)\]", llm)}
    bad = cited - valid
    for b in bad:
        llm = llm.replace(f"[E{b}]", "[citation removed: not in evidence]")
    used = sorted(cited & valid)
    if not used:
        return {"answer": "NOT ENOUGH EVIDENCE — the generated answer could not be tied to retrieved evidence, so it was withheld.",
                "citations": [citation_for_chunk(session, h) for h in hits], "confidence": 0.2, "conflicts": [], "missing_evidence": True, "mode": mode}
    cits = []
    for i in used:
        c = citation_for_chunk(session, hits[i - 1]); c["label"] = f"E{i}"; cits.append(c)
    conf = round(min(0.9, 0.5 + 0.1 * len(used) + (0.1 if mode == "llm" else 0)), 3)
    return {"answer": llm, "citations": cits, "confidence": conf, "conflicts": [], "missing_evidence": False, "mode": mode}


# --------------------------------------------------------------------------- entry point

def answer(session, q: str, user=None, filters: Optional[dict] = None) -> dict:
    filters = filters or {}
    planner = Planner(session)
    allowed = _acl_docs(session, user)
    route = planner.classify(q)
    res: dict
    if route == "numeric" or route == "comparison":
        res = answer_numeric(session, q, planner, allowed, filters)
        if res.get("missing_evidence"):
            rag = answer_rag(session, q, allowed, filters, required_terms=res.get("unknown_terms"))
            rag["structured_attempt"] = {k: res.get(k) for k in ("entities", "predicate", "period", "reason")}
            res = rag
            route = f"{route}→rag"
    elif route == "conflict":
        res = answer_conflicts(session, q, planner, allowed)
    elif route == "topic":
        from .. import topics
        t = topics.topics_summary(session, filters)
        res = {"answer": t["summary"], "citations": [], "confidence": 0.8, "conflicts": [], "missing_evidence": False, "topics": t["topics"]}
    elif route == "report":
        res = {"answer": "Use the Report Generator (POST /api/v1/reports/generate) — choose a template, period and entities; the report is built only from validated facts with citations.",
               "citations": [], "confidence": 1.0, "conflicts": [], "missing_evidence": False}
    else:
        res = answer_rag(session, q, allowed, filters, required_terms=_unknown_terms(q, planner) if not planner.find_entities(q) else None)
    res["route"] = route
    res["query"] = q
    return res
