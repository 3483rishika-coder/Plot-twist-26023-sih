"""End-to-end pipeline orchestration (PRD §8): upload → hash → classify → per-page Document AI →
facts → validation → conflicts → index, with status transitions and audit entries.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pymupdf

from . import config, store
from .db import (AuditLog, Cell, Document, DocumentVersion, Fact, Job, Page, Region, Table, next_code, session_scope)
from .docai import pages as docai_pages
from .docai.normalize import parse_period
from .knowledge import validation
from .knowledge.entities import Resolver, seed_entities
from .knowledge.facts import DocContext, facts_from_table, facts_from_text
from .retrieval import index as retrieval_index

STATUS_FLOW = ["queued", "processing", "extracted", "validated", "reviewed", "published", "failed"]


def audit(session, action: str, user=None, entity_type: str | None = None, entity_id: str | None = None,
          query: str | None = None, document_id: str | None = None, detail: dict | None = None):
    session.add(AuditLog(user_id=getattr(user, "user_id", None), username=getattr(user, "username", "system"), action=action,
                         entity_type=entity_type, entity_id=entity_id, query=query, document_accessed=document_id, detail=detail or {}))


# --------------------------------------------------------------------------- ingestion

def infer_doc_type(title: str, filename: str) -> str:
    t = f"{title} {filename}".lower()
    if "monthly" in t or "msg-" in t:
        return "monthly_statistics"
    if "directory" in t:
        return "coal_directory"
    if "inventory" in t:
        return "inventory"
    if "annual report" in t or "annualreport" in t:
        return "annual_report"
    if re.search(r"promotion|order|contract|agreement|tender|circular|office", t):
        return "administrative"
    return "report"


def infer_period(title: str, first_text: str) -> Optional[str]:
    for src in (title, first_text[:600] if first_text else ""):
        if not src:
            continue
        p = parse_period(src)
        if p and p.scope in ("fy", "month"):
            return p.key
    m = re.search(r"annualreport(20\d{2})", title.lower() + " ") or re.search(r"annual report (20\d{2})", title.lower())
    if m:
        y = int(m.group(1)); return f"FY{y - 1}-{y % 100:02d}"
    return None


def ingest_file(session, path: str | Path, title: str | None = None, source: str | None = None, source_url: str | None = None,
                doc_type: str | None = None, period: str | None = None, acl_level: str = "public", link: bool = False,
                department: str | None = None, meta: dict | None = None, user=None, enqueue: bool = True,
                job_params: dict | None = None) -> tuple[Document, bool]:
    """Register a document. Returns (document, created). Exact duplicates (same SHA-256) are not re-ingested."""
    path = Path(path)
    sha = store.sha256_file(path)
    existing = session.query(Document).filter(Document.sha256 == sha).first()
    if existing is not None:
        audit(session, "upload_duplicate", user, "document", existing.document_id, detail={"filename": path.name})
        return existing, False
    title = title or path.stem
    doc = Document(code=next_code(session, "document", "DOC"), original_filename=path.name, title=title, sha256=sha,
                   source=source, source_url=source_url, document_type=doc_type or infer_doc_type(title, path.name),
                   department=department, acl_level=acl_level, status="queued", meta=meta or {}, file_size=path.stat().st_size)
    session.add(doc); session.flush()
    doc.storage_location = store.save_original(path, doc.document_id, path.name, link=link)
    try:
        pdf = pymupdf.open(doc.storage_location)
        doc.page_count = pdf.page_count
        first_text = " ".join(pdf[i].get_text() for i in range(min(3, pdf.page_count)))
        sample = list(range(min(pdf.page_count, 12)))
        scanned = sum(1 for i in sample if docai_pages.is_scanned_page(pdf[i]))
        doc.scan_ratio = round(scanned / max(len(sample), 1), 2)
        doc.is_scanned = doc.scan_ratio >= 0.5
        pdf.close()
    except Exception as e:  # not a PDF → still stored, processing will fail gracefully
        first_text = ""
        doc.meta = {**(doc.meta or {}), "open_error": str(e)}
    doc.period = period or infer_period(title, first_text)
    if doc.period and not doc.document_date:
        p = parse_period(doc.period)
        if p and p.year:
            doc.document_date = f"{p.year:04d}-{p.key[-2:] if p.scope == 'month' else '03'}-31"
    # version linking: same family (normalized title without period tokens)
    fam = re.sub(r"(20\d{2}\s*-\s*\d{2,4}|fy\s*\d{2,4}|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\W*\d{2,4})", "", title.lower())
    fam = re.sub(r"[^a-z]+", " ", fam).strip()
    doc.family_key = fam
    prev = session.query(Document).filter(Document.family_key == fam, Document.document_id != doc.document_id).order_by(Document.created_at).all()
    doc.version = len(prev) + 1
    session.add(DocumentVersion(document_id=doc.document_id, parent_document_id=prev[-1].document_id if prev else None, sha256=sha, version_number=doc.version))
    audit(session, "upload", user, "document", doc.document_id, detail={"filename": path.name, "sha256": sha, "code": doc.code})
    if enqueue:
        session.add(Job(document_id=doc.document_id, stage="process", status="queued", params=job_params or {}))
    session.flush()
    return doc, True


# --------------------------------------------------------------------------- processing

def _persist_page(session, doc: Document, pr: docai_pages.PageResult) -> tuple[Page, list[tuple[Region, Table, dict]]]:
    page = Page(document_id=doc.document_id, page_number=pr.page_number, width=pr.width, height=pr.height,
                is_scanned=pr.is_scanned, text=pr.text, quality=pr.quality, ocr_confidence=pr.ocr_confidence, status="extracted")
    session.add(page); session.flush()
    table_regions = []
    for r in pr.regions:
        reg = Region(page_id=page.page_id, document_id=doc.document_id, type=r.type, bbox=r.bbox, reading_order=r.reading_order,
                     confidence=r.confidence, text=r.text, meta=r.meta)
        session.add(reg); session.flush()
        if r.table is not None:
            t = r.table
            tab = Table(region_id=reg.region_id, page_id=page.page_id, document_id=doc.document_id, caption=t.caption, headers=t.headers,
                        rows=t.rows, n_rows=len(t.rows), n_cols=len(t.headers), header_rows=t.header_rows, unit_hint=t.unit_hint,
                        method=t.method, confidence=t.confidence, meta={"row_bboxes": t.row_bboxes, "col_bounds": [list(b) for b in t.col_bounds], **t.meta})
            session.add(tab); session.flush()
            cells_by_pos = {}
            for c in t.cells:
                cell = Cell(table_id=tab.table_id, row_index=c.row, col_index=c.col, bbox=c.bbox, text=c.text, confidence=c.conf)
                session.add(cell)
                cells_by_pos[(c.row, c.col)] = cell
            session.flush()
            table_regions.append((reg, tab, {k: {"cell_id": v.cell_id, "bbox": v.bbox, "conf": v.confidence} for k, v in cells_by_pos.items()}))
    return page, table_regions


def _group_multipage_tables(session, tables: list[Table]) -> None:
    """Link continuation tables across consecutive pages (same column count, matching/empty headers, top-of-page start)."""
    by_page = defaultdict(list)
    pages = {}
    for t in tables:
        by_page[t.page_id].append(t)
    page_rows = {p.page_id: p for p in session.query(Page).filter(Page.page_id.in_(list(by_page.keys()))).all()} if by_page else {}
    ordered = sorted(page_rows.values(), key=lambda p: p.page_number)
    prev_last = None
    prev_page_no = None
    for p in ordered:
        tabs = sorted(by_page[p.page_id], key=lambda t: t.meta.get("row_bboxes", [[0, 0, 0, 0]])[0][1] if t.meta.get("row_bboxes") else 0)
        first = tabs[0]
        if prev_last is not None and prev_page_no == p.page_number - 1:
            same_cols = first.n_cols == prev_last.n_cols
            h1 = [h.lower().strip() for h in (prev_last.headers or [])]
            h2 = [h.lower().strip() for h in (first.headers or [])]
            hdr_match = h1 == h2 or (sum(1 for h in h2 if h) <= 1) or _header_similarity(h1, h2) > 0.7
            top = first.meta.get("row_bboxes", [[0, 0, 0, 0]])[0][1] < 0.25 * (p.height or 800)
            if same_cols and hdr_match and top:
                gid = prev_last.table_group_id or prev_last.table_id
                prev_last.table_group_id = gid
                first.table_group_id = gid
                seq = list(prev_last.page_sequence or [prev_page_no])
                if p.page_number not in seq:
                    seq.append(p.page_number)
                for t in session.query(Table).filter(Table.table_group_id == gid).all():
                    t.page_sequence = seq
                first.page_sequence = seq
                if not any(first.headers) and prev_last.headers:
                    first.headers = prev_last.headers
                    first.meta = {**first.meta, "headers_inherited_from": prev_last.table_id}
        prev_last = tabs[-1]
        prev_page_no = p.page_number


def _header_similarity(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    same = sum(1 for x, y in zip(a, b) if x == y or (x and y and (x in y or y in x)))
    return same / len(a)


def _intro_text(regions, table_bbox, max_gap: float = 90.0) -> str:
    """Text of the nearest paragraph/heading printed just above a table (used as lead-in context)."""
    best, best_gap = None, None
    for r in regions:
        if r.type not in ("text", "title", "caption") or not r.text or len(r.text.strip()) < 30:
            continue                                  # skip unit hints such as '(In MT)'
        gap = table_bbox[1] - r.bbox[3]
        if -2 <= gap <= max_gap and (best_gap is None or gap < best_gap):
            best, best_gap = r, gap
    return best.text[-600:] if best else ""


def process_document(document_id: str, max_pages: int | None = None, page_range: tuple[int, int] | None = None,
                     force_ocr: bool = False, reindex: bool = True, progress=None) -> dict:
    t0 = time.time()
    stats = {"pages": 0, "regions": 0, "tables": 0, "facts": 0, "scanned_pages": 0, "narrative_facts": 0}
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            raise ValueError("document not found")
        seed_entities(session)
        doc.status = "processing"; doc.error = None
        audit(session, "process_start", None, "document", document_id)
        session.commit()
        # clear previous derived data (idempotent reprocessing)
        _clear_derived(session, document_id)
        session.commit()
        try:
            pdf = pymupdf.open(doc.storage_location)
        except Exception as e:
            doc.status = "failed"; doc.error = f"cannot open: {e}"
            return stats
        n = pdf.page_count
        idxs = list(range(n))
        if page_range:
            idxs = [i for i in idxs if page_range[0] - 1 <= i <= page_range[1] - 1]
        if max_pages and max_pages > 0:
            idxs = idxs[:max_pages]
        doc_ctx = DocContext(document_id=doc.document_id, doc_code=doc.code, doc_type=doc.document_type, period=doc.period,
                             default_unit="million tonnes", title=doc.title or "")
        resolver = Resolver(session)
        all_tables: list[Table] = []
        drafts_all = []
        asof_carry = ""
        try:
            for k, i in enumerate(idxs):
                pr = docai_pages.analyze_page(pdf, i, force_ocr=force_ocr)
                m_asof = re.search(r"as\s*on\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4})", pr.text or "", re.I)
                if m_asof:
                    asof_carry = m_asof.group(0)
                page_ctx = (pr.text or "") + ("\n" + asof_carry if asof_carry and not m_asof else "")
                page, table_regions = _persist_page(session, doc, pr)
                stats["pages"] += 1
                stats["regions"] += len(pr.regions)
                stats["tables"] += len(pr.tables)
                stats["scanned_pages"] += int(pr.is_scanned)
                # facts from tables
                for reg, tab, cells_by_pos in table_regions:
                    all_tables.append(tab)
                    intro = _intro_text(pr.regions, reg.bbox)
                    drafts = facts_from_table(tab.headers, tab.rows, cells_by_pos, tab.caption or "", tab.unit_hint or "", doc_ctx, resolver,
                                              tab.confidence or 0.8, reg.confidence or 0.9, tab.table_id, reg.region_id, intro=intro, page_text=page_ctx)
                    for d in drafts:
                        drafts_all.append((d, page))
                # facts from narrative regions
                for r in pr.regions:
                    if r.type in ("text", "caption", "title") and r.text and len(r.text) > 40:
                        regrow = session.query(Region).filter(Region.page_id == page.page_id, Region.reading_order == r.reading_order).first()
                        nd = facts_from_text(r.text, r.bbox, regrow.region_id if regrow else None, doc_ctx, resolver, r.confidence)
                        stats["narrative_facts"] += len(nd)
                        for d in nd:
                            drafts_all.append((d, page))
                if progress:
                    progress(k + 1, len(idxs), pr)
                if (k + 1) % 5 == 0:
                    session.commit()
            _group_multipage_tables(session, all_tables)
            # persist facts
            facts: list[Fact] = []
            for d, page in drafts_all:
                f = Fact(code=next_code(session, "fact", "F", 5), subject_entity_id=d.subject_entity_id, subject_text=d.subject_text,
                         predicate=d.predicate, qualifier=d.qualifier, period=d.period, period_scope=d.period_scope, value=d.value, unit=d.unit,
                         dimension_entity_id=d.dimension_entity_id, dimension_text=d.dimension_text,
                         display_value=d.display_value, original_value=d.original_value, original_unit=d.original_unit,
                         document_id=doc.document_id, page_id=page.page_id, page_number=page.page_number, region_id=d.region_id,
                         table_id=d.table_id, cell_id=d.cell_id, bbox=d.bbox, context=d.context, extraction_method=d.extraction_method,
                         confidence={**d.confidence, "_agg_level": d.aggregate_level, "_row": d.row_index, "_col": d.col_index,
                                     **({"_partial_total": True} if (d.meta or {}).get("partial_total") else {})},
                         validation_status="pending")
                f._meta = d.meta
                session.add(f); facts.append(f)
            session.flush()
            stats["facts"] = len(facts)
            # validation
            validation.validate_document_facts(session, doc, facts)
            by_table = defaultdict(list)
            for f in facts:
                if f.table_id:
                    by_table[f.table_id].append(f)
            validation.arithmetic_checks(session, by_table)
            validation.finalize_confidence(session, facts)
            doc.status = "extracted"
            session.commit()
            # cross-document validation (only groups touching this document's subjects)
            subj = {f.subject_entity_id for f in facts if f.subject_entity_id}
            xstats = validation.cross_document_validation(session, subj) if subj else {}
            affected = session.query(Fact).filter(Fact.subject_entity_id.in_(list(subj))).all() if subj else []
            validation.finalize_confidence(session, affected)
            doc.status = "validated"
            stats["cross_document"] = xstats
            session.commit()
            if reindex:
                retrieval_index.index_document(session, doc)
            doc.meta = {**(doc.meta or {}), "stats": stats, "processed_at": dt.datetime.utcnow().isoformat(), "seconds": round(time.time() - t0, 1)}
            audit(session, "process_done", None, "document", document_id, detail=stats)
        except Exception as e:
            session.rollback()
            doc = session.get(Document, document_id)
            doc.status = "failed"; doc.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
            audit(session, "process_failed", None, "document", document_id, detail={"error": str(e)})
            raise
        finally:
            pdf.close()
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def _clear_derived(session, document_id: str) -> None:
    from sqlalchemy import delete
    from .db import Chunk, Validation as V, Conflict
    fact_ids = [f.fact_id for f in session.query(Fact.fact_id).filter(Fact.document_id == document_id).all()]
    if fact_ids:
        for i in range(0, len(fact_ids), 500):
            session.execute(delete(V).where(V.fact_id.in_(fact_ids[i:i + 500])))
        session.execute(delete(Fact).where(Fact.document_id == document_id))
    table_ids = [t.table_id for t in session.query(Table.table_id).filter(Table.document_id == document_id).all()]
    if table_ids:
        for i in range(0, len(table_ids), 500):
            session.execute(delete(Cell).where(Cell.table_id.in_(table_ids[i:i + 500])))
        session.execute(delete(Table).where(Table.document_id == document_id))
    session.execute(delete(Region).where(Region.document_id == document_id))
    session.execute(delete(Page).where(Page.document_id == document_id))
    retrieval_index.delete_document(session, document_id)
    session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    # conflicts referencing removed facts are refreshed by the next cross-document pass
    for c in session.query(Conflict).filter(Conflict.status == "open").all():
        if any(fid in fact_ids for fid in (c.fact_ids or [])):
            c.status = "resolved"; c.resolution = "auto: source document reprocessed"


# --------------------------------------------------------------------------- job runner

def run_pending_jobs(limit: int = 1, progress=None) -> int:
    done = 0
    for _ in range(limit):
        with session_scope() as session:
            job = session.query(Job).filter(Job.status == "queued").order_by(Job.created_at).first()
            if job is None:
                return done
            job.status = "running"; job.attempts = (job.attempts or 0) + 1
            job_id, doc_id, params, attempts = job.job_id, job.document_id, dict(job.params or {}), job.attempts
        try:
            pr = params.get("page_range")
            process_document(doc_id, max_pages=params.get("max_pages"), page_range=tuple(pr) if pr else None,
                             force_ocr=bool(params.get("force_ocr")), progress=progress)
            with session_scope() as session:
                job = session.get(Job, job_id); job.status = "done"
        except Exception as e:
            with session_scope() as session:
                job = session.get(Job, job_id)
                job.error = {"error": str(e), "trace": traceback.format_exc()[-2000:]}
                job.status = "queued" if attempts < 3 else "dead"     # retry with backoff, then dead-letter
            if attempts < 3:
                time.sleep(min(30, 2 ** attempts))
        done += 1
    return done
