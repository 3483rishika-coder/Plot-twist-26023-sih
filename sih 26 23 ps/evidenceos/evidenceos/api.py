"""FastAPI application — Plane 5 (application + governance). Serves the REST API (PRD §14) and the web UI."""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func

from . import analytics, cache, config, pipeline, reports, store, topics
from .db import (AuditLog, Cell, Conflict, Correction, Document, Entity, Fact, Job, Page, Region, Report, Table, User, Validation,
                 init_db, session_scope, SessionLocal)
from .docai.normalize import to_display
from .integrations import apisetu_client, bhashini_client, datagov_client, digilocker_client, get_all_integrations_status
from .knowledge import validation as validation_mod
from .knowledge.entities import seed_entities
from .retrieval import qa

app = FastAPI(title="Koyla AI API", version="0.1.0", description="Evidence-first AI reporting platform for CMPDI/Coal India (PS 26023)")

_worker_thread: Optional[threading.Thread] = None
_worker_state = {"running": False, "current": None, "progress": None}


# --------------------------------------------------------------------------- auth / RBAC

def seed_users(session):
    if session.query(User).count():
        return
    for role, key in config.DEMO_API_KEYS.items():
        session.add(User(username=role, role=role, api_key=key, department="CIL HQ" if role in ("admin", "officer") else "CMPDI"))
    session.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(x_api_key: Optional[str] = Header(default=None), k: Optional[str] = Query(default=None), db=Depends(get_db)) -> User:
    # header for API clients; `?k=` query fallback for <img>/<a download> tags in the browser UI
    key = x_api_key or k or config.DEMO_API_KEYS["viewer"]
    u = db.query(User).filter(User.api_key == key).first()
    if u is None:
        raise HTTPException(401, "invalid API key")
    return u


def require(perm: str):
    def dep(user: User = Depends(current_user)):
        perms = config.ROLE_PERMISSIONS.get(user.role, set())
        if "*" in perms or perm in perms:
            return user
        raise HTTPException(403, f"role '{user.role}' lacks permission '{perm}'")
    return dep


def _visible(db, user: User):
    max_lvl = config.ROLE_MAX_ACL.get(user.role, 0)
    return [d for d in db.query(Document).order_by(Document.created_at).all() if config.ACL_ORDER.get(d.acl_level or "public", 0) <= max_lvl]


def _doc_or_404(db, user, document_id) -> Document:
    d = db.get(Document, document_id)
    if d is None:
        raise HTTPException(404, "document not found")
    if config.ACL_ORDER.get(d.acl_level or "public", 0) > config.ROLE_MAX_ACL.get(user.role, 0):
        raise HTTPException(403, "document ACL denies access")
    return d


# --------------------------------------------------------------------------- lifecycle / worker

def _worker_loop():
    _worker_state["running"] = True
    while True:
        try:
            def prog(k, n, pr):
                _worker_state["progress"] = {"page": k, "pages": n, "scanned": pr.is_scanned}
            n = pipeline.run_pending_jobs(limit=1, progress=prog)
            if n == 0:
                time.sleep(2.0)
        except Exception:
            time.sleep(3.0)


@app.on_event("startup")
def _startup():
    init_db()
    with session_scope() as s:
        seed_entities(s); seed_users(s)
        # requeue jobs interrupted by a restart
        for j in s.query(Job).filter(Job.status == "running").all():
            j.status = "queued"
    global _worker_thread
    if os.environ.get("EVIDENCEOS_WORKER", "1") != "0" and _worker_thread is None:
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="evidenceos-worker")
        _worker_thread.start()


# --------------------------------------------------------------------------- documents

@app.post("/api/v1/documents/upload")
async def upload(file: UploadFile = File(...), title: str = Form(None), source: str = Form(None), document_type: str = Form(None),
                 period: str = Form(None), acl_level: str = Form("public"), department: str = Form(None), max_pages: int = Form(0),
                 user: User = Depends(require("upload")), db=Depends(get_db)):
    if not file.filename.lower().endswith((".pdf",)):
        raise HTTPException(415, "only PDF is supported in this build (Excel/Word: convert to PDF or use the CLI converter)")
    tmpdir = Path(tempfile.mkdtemp(prefix="eos-up-"))
    tmp = tmpdir / Path(file.filename).name
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    doc, created = pipeline.ingest_file(db, tmp, title=title, source=source, doc_type=document_type, period=period, acl_level=acl_level,
                                        department=department, user=user, job_params={"max_pages": max_pages or None})
    db.commit()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return {"document_id": doc.document_id, "code": doc.code, "sha256": doc.sha256, "status": doc.status, "duplicate": not created,
            "page_count": doc.page_count, "is_scanned": doc.is_scanned}


@app.get("/api/v1/documents")
def list_documents(user: User = Depends(require("read")), db=Depends(get_db)):
    out = []
    for d in _visible(db, user):
        nf = db.query(func.count(Fact.fact_id)).filter(Fact.document_id == d.document_id).scalar()
        nt = db.query(func.count(Table.table_id)).filter(Table.document_id == d.document_id).scalar()
        npg = db.query(func.count(Page.page_id)).filter(Page.document_id == d.document_id).scalar()
        job = db.query(Job).filter(Job.document_id == d.document_id).order_by(Job.created_at.desc()).first()
        out.append(_doc_json(d) | {"facts": nf, "tables": nt, "pages_processed": npg, "job": {"status": job.status, "attempts": job.attempts, "params": job.params} if job else None})
    return out


def _doc_json(d: Document):
    return {"document_id": d.document_id, "code": d.code, "title": d.title, "filename": d.original_filename, "sha256": d.sha256, "source": d.source,
            "source_url": d.source_url, "document_type": d.document_type, "period": d.period, "version": d.version, "family_key": d.family_key,
            "page_count": d.page_count, "is_scanned": d.is_scanned, "scan_ratio": d.scan_ratio, "acl_level": d.acl_level, "status": d.status,
            "error": d.error, "file_size": d.file_size, "created_at": d.created_at.isoformat() if d.created_at else None, "meta": d.meta}


@app.get("/api/v1/documents/{document_id}")
def get_document(document_id: str, user: User = Depends(require("read")), db=Depends(get_db)):
    d = _doc_or_404(db, user, document_id)
    pipeline.audit(db, "view_document", user, "document", document_id, document_id=document_id); db.commit()
    return _doc_json(d)


@app.get("/api/v1/documents/{document_id}/status")
def doc_status(document_id: str, user: User = Depends(require("read")), db=Depends(get_db)):
    d = _doc_or_404(db, user, document_id)
    job = db.query(Job).filter(Job.document_id == document_id).order_by(Job.created_at.desc()).first()
    return {"document_id": document_id, "status": d.status, "error": d.error, "job": {"status": job.status, "attempts": job.attempts, "error": job.error} if job else None,
            "worker": _worker_state["progress"] if _worker_state.get("current") in (None, document_id) else None,
            "pages_processed": db.query(func.count(Page.page_id)).filter(Page.document_id == document_id).scalar(), "page_count": d.page_count}


@app.post("/api/v1/documents/{document_id}/reprocess")
def reprocess(document_id: str, max_pages: int = 0, page_from: int = 0, page_to: int = 0, force_ocr: bool = False,
              user: User = Depends(require("upload")), db=Depends(get_db)):
    _doc_or_404(db, user, document_id)
    params = {"max_pages": max_pages or None, "force_ocr": force_ocr}
    if page_from and page_to:
        params["page_range"] = [page_from, page_to]
    db.add(Job(document_id=document_id, stage="process", status="queued", params=params)); db.commit()
    return {"queued": True, "params": params}


@app.get("/api/v1/documents/{document_id}/pages")
def doc_pages(document_id: str, user: User = Depends(require("read")), db=Depends(get_db)):
    _doc_or_404(db, user, document_id)
    pages = db.query(Page).filter(Page.document_id == document_id).order_by(Page.page_number).all()
    return [{"page_id": p.page_id, "page_number": p.page_number, "width": p.width, "height": p.height, "is_scanned": p.is_scanned,
             "quality": p.quality, "ocr_confidence": p.ocr_confidence, "n_regions": len(p.regions),
             "n_tables": db.query(func.count(Table.table_id)).filter(Table.page_id == p.page_id).scalar(),
             "n_facts": db.query(func.count(Fact.fact_id)).filter(Fact.page_id == p.page_id).scalar()} for p in pages]


@app.get("/api/v1/documents/{document_id}/pages/{page_number}")
def doc_page(document_id: str, page_number: int, user: User = Depends(require("read")), db=Depends(get_db)):
    d = _doc_or_404(db, user, document_id)
    p = db.query(Page).filter(Page.document_id == document_id, Page.page_number == page_number).first()
    if p is None:
        return {"document_id": document_id, "page_number": page_number, "processed": False, "page_count": d.page_count, "regions": [], "tables": [], "facts": []}
    regions = db.query(Region).filter(Region.page_id == p.page_id).order_by(Region.reading_order).all()
    tables = db.query(Table).filter(Table.page_id == p.page_id).all()
    facts = db.query(Fact).filter(Fact.page_id == p.page_id).all()
    ents = {e.entity_id: e for e in db.query(Entity).filter(Entity.entity_id.in_({f.subject_entity_id for f in facts if f.subject_entity_id})).all()} if facts else {}
    pipeline.audit(db, "view_page", user, "page", p.page_id, document_id=document_id, detail={"page": page_number}); db.commit()
    return {"document_id": document_id, "document_code": d.code, "title": d.title, "page_number": page_number, "page_id": p.page_id, "processed": True,
            "page_count": d.page_count, "width": p.width, "height": p.height, "is_scanned": p.is_scanned, "quality": p.quality, "ocr_confidence": p.ocr_confidence,
            "text": p.text, "image_url": f"/api/v1/pages/{p.page_id}/image",
            "regions": [{"region_id": r.region_id, "type": r.type, "bbox": r.bbox, "reading_order": r.reading_order, "confidence": r.confidence, "text": r.text, "meta": r.meta} for r in regions],
            "tables": [_table_json(db, t) for t in tables],
            "facts": [_fact_json(f, d, ents.get(f.subject_entity_id)) for f in facts]}


@app.get("/api/v1/pages/{page_id}/image")
def page_image(page_id: str, dpi: int = 0, user: User = Depends(require("read")), db=Depends(get_db)):
    p = db.get(Page, page_id)
    if p is None:
        raise HTTPException(404)
    d = _doc_or_404(db, user, p.document_id)
    path = store.render_page(d.storage_location, d.document_id, p.page_number, dpi or config.VIEW_DPI)
    return FileResponse(str(path), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/v1/pages/{page_id}/regions")
def page_regions(page_id: str, user: User = Depends(require("read")), db=Depends(get_db)):
    p = db.get(Page, page_id)
    if p is None:
        raise HTTPException(404)
    _doc_or_404(db, user, p.document_id)
    return [{"region_id": r.region_id, "type": r.type, "bbox": r.bbox, "reading_order": r.reading_order, "confidence": r.confidence, "text": r.text, "meta": r.meta}
            for r in db.query(Region).filter(Region.page_id == page_id).order_by(Region.reading_order).all()]


@app.get("/api/v1/regions/{region_id}/crop")
def region_crop(region_id: str, user: User = Depends(require("read")), db=Depends(get_db)):
    r = db.get(Region, region_id)
    if r is None:
        raise HTTPException(404)
    d = _doc_or_404(db, user, r.document_id)
    p = db.get(Page, r.page_id)
    path = store.render_crop(d.storage_location, d.document_id, p.page_number, r.bbox, f"region-{region_id}")
    return FileResponse(str(path), media_type="image/png")


def _table_json(db, t: Table):
    cells = db.query(Cell).filter(Cell.table_id == t.table_id).all()
    return {"table_id": t.table_id, "region_id": t.region_id, "page_id": t.page_id, "table_group_id": t.table_group_id, "page_sequence": t.page_sequence,
            "caption": t.caption, "headers": t.headers, "rows": t.rows, "n_rows": t.n_rows, "n_cols": t.n_cols, "header_rows": t.header_rows,
            "unit_hint": t.unit_hint, "method": t.method, "confidence": t.confidence,
            "cells": [{"cell_id": c.cell_id, "row": c.row_index, "col": c.col_index, "bbox": c.bbox, "text": c.text, "confidence": c.confidence} for c in cells]}


@app.get("/api/v1/tables/{table_id}")
def get_table(table_id: str, user: User = Depends(require("read")), db=Depends(get_db)):
    t = db.get(Table, table_id)
    if t is None:
        raise HTTPException(404)
    _doc_or_404(db, user, t.document_id)
    facts = db.query(Fact).filter(Fact.table_id == table_id).all()
    d = db.get(Document, t.document_id)
    return _table_json(db, t) | {"facts": [_fact_json(f, d, db.get(Entity, f.subject_entity_id)) for f in facts]}


# --------------------------------------------------------------------------- facts / validation / conflicts

def _fact_json(f: Fact, d: Document | None = None, e: Entity | None = None):
    return {"fact_id": f.fact_id, "code": f.code, "subject": e.canonical_name if e else f.subject_text, "subject_text": f.subject_text,
            "subject_type": e.entity_type if e else None, "subject_entity_id": f.subject_entity_id, "predicate": f.predicate, "dimension": f.dimension_text, "qualifier": f.qualifier,
            "period": f.period, "period_scope": f.period_scope, "value": f.value, "unit": f.unit, "display": to_display(f.value, f.unit),
            "display_value": f.display_value, "original_value": f.original_value, "original_unit": f.original_unit,
            "document_id": f.document_id, "document_code": d.code if d else None, "document_title": d.title if d else None, "page_number": f.page_number,
            "page_id": f.page_id, "region_id": f.region_id, "table_id": f.table_id, "cell_id": f.cell_id, "bbox": f.bbox, "context": f.context,
            "extraction_method": f.extraction_method, "confidence": {k: v for k, v in (f.confidence or {}).items() if not k.startswith("_")},
            "overall_confidence": f.overall_confidence, "validation_status": f.validation_status, "review_status": f.review_status,
            "crop_url": f"/api/v1/facts/{f.fact_id}/crop", "viewer_url": f"/#/viewer/{f.document_id}/{f.page_number}?region={f.region_id or ''}&fact={f.fact_id}"}


@app.get("/api/v1/facts")
def list_facts(subject: Optional[str] = None, predicate: Optional[str] = None, period: Optional[str] = None, document_id: Optional[str] = None,
               status: Optional[str] = None, review: Optional[str] = None, min_conf: float = 0.0, limit: int = 200, offset: int = 0,
               user: User = Depends(require("read")), db=Depends(get_db)):
    allowed = {d.document_id for d in _visible(db, user)}
    q = db.query(Fact).filter(Fact.document_id.in_(list(allowed)))
    if predicate:
        q = q.filter(Fact.predicate.like(f"{predicate}%"))
    if period:
        q = q.filter(Fact.period.like(f"{period}%"))
    if document_id:
        q = q.filter(Fact.document_id == document_id)
    if status:
        q = q.filter(Fact.validation_status == status)
    if review:
        q = q.filter(Fact.review_status == review)
    if min_conf:
        q = q.filter(Fact.overall_confidence >= min_conf)
    if subject:
        ids = [e.entity_id for e in db.query(Entity).filter(Entity.canonical_name.ilike(f"%{subject}%")).all()]
        q = q.filter((Fact.subject_entity_id.in_(ids)) | (Fact.subject_text.ilike(f"%{subject}%")))
    total = q.count()
    facts = q.order_by(Fact.overall_confidence.asc().nullsfirst() if review else Fact.created_at.desc()).offset(offset).limit(limit).all()
    docs = {d.document_id: d for d in db.query(Document).filter(Document.document_id.in_({f.document_id for f in facts})).all()} if facts else {}
    ents = {e.entity_id: e for e in db.query(Entity).filter(Entity.entity_id.in_({f.subject_entity_id for f in facts if f.subject_entity_id})).all()} if facts else {}
    return {"total": total, "items": [_fact_json(f, docs.get(f.document_id), ents.get(f.subject_entity_id)) for f in facts]}


@app.get("/api/v1/facts/{fact_id}")
def get_fact(fact_id: str, user: User = Depends(require("read")), db=Depends(get_db)):
    f = db.get(Fact, fact_id)
    if f is None:
        raise HTTPException(404)
    d = _doc_or_404(db, user, f.document_id)
    vals = db.query(Validation).filter(Validation.fact_id == fact_id).order_by(Validation.created_at).all()
    conflicts = [c for c in db.query(Conflict).filter(Conflict.subject_entity_id == f.subject_entity_id, Conflict.period == f.period).all() if fact_id in (c.fact_ids or [])]
    corr = db.query(Correction).filter(Correction.fact_id == fact_id).order_by(Correction.created_at).all()
    pipeline.audit(db, "view_fact", user, "fact", fact_id, document_id=f.document_id); db.commit()
    return _fact_json(f, d, db.get(Entity, f.subject_entity_id)) | {
        "evidence": {"document_id": f.document_id, "document_code": d.code, "title": d.title, "sha256": d.sha256, "page": f.page_number, "region_id": f.region_id,
                     "table_id": f.table_id, "cell_id": f.cell_id, "bbox": f.bbox, "crop_url": f"/api/v1/facts/{f.fact_id}/crop", "page_image_url": f"/api/v1/pages/{f.page_id}/image"},
        "validations": [{"rule_id": v.rule_id, "rule_type": v.rule_type, "status": v.status, "message": v.message, "evidence": v.evidence} for v in vals],
        "conflicts": [{"conflict_id": c.conflict_id, "code": c.code, "values": c.values, "severity": c.severity, "status": c.status, "explanation": c.explanation} for c in conflicts],
        "corrections": [{"action": c.action, "old_value": c.old_value, "new_value": c.new_value, "note": c.note, "at": c.created_at.isoformat()} for c in corr]}


@app.get("/api/v1/facts/{fact_id}/crop")
def fact_crop(fact_id: str, user: User = Depends(require("read")), db=Depends(get_db)):
    f = db.get(Fact, fact_id)
    if f is None:
        raise HTTPException(404)
    d = _doc_or_404(db, user, f.document_id)
    bbox = f.bbox
    if not bbox and f.region_id:
        r = db.get(Region, f.region_id); bbox = r.bbox if r else None
    if not bbox:
        raise HTTPException(404, "no bbox")
    # widen tiny cell boxes to the whole table row for context
    if f.table_id and f.cell_id:
        t = db.get(Table, f.table_id)
        rb = (t.meta or {}).get("row_bboxes") or []
        row = (f.confidence or {}).get("_row")
        if row is not None and row < len(rb):
            bbox = [rb[row][0], min(bbox[1], rb[row][1]), rb[row][2], max(bbox[3], rb[row][3])]
    path = store.render_crop(d.storage_location, d.document_id, f.page_number, bbox, f"fact-{fact_id}")
    return FileResponse(str(path), media_type="image/png")


@app.get("/api/v1/conflicts")
def list_conflicts(status: str = "open", user: User = Depends(require("read")), db=Depends(get_db)):
    allowed = {d.document_id for d in _visible(db, user)}
    q = db.query(Conflict)
    if status != "all":
        q = q.filter(Conflict.status == status)
    out = []
    for c in q.order_by(Conflict.severity.desc(), Conflict.created_at.desc()).all():
        if any(v.get("document_id") not in allowed for v in (c.values or [])):
            continue
        out.append({"conflict_id": c.conflict_id, "code": c.code, "subject": c.subject_text, "subject_entity_id": c.subject_entity_id, "predicate": c.predicate,
                    "period": c.period, "unit": c.unit, "values": c.values, "severity": c.severity, "status": c.status, "resolution": c.resolution,
                    "explanation": c.explanation, "fact_ids": c.fact_ids})
    return out


@app.post("/api/v1/conflicts/{conflict_id}/resolve")
def resolve_conflict(conflict_id: str, body: dict, user: User = Depends(require("review")), db=Depends(get_db)):
    c = db.get(Conflict, conflict_id)
    if c is None:
        raise HTTPException(404)
    action = body.get("action", "resolve")
    c.status = "ignored" if action == "ignore" else "resolved"
    c.resolution = body.get("resolution") or f"{action} by {user.username}"
    if body.get("preferred_fact_id"):
        c.resolved_fact_id = body["preferred_fact_id"]
        for fid in c.fact_ids or []:
            f = db.get(Fact, fid)
            if f:
                f.validation_status = "verified" if fid == body["preferred_fact_id"] else "rejected"
                f.review_status = "done"
                db.add(Correction(fact_id=fid, user_id=user.user_id, action="accept" if fid == body["preferred_fact_id"] else "reject",
                                  old_value=f.display_value, new_value=f.display_value, note=f"conflict {c.code} resolution"))
    pipeline.audit(db, "resolve_conflict", user, "conflict", conflict_id, detail=body); db.commit()
    return {"ok": True, "status": c.status}


@app.post("/api/v1/review/{fact_id}")
def review_fact(fact_id: str, body: dict, user: User = Depends(require("review")), db=Depends(get_db)):
    f = db.get(Fact, fact_id)
    if f is None:
        raise HTTPException(404)
    action = body.get("action")
    if action not in ("accept", "edit", "reject"):
        raise HTTPException(400, "action must be accept|edit|reject")
    old = f.display_value
    if action == "accept":
        f.validation_status = "verified"
    elif action == "reject":
        f.validation_status = "rejected"
    else:
        val = body.get("value")
        if val is None:
            raise HTTPException(400, "value required for edit")
        scale = {"tonnes": 1e6, "cubic_metres": 1e6, "inr": 1e7}.get(f.unit, 1.0)
        f.value = float(val) * scale
        f.display_value = f"{float(val):g}"
        f.validation_status = "verified"
    f.review_status = "done"
    conf = dict(f.confidence or {}); conf["human_verified"] = action != "reject"; conf["validation"] = 1.0 if action != "reject" else 0.0
    f.confidence = conf
    f.overall_confidence = 1.0 if action != "reject" else 0.0
    db.add(Correction(fact_id=fact_id, user_id=user.user_id, action=action, old_value=old, new_value=f.display_value, note=body.get("note")))
    pipeline.audit(db, f"review_{action}", user, "fact", fact_id, document_id=f.document_id, detail={"old": old, "new": f.display_value})
    db.commit()
    # refresh cross-document comparison for this subject
    validation_mod.cross_document_validation(db, [f.subject_entity_id] if f.subject_entity_id else None)
    affected = db.query(Fact).filter(Fact.subject_entity_id == f.subject_entity_id).all() if f.subject_entity_id else [f]
    validation_mod.finalize_confidence(db, affected)
    db.commit()
    return {"ok": True, "fact": _fact_json(f, db.get(Document, f.document_id), db.get(Entity, f.subject_entity_id))}


@app.get("/api/v1/review/queue")
def review_queue(limit: int = 100, user: User = Depends(require("read")), db=Depends(get_db)):
    allowed = {d.document_id for d in _visible(db, user)}
    q = db.query(Fact).filter(Fact.document_id.in_(list(allowed)), Fact.review_status.in_(["mandatory", "recommended"]), Fact.validation_status.notin_(["verified", "rejected"]))
    facts = q.order_by(Fact.overall_confidence.asc()).limit(limit).all()
    docs = {d.document_id: d for d in db.query(Document).all()}
    ents = {e.entity_id: e for e in db.query(Entity).all()}
    return {"total": q.count(), "items": [_fact_json(f, docs.get(f.document_id), ents.get(f.subject_entity_id)) for f in facts]}


# --------------------------------------------------------------------------- query / reports / topics

@app.post("/api/v1/query")
def query(body: dict, user: User = Depends(require("query")), db=Depends(get_db)):
    q = (body.get("query") or "").strip()
    if not q:
        raise HTTPException(400, "query required")
    t0 = time.time()
    res = qa.answer(db, q, user=user, filters=body.get("filters") or {})
    res["elapsed_ms"] = int((time.time() - t0) * 1000)
    pipeline.audit(db, "query", user, query=q, detail={"route": res.get("route"), "confidence": res.get("confidence"), "n_citations": len(res.get("citations", [])),
                                                        "documents": sorted({c.get("document_code") for c in res.get("citations", []) if c.get("document_code")}),
                                                        "answer": (res.get("answer") or "")[:500]})
    db.commit()
    return res


@app.get("/api/v1/reports/templates")
def report_templates(user: User = Depends(require("read"))):
    return [{"template_id": k, **v} for k, v in reports.TEMPLATES.items()]


@app.post("/api/v1/reports/generate")
def generate_report(body: dict, user: User = Depends(require("report")), db=Depends(get_db)):
    tpl = body.get("template_id")
    params = body.get("params") or {}
    rep = reports.generate(db, tpl, params, user=user)
    pipeline.audit(db, "generate_report", user, "report", rep.report_id, detail={"template": tpl, "params": params, "facts_used": len(rep.facts_used or [])})
    db.commit()
    return {"report_id": rep.report_id, "title": rep.title, "download_url": f"/api/v1/reports/{rep.report_id}", "facts_used": len(rep.facts_used or []),
            "missing": rep.missing, "status": rep.status}


@app.get("/api/v1/reports")
def list_reports(user: User = Depends(require("read")), db=Depends(get_db)):
    return [{"report_id": r.report_id, "title": r.title, "template_id": r.template_id, "status": r.status, "facts_used": len(r.facts_used or []),
             "missing": r.missing, "created_at": r.created_at.isoformat(), "download_url": f"/api/v1/reports/{r.report_id}"}
            for r in db.query(Report).order_by(Report.created_at.desc()).all()]


@app.get("/api/v1/reports/{report_id}")
def download_report(report_id: str, user: User = Depends(require("read")), db=Depends(get_db)):
    r = db.get(Report, report_id)
    if r is None or not r.document_path or not Path(r.document_path).exists():
        raise HTTPException(404)
    pipeline.audit(db, "download_report", user, "report", report_id); db.commit()
    return FileResponse(r.document_path, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        filename=f"{r.title.replace(' ', '_')[:60]}.docx")


@app.get("/api/v1/topics")
def get_topics(year: Optional[str] = None, doc_type: Optional[str] = None, source: Optional[str] = None, user: User = Depends(require("read")), db=Depends(get_db)):
    filters = {k: v for k, v in {"year": year, "doc_type": doc_type, "source": source}.items() if v}
    return topics.compute(db, filters)


# --------------------------------------------------------------------------- admin / audit / stats

@app.get("/api/v1/audit")
def audit_log(limit: int = 200, action: Optional[str] = None, user: User = Depends(require("audit")), db=Depends(get_db)):
    q = db.query(AuditLog)
    if action:
        q = q.filter(AuditLog.action == action)
    return [{"audit_id": a.audit_id, "user": a.username, "action": a.action, "entity_type": a.entity_type, "entity_id": a.entity_id, "query": a.query,
             "document": a.document_accessed, "detail": a.detail, "timestamp": a.timestamp.isoformat()} for a in q.order_by(AuditLog.timestamp.desc()).limit(limit).all()]


@app.get("/api/v1/stats")
def stats(user: User = Depends(require("read")), db=Depends(get_db)):
    allowed = [d.document_id for d in _visible(db, user)]
    fq = db.query(Fact).filter(Fact.document_id.in_(allowed))
    by_status = dict(db.query(Fact.validation_status, func.count(Fact.fact_id)).filter(Fact.document_id.in_(allowed)).group_by(Fact.validation_status).all())
    by_review = dict(db.query(Fact.review_status, func.count(Fact.fact_id)).filter(Fact.document_id.in_(allowed)).group_by(Fact.review_status).all())
    return {"documents": len(allowed), "pages": db.query(func.count(Page.page_id)).filter(Page.document_id.in_(allowed)).scalar(),
            "scanned_pages": db.query(func.count(Page.page_id)).filter(Page.document_id.in_(allowed), Page.is_scanned == True).scalar(),
            "regions": db.query(func.count(Region.region_id)).filter(Region.document_id.in_(allowed)).scalar(),
            "tables": db.query(func.count(Table.table_id)).filter(Table.document_id.in_(allowed)).scalar(),
            "facts": fq.count(), "facts_by_status": by_status, "facts_by_review": by_review,
            "conflicts_open": db.query(func.count(Conflict.conflict_id)).filter(Conflict.status == "open").scalar(),
            "entities": db.query(func.count(Entity.entity_id)).scalar(),
            "jobs": dict(db.query(Job.status, func.count(Job.job_id)).group_by(Job.status).all()),
            "worker": _worker_state["progress"], "llm_configured": bool(config.LLM_API_KEY or config.LLM_BASE_URL),
            "llm_provider": config.LLM_PROVIDER, "llm_model": config.LLM_MODEL,
            "redis": cache.get_redis_status(), "minio": store.get_minio_status(),
            "bhashini_enabled": config.BHASHINI_ENABLED,
            "embeddings": config.EMBEDDINGS_ENABLED, "user": {"username": user.username, "role": user.role}}


# --------------------------------------------------------------------------- Integrations & Tabular Analytics

@app.get("/api/v1/integrations/status")
def get_integrations(user: User = Depends(require("read"))):
    """ Consolidated health and connectivity status of Bhashini, API Setu, DigiLocker, data.gov.in, Redis, and MinIO."""
    return get_all_integrations_status()


@app.post("/api/v1/integrations/bhashini/ocr")
async def bhashini_ocr(file: UploadFile = File(...), lang: str = Form("hi"), user: User = Depends(require("upload"))):
    """Run Bhashini Indic OCR on an uploaded regional document / page image."""
    content = await file.read()
    return bhashini_client.ocr(content, source_language=lang)


@app.post("/api/v1/integrations/bhashini/translate")
def bhashini_translate(payload: dict, user: User = Depends(require("read"))):
    """Translate Indic mining circulars or regional text to English via Bhashini NMT."""
    text = payload.get("text", "")
    src = payload.get("source_language", "hi")
    tgt = payload.get("target_language", "en")
    translated = bhashini_client.translate(text, source_language=src, target_language=tgt)
    return {"original": text, "translated": translated, "source_language": src, "target_language": tgt}


@app.get("/api/v1/integrations/apisetu/leases")
def apisetu_leases(state: Optional[str] = None, company: Optional[str] = None, user: User = Depends(require("read"))):
    """Fetch verified coal lease and block records from API Setu Ministry of Coal gateway."""
    return apisetu_client.fetch_coal_leases(state=state, company=company)


@app.post("/api/v1/integrations/digilocker/import")
def digilocker_import(payload: dict, user: User = Depends(require("upload")), db=Depends(get_db)):
    """Pull verified document from DigiLocker, verify digital signature, and ingest into EvidenceOS."""
    uri = payload.get("uri", "")
    if not uri:
        raise HTTPException(400, "Missing 'uri' parameter")
    return digilocker_client.import_document_to_evidenceos(uri, user, db)


@app.post("/api/v1/integrations/datagov/sync")
def datagov_sync(dataset: str = Query("production"), period: str = Query("FY2024-25"), user: User = Depends(require("upload")), db=Depends(get_db)):
    """Sync published datasets from data.gov.in (OGD Platform India) directly into verified facts."""
    return datagov_client.sync_to_facts(db, dataset_type=dataset, period=period)


@app.get("/api/v1/tables/{table_id}/export")
def export_table_endpoint(table_id: str, format: str = Query("csv"), user: User = Depends(require("read")), db=Depends(get_db)):
    """Export an extracted table as a pandas DataFrame in CSV, Excel (.xlsx), Parquet, or JSON format."""
    content, media_type, ext = analytics.export_table(db, table_id, fmt=format)
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="table_{table_id[:8]}.{ext}"'}
    )


@app.get("/api/v1/analytics/production-matrix")
def production_matrix_endpoint(period: Optional[str] = None, user: User = Depends(require("read")), db=Depends(get_db)):
    """Generate a dynamic pandas pivot table across subsidiaries and mining production/dispatch metrics."""
    return analytics.facts_to_production_matrix(db, period=period)


@app.get("/api/v1/entities")
def list_entities(q: Optional[str] = None, type: Optional[str] = None, user: User = Depends(require("read")), db=Depends(get_db)):
    qq = db.query(Entity)
    if q:
        qq = qq.filter(Entity.canonical_name.ilike(f"%{q}%"))
    if type:
        qq = qq.filter(Entity.entity_type == type)
    out = []
    for e in qq.order_by(Entity.entity_type, Entity.canonical_name).limit(500).all():
        out.append({"entity_id": e.entity_id, "name": e.canonical_name, "type": e.entity_type, "parent_entity_id": e.parent_entity_id, "meta": e.meta,
                    "facts": db.query(func.count(Fact.fact_id)).filter(Fact.subject_entity_id == e.entity_id).scalar()})
    return out


@app.get("/api/v1/me")
def me(user: User = Depends(current_user)):
    return {"username": user.username, "role": user.role, "permissions": sorted(config.ROLE_PERMISSIONS.get(user.role, set()))}


@app.get("/api/v1/roles")
def roles(user: User = Depends(require("read"))):
    # demo convenience: expose the seeded demo keys so evaluators can switch roles in the UI
    return [{"role": r, "api_key": k, "permissions": sorted(config.ROLE_PERMISSIONS.get(r, set())), "max_acl": config.ROLE_MAX_ACL.get(r)} for r, k in config.DEMO_API_KEYS.items()]


# --------------------------------------------------------------------------- UI

@app.get("/", response_class=HTMLResponse)
def ui_index():
    return (config.WEB_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")
