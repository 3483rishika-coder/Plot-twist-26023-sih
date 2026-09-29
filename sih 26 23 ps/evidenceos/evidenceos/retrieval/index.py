"""Hybrid retrieval index (PRD §12): multi-level chunks, SQLite FTS5 (BM25) + dense embeddings (fastembed) fused with RRF.

Chunk levels: document, section, page, table, fact.  Header/footer/page-number regions are excluded.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Optional

import numpy as np
from sqlalchemy import text as sql_text

from .. import config
from ..db import Chunk, Document, Fact, Page, Region, Table, Entity
from ..docai.normalize import to_display

_embedder = None
_lock = threading.Lock()


def get_embedder():
    global _embedder
    if not config.EMBEDDINGS_ENABLED:
        return None
    with _lock:
        if _embedder is None:
            try:
                import os
                os.environ.setdefault("FASTEMBED_CACHE_PATH", str(config.CACHE_DIR / "fastembed"))
                from fastembed import TextEmbedding
                _embedder = TextEmbedding(model_name=config.EMBEDDING_MODEL)
            except Exception:
                _embedder = False
    return _embedder or None


def embed(texts: list[str]) -> Optional[np.ndarray]:
    emb = get_embedder()
    if emb is None or not texts:
        return None
    texts = [t[:1200] for t in texts]
    vecs = np.array(list(emb.embed(texts, batch_size=8)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs / norms


def _window(text: str, size: int = 1100, overlap: int = 150) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            cut = text.rfind(". ", i, j)
            if cut > i + size // 2:
                j = cut + 1
        piece = text[i:j].strip()
        if piece:
            out.append(piece)
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return out


def build_chunks(session, doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    meta_base = {"doc_code": doc.code, "title": doc.title, "doc_type": doc.document_type, "period": doc.period, "source": doc.source, "acl": doc.acl_level}
    chunks.append(Chunk(level="document", document_id=doc.document_id, page_number=1,
                        text=f"{doc.title}. Publisher: {doc.source or 'unknown'}. Type: {doc.document_type}. Period: {doc.period or 'n/a'}.", meta=meta_base))
    pages = session.query(Page).filter(Page.document_id == doc.document_id).order_by(Page.page_number).all()
    for p in pages:
        regs = session.query(Region).filter(Region.page_id == p.page_id).order_by(Region.reading_order).all()
        body = [r for r in regs if r.type in ("text", "title", "caption", "table", "handwriting") and (r.text or "").strip()]
        # sections: title + following text
        cur_title = None
        for r in regs:
            if r.type == "title" and r.text:
                cur_title = r.text.strip()[:120]
            elif r.type == "text" and cur_title and r.text and len(r.text) > 200:
                chunks.append(Chunk(level="section", document_id=doc.document_id, page_id=p.page_id, page_number=p.page_number, region_id=r.region_id,
                                    text=f"{cur_title}\n{r.text[:900]}", meta={**meta_base, "section": cur_title}))
                cur_title = None
        page_text = "\n".join(r.text for r in body if r.type != "table")
        for w in _window(page_text):
            chunks.append(Chunk(level="page", document_id=doc.document_id, page_id=p.page_id, page_number=p.page_number, text=w, meta=meta_base))
    for t in session.query(Table).filter(Table.document_id == doc.document_id).all():
        hdr = " | ".join(h for h in (t.headers or []))
        rows = "\n".join(" | ".join(r) for r in (t.rows or [])[:60])
        txt = f"TABLE {t.caption or ''}\n{hdr}\n{rows}".strip()
        for w in _window(txt, 1400, 100):
            chunks.append(Chunk(level="table", document_id=doc.document_id, page_id=t.page_id, region_id=t.region_id, table_id=t.table_id,
                                page_number=session.get(Page, t.page_id).page_number if t.page_id else None, text=w,
                                meta={**meta_base, "caption": t.caption, "unit": t.unit_hint}))
    ents = {e.entity_id: e for e in session.query(Entity).all()}
    for f in session.query(Fact).filter(Fact.document_id == doc.document_id).all():
        e = ents.get(f.subject_entity_id)
        name = e.canonical_name if e else f.subject_text
        txt = f"{name} ({f.subject_text}) {f.predicate.replace('_', ' ')} {f.qualifier} {f.period}: {to_display(f.value, f.unit)} [{f.original_value}] — {doc.title}, page {f.page_number}"
        chunks.append(Chunk(level="fact", document_id=doc.document_id, page_id=f.page_id, page_number=f.page_number, region_id=f.region_id,
                            table_id=f.table_id, fact_id=f.fact_id, text=txt, meta={**meta_base, "predicate": f.predicate, "period": f.period, "subject": name}))
    return chunks


def index_document(session, doc: Document) -> int:
    delete_document(session, doc.document_id)
    session.query(Chunk).filter(Chunk.document_id == doc.document_id).delete(synchronize_session=False)
    chunks = build_chunks(session, doc)
    vecs = embed([c.text for c in chunks if c.level != "fact"]) if chunks else None
    vi = 0
    for c in chunks:
        if vecs is not None and c.level != "fact":
            c.embedding = vecs[vi].tobytes(); vi += 1
        session.add(c)
    session.flush()
    if config.DATABASE_URL.startswith("sqlite"):
        for c in chunks:
            session.execute(sql_text("INSERT INTO chunks_fts(chunk_id, text) VALUES (:id, :t)"), {"id": c.chunk_id, "t": c.text})
    session.commit()
    return len(chunks)


def delete_document(session, document_id: str) -> None:
    if config.DATABASE_URL.startswith("sqlite"):
        ids = [c.chunk_id for c in session.query(Chunk.chunk_id).filter(Chunk.document_id == document_id).all()]
        for i in range(0, len(ids), 400):
            session.execute(sql_text("DELETE FROM chunks_fts WHERE chunk_id IN (" + ",".join(f"'{x}'" for x in ids[i:i + 400]) + ")"))


def _fts_query(q: str) -> str:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]{1,}", q)
    toks = [t.replace("'", "").replace("-", " ") for t in toks if t.lower() not in STOP]
    parts = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        parts.append('"' + t + '"' if " " in t else t)
    return " OR ".join(parts) if parts else '""'


STOP = {"what", "was", "were", "is", "are", "the", "of", "in", "for", "a", "an", "to", "and", "how", "much", "many", "did",
        "does", "do", "on", "by", "with", "during", "at", "which", "who", "tell", "me", "about", "give", "show", "please", "value"}


def search(session, query: str, top_k: int = config.RETRIEVAL_TOP_K, levels: Optional[list[str]] = None,
           allowed_doc_ids: Optional[set[str]] = None, filters: Optional[dict] = None) -> list[dict]:
    """Hybrid search: BM25 (FTS5) ∪ dense cosine → reciprocal rank fusion → metadata boosts."""
    filters = filters or {}
    scores: dict[str, float] = {}
    ranks_fts: list[str] = []
    if config.DATABASE_URL.startswith("sqlite"):
        try:
            rows = session.execute(sql_text("SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts WHERE chunks_fts MATCH :q ORDER BY s LIMIT :k"),
                                   {"q": _fts_query(query), "k": top_k * 3}).fetchall()
            ranks_fts = [r[0] for r in rows]
        except Exception:
            ranks_fts = []
    qv = embed([query])
    ranks_vec: list[str] = []
    if qv is not None:
        q_ = session.query(Chunk.chunk_id, Chunk.embedding, Chunk.document_id, Chunk.level).filter(Chunk.embedding.isnot(None))
        if levels:
            q_ = q_.filter(Chunk.level.in_(levels))
        rows = q_.all()
        if rows:
            mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
            sims = mat @ qv[0]
            order = np.argsort(-sims)[: top_k * 3]
            ranks_vec = [rows[i][0] for i in order if sims[i] > 0.2]
    k = 60.0
    for rank, cid in enumerate(ranks_fts):
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
    for rank, cid in enumerate(ranks_vec):
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
    if not scores:
        return []
    ids = list(scores)
    chunks = {c.chunk_id: c for c in session.query(Chunk).filter(Chunk.chunk_id.in_(ids)).all()}
    out = []
    ql = query.lower()
    for cid, s in scores.items():
        c = chunks.get(cid)
        if c is None:
            continue
        if levels and c.level not in levels:
            continue
        if allowed_doc_ids is not None and c.document_id not in allowed_doc_ids:
            continue
        m = c.meta or {}
        if filters.get("year") and m.get("period") and str(filters["year"])[-2:] not in str(m.get("period")) and str(filters["year"]) not in str(m.get("period")):
            s *= 0.6
        if filters.get("doc_type") and m.get("doc_type") != filters["doc_type"]:
            s *= 0.5
        # lexical overlap boost (cheap reranker)
        toks = [t for t in re.findall(r"[a-z0-9]{3,}", ql) if t not in STOP]
        if toks:
            tl = c.text.lower()
            ov = sum(1 for t in toks if t in tl) / len(toks)
            s *= (1.0 + 0.8 * ov)
        if c.level == "fact":
            s *= 1.15
        out.append({"chunk_id": cid, "score": s, "level": c.level, "document_id": c.document_id, "page_id": c.page_id, "page_number": c.page_number,
                    "region_id": c.region_id, "table_id": c.table_id, "fact_id": c.fact_id, "text": c.text, "meta": m})
    out.sort(key=lambda r: -r["score"])
    return out[:top_k]
