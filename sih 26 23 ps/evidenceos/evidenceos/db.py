"""Database models — the *system of record* (PRD §9).

Original document → page → region → table/cell → fact → validation/conflict,
with users, reports and an append-only audit log.  Works on SQLite (SIH) and
PostgreSQL (production) through SQLAlchemy; JSON columns map to JSONB on PG.
"""
from __future__ import annotations

import datetime as dt
import uuid
from contextlib import contextmanager

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        LargeBinary, String, Text, create_engine, event, text)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from . import config

Base = declarative_base()


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Document(Base):
    __tablename__ = "documents"
    document_id = Column(String(36), primary_key=True, default=new_id)
    code = Column(String(24), unique=True)                 # human readable e.g. DOC-0004
    original_filename = Column(Text)
    title = Column(Text)
    sha256 = Column(String(64), index=True)
    source = Column(Text)                                  # publisher / subsidiary
    source_url = Column(Text)
    document_type = Column(String(64))                     # monthly_statistics, annual_report, coal_directory, inventory, administrative
    document_date = Column(String(10))                     # ISO date if known
    period = Column(String(24))                            # e.g. FY2024-25 or 2025-03
    department = Column(Text)
    version = Column(Integer, default=1)
    family_key = Column(Text)                              # documents of the same report family
    storage_location = Column(Text)
    file_size = Column(Integer)
    page_count = Column(Integer)
    is_scanned = Column(Boolean, default=False)
    scan_ratio = Column(Float)
    acl_level = Column(String(16), default="public")
    status = Column(String(24), default="queued")          # queued/processing/extracted/validated/reviewed/published/failed
    error = Column(Text)
    meta = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    version_id = Column(String(36), primary_key=True, default=new_id)
    document_id = Column(String(36), ForeignKey("documents.document_id"))
    parent_document_id = Column(String(36), nullable=True)
    sha256 = Column(String(64))
    version_number = Column(Integer)
    similarity_score = Column(Float)
    created_at = Column(DateTime, default=utcnow)


class Page(Base):
    __tablename__ = "pages"
    page_id = Column(String(36), primary_key=True, default=new_id)
    document_id = Column(String(36), ForeignKey("documents.document_id"), index=True)
    page_number = Column(Integer)                          # 1-based
    width = Column(Float)                                  # PDF points
    height = Column(Float)
    image_path = Column(Text)
    is_scanned = Column(Boolean, default=False)
    text = Column(Text)                                    # full page text (text layer or OCR)
    quality = Column(JSON, default=dict)
    ocr_confidence = Column(Float)
    status = Column(String(24), default="pending")
    section = Column(Text)                                 # detected section / table caption context

    document = relationship("Document", back_populates="pages")
    regions = relationship("Region", back_populates="page", cascade="all, delete-orphan")


class Region(Base):
    __tablename__ = "regions"
    region_id = Column(String(36), primary_key=True, default=new_id)
    page_id = Column(String(36), ForeignKey("pages.page_id"), index=True)
    document_id = Column(String(36), index=True)
    type = Column(String(24))                              # text, title, table, figure, chart, handwriting, header, footer, page_number, caption
    bbox = Column(JSON)                                    # [x0,y0,x1,y1] in PDF points
    reading_order = Column(Integer)
    confidence = Column(Float)
    text = Column(Text)
    image_path = Column(Text)
    meta = Column(JSON, default=dict)

    page = relationship("Page", back_populates="regions")


class Table(Base):
    __tablename__ = "tables"
    table_id = Column(String(36), primary_key=True, default=new_id)
    region_id = Column(String(36), ForeignKey("regions.region_id"), index=True)
    page_id = Column(String(36), index=True)
    document_id = Column(String(36), index=True)
    table_group_id = Column(String(36), index=True)
    page_sequence = Column(JSON)
    caption = Column(Text)
    headers = Column(JSON)                                 # list of column labels (merged header rows)
    rows = Column(JSON)                                    # list of list of cell text
    n_rows = Column(Integer)
    n_cols = Column(Integer)
    header_rows = Column(Integer)
    unit_hint = Column(Text)
    method = Column(String(24))                            # pymupdf, wordgrid, ocrgrid
    confidence = Column(Float)
    meta = Column(JSON, default=dict)


class Cell(Base):
    __tablename__ = "cells"
    cell_id = Column(String(36), primary_key=True, default=new_id)
    table_id = Column(String(36), ForeignKey("tables.table_id"), index=True)
    row_index = Column(Integer)
    col_index = Column(Integer)
    bbox = Column(JSON)
    text = Column(Text)
    value = Column(Float)
    unit = Column(Text)
    confidence = Column(Float)


class Entity(Base):
    __tablename__ = "entities"
    entity_id = Column(String(36), primary_key=True, default=new_id)
    canonical_name = Column(Text, index=True)
    entity_type = Column(String(24))                       # company, subsidiary, mine, state, coalfield, sector, country, organization
    parent_entity_id = Column(String(36), nullable=True)
    meta = Column(JSON, default=dict)
    is_aggregate = Column(Boolean, default=False)


class EntityAlias(Base):
    __tablename__ = "entity_aliases"
    alias_id = Column(String(36), primary_key=True, default=new_id)
    entity_id = Column(String(36), ForeignKey("entities.entity_id"), index=True)
    alias = Column(Text, index=True)
    alias_norm = Column(Text, index=True)
    source = Column(Text)
    confidence = Column(Float, default=1.0)


class Fact(Base):
    __tablename__ = "facts"
    fact_id = Column(String(36), primary_key=True, default=new_id)
    code = Column(String(24), index=True)
    subject_entity_id = Column(String(36), ForeignKey("entities.entity_id"), index=True)
    subject_text = Column(Text)
    predicate = Column(String(64), index=True)
    dimension_entity_id = Column(String(36), nullable=True, index=True)   # e.g. consuming sector for dispatch facts
    dimension_text = Column(Text)
    qualifier = Column(String(24))                         # actual, provisional, target, projected
    period = Column(String(24), index=True)                # FY2024-25 | 2025-03 | asof:2024-04-01
    period_scope = Column(String(16))                      # fy, month, ytd, asof, year
    value = Column(Float)                                  # canonical value (tonnes, cubic metres, percent, count)
    unit = Column(String(24))                              # canonical unit
    display_value = Column(Text)                           # value in original display unit e.g. 52.04
    original_value = Column(Text)
    original_unit = Column(Text)
    document_id = Column(String(36), ForeignKey("documents.document_id"), index=True)
    page_id = Column(String(36), index=True)
    page_number = Column(Integer)
    region_id = Column(String(36))
    table_id = Column(String(36))
    cell_id = Column(String(36))
    bbox = Column(JSON)
    context = Column(Text)                                 # header path / sentence
    extraction_method = Column(String(24))                 # table, narrative
    confidence = Column(JSON, default=dict)
    overall_confidence = Column(Float)
    validation_status = Column(String(24), default="pending")  # valid, conflict, low_confidence, rejected, verified
    review_status = Column(String(16), default="none")     # none, recommended, mandatory, done
    created_at = Column(DateTime, default=utcnow)


class Validation(Base):
    __tablename__ = "validations"
    validation_id = Column(String(36), primary_key=True, default=new_id)
    fact_id = Column(String(36), ForeignKey("facts.fact_id"), index=True)
    rule_id = Column(String(16))
    rule_type = Column(String(24))                         # numeric, structural, temporal, cross_document, arithmetic, entity, ocr
    status = Column(String(12))                            # pass, fail, warning
    message = Column(Text)
    evidence = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)


class Conflict(Base):
    __tablename__ = "conflicts"
    conflict_id = Column(String(36), primary_key=True, default=new_id)
    code = Column(String(24))
    subject_entity_id = Column(String(36), index=True)
    subject_text = Column(Text)
    predicate = Column(String(64))
    period = Column(String(24))
    unit = Column(String(24))
    fact_ids = Column(JSON)
    values = Column(JSON)
    severity = Column(String(12))                          # minor, major
    explanation = Column(Text)
    status = Column(String(12), default="open")            # open, resolved, ignored
    resolution = Column(Text)
    resolved_fact_id = Column(String(36))
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Correction(Base):
    """Human-in-the-loop ground truth (PRD §11.6.4 feedback loop)."""
    __tablename__ = "corrections"
    correction_id = Column(String(36), primary_key=True, default=new_id)
    fact_id = Column(String(36), index=True)
    user_id = Column(String(36))
    action = Column(String(12))                            # accept, edit, reject
    old_value = Column(Text)
    new_value = Column(Text)
    note = Column(Text)
    created_at = Column(DateTime, default=utcnow)


class Chunk(Base):
    """Multi-level retrieval units (document/section/page/table/fact) — PRD §12.1."""
    __tablename__ = "chunks"
    chunk_id = Column(String(36), primary_key=True, default=new_id)
    level = Column(String(12), index=True)
    document_id = Column(String(36), index=True)
    page_id = Column(String(36))
    page_number = Column(Integer)
    region_id = Column(String(36))
    table_id = Column(String(36))
    fact_id = Column(String(36))
    text = Column(Text)
    meta = Column(JSON, default=dict)
    embedding = Column(LargeBinary)


class Report(Base):
    __tablename__ = "reports"
    report_id = Column(String(36), primary_key=True, default=new_id)
    template_id = Column(String(48))
    title = Column(Text)
    generated_by = Column(String(36))
    params = Column(JSON, default=dict)
    document_path = Column(Text)
    facts_used = Column(JSON, default=list)
    missing = Column(JSON, default=list)
    status = Column(String(16), default="generated")
    created_at = Column(DateTime, default=utcnow)


class User(Base):
    __tablename__ = "users"
    user_id = Column(String(36), primary_key=True, default=new_id)
    username = Column(String(64), unique=True)
    role = Column(String(24))
    api_key = Column(String(128), unique=True)
    department = Column(Text)
    created_at = Column(DateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    audit_id = Column(String(36), primary_key=True, default=new_id)
    user_id = Column(String(36))
    username = Column(String(64))
    action = Column(String(48), index=True)
    entity_type = Column(String(32))
    entity_id = Column(String(36))
    query = Column(Text)
    document_accessed = Column(String(36))
    detail = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=utcnow, index=True)


class Job(Base):
    __tablename__ = "jobs"
    job_id = Column(String(36), primary_key=True, default=new_id)
    document_id = Column(String(36), index=True)
    stage = Column(String(24), default="process")
    status = Column(String(16), default="queued")          # queued, running, done, failed, dead
    attempts = Column(Integer, default=0)
    params = Column(JSON, default=dict)
    error = Column(JSON)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Analytics(Base):
    __tablename__ = "analytics"
    key = Column(String(128), primary_key=True)
    payload = Column(JSON)
    created_at = Column(DateTime, default=utcnow)


class Counter(Base):
    __tablename__ = "counters"
    name = Column(String(32), primary_key=True)
    value = Column(Integer, default=0)


# --------------------------------------------------------------------------
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 60} if config.DATABASE_URL.startswith("sqlite") else {},
    future=True,
)

if config.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=60000")
        cur.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        if config.DATABASE_URL.startswith("sqlite"):
            conn.execute(text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='porter unicode61')"
            ))


@contextmanager
def session_scope():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def next_code(session, name: str, prefix: str, width: int = 4) -> str:
    c = session.get(Counter, name)
    if c is None:
        c = Counter(name=name, value=0)
        session.add(c)
    c.value = (c.value or 0) + 1
    session.flush()
    return f"{prefix}-{c.value:0{width}d}"
