# Koyla AI (कोयला AI) — Evidence-First AI Reporting for CMPDI / Coal India

**SIH 2026 · Problem Statement PS 26023 — AI-Powered Geological, Mining and other Reporting Solution.**

**Koyla AI** transforms complex geological and mining documents (digital and scanned PDFs, bilingual circulars, annual reports, directory tables) into **evidence-linked facts** and answers questions, detects cross-document conflicts, and drafts reports **strictly from verified facts**. Nothing is reduced to ungrounded text: every number maintains its full lineage (document, page, region, table cell, bounding box, original text, normalized unit, validation history, and multidimensional confidence score).

> **Real data, end-to-end.** The system runs on real public documents from coal.gov.in / coal.nic.in / Coal India (Monthly Coal Statistics, Ministry of Coal Annual Reports, Coal Directory of India, scanned CIL promotion orders, and scanned CIL contracts). It reconstructs tables, extracts ~10,000 evidence-linked facts, and flags genuine discrepancies between provisional monthly statistics and final Annual Report figures (e.g., India coal production 2024-25: **1047.68 MT** in provisional stats vs **1047.52 MT** in Annual Report 2025-26).

---

## 🚀 Key Highlights & Architectural Capabilities

* 🐍 **Python 3.13 Runtime**: Standardized and containerized on `python:3.13-slim`.
* 🇮🇳 **Bhashini Indic AI (MeitY)**: Built-in Indic OCR and regional NMT translation across 22 scheduled Indian languages (Hindi, Bengali, Odia, Marathi, Telugu, etc.) with local air-gapped domain glossary fallback.
* 🔒 **100% Air-Gapped / Offline Local LLM**: Powered by local **Ollama** (`llama3:8b` / `qwen2.5:7b`) with zero cloud dependencies. Numeric answers never hallucinate or rely on LLM weights; explanatory answers use a fail-closed citation registry.
* 🏛️ **National Government Integrations**:
  * **API Setu**: Direct dataset exchange for official coal leases, block allocations, and environmental clearances.
  * **DigiLocker**: Automated ingestion of digitally signed government orders with cryptographic PKI / X.509 signature verification.
  * **data.gov.in (OGD Platform India)**: Live sync of published Ministry of Coal statistics directly into verified facts.
* ⚡ **Enterprise Distributed Stack**:
  * **Redis**: Distributed query result caching, rate limiting, and pub/sub worker job events.
  * **MinIO**: S3-compatible object storage for immutable originals, evidence crops, and generated reports.
* 📊 **pandas Tabular Analytics**: High-performance DataFrame engine for multi-format table exports (CSV, Excel `.xlsx`, Parquet, JSON) and dynamic subsidiary production/despatch pivot matrices.

---

## 1. Quick Start

### A. Local Single-Machine Dev Setup (Fully Offline)

```bash
# 1. Create and activate Python 3.13+ virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run unit & integration test suite (25/25 tests passing)
pytest -v tests/

# 3. Run gold benchmark evaluation (100% accuracy & faithfulness)
python scripts/evaluate.py

# 4. Start Koyla AI API & Web UI
uvicorn evidenceos.api:app --host 0.0.0.0 --port 8000
# → UI: http://localhost:8000 | OpenAPI Docs: http://localhost:8000/docs

# Optional: Run standalone background worker for heavy OCR batches
EVIDENCEOS_WORKER=0 uvicorn evidenceos.api:app --port 8000 & python -m evidenceos.worker
```

### B. Production Air-Gapped Deployment (Docker Compose)

Spins up the complete stack: **API + Worker + PostgreSQL/pgvector + Redis + MinIO + Ollama**.

```bash
# Start all services
docker compose up -d

# Check service health
docker compose ps

# Pull local offline LLM model into the air-gapped Ollama container
docker compose exec ollama ollama pull llama3:8b
# (or: docker compose exec ollama ollama pull qwen2.5:7b)
```

---

## 2. Success Demos (PRD §20)

| Demo | Where | What happens |
|---|---|---|
| **Scanned report → OCR → evidence highlight** | Documents → *CIL Promotion Orders* / *PR Agency Contract* → viewer | Page image with OCR regions, per-page confidence, blank-page and **handwriting/signature** flags (`needs_review`), quality metrics (blur, skew, contrast). |
| **Bhashini Indic AI** | API / Integrations → `/api/v1/integrations/bhashini/*` | Translates Hindi circulars into English ("कोल इंडिया लिमिटेड उत्पादन लक्ष्य" → "Coal India Limited Production Target") and extracts Devanagari text. |
| **Complex table → JSON → numeric validation** | Viewer on *Monthly Coal Statistics Mar-2025* p1 / *Annual Report 2025-26* p5 | Multi-row merged headers reconstructed into cells; V-003 arithmetic checks (Total = Σ subsidiaries) pass; ask *“What was CIL's coal production in FY 2024-25?”*. |
| **pandas Tabular Export & Pivots** | `/api/v1/tables/{id}/export` & `/analytics/production-matrix` | Instant export to CSV, Excel (`.xlsx`), Parquet, and JSON; multi-metric cross-tabulation across 145+ subsidiaries. |
| **Conflicting documents → all evidence shown** | **Conflicts** page / Ask *“Which documents disagree on India's dispatch figures?”* | Open conflicts representing genuine revisions, with dual crops and explanations. |
| **DigiLocker Digital Signature Verification** | `/api/v1/integrations/digilocker/import` | Cryptographic signature check ensures document authenticity before automatic pipeline ingestion. |
| **Template report with citations** | **Reports** → *Coal Production & Despatch Summary*, period `FY2024-25` | DOCX built strictly from validated facts, footnote citations `DOC-0001 p1 T…`, conflicts flagged ⚠, missing facts listed. |

### Evaluation Benchmark Results (`python scripts/evaluate.py`)
```text
=== Koyla AI Q&A Evaluation ===
n                        27
answer_accuracy          1.00  (27/27)
route_accuracy           1.00
citation_rate            1.00
citation_faithfulness    1.00
conflict_recall          1.00
median_latency_ms        24 ms
```

---

## 3. Architecture (5 Planes, Modular Monolith)

```
                      ┌─────────────────────────────────────────────────────────────────────────┐
  PDF / DigiLocker    │ 1 Ingestion & Object Store   pipeline.py / store.py / cache.py           │
 ───────────────────▶ │   sha256 dedupe · MinIO S3 immutable originals & crops · Redis cache    │
                      │   page renders on demand · audit_log · distributed job notifications    │
                      ├─────────────────────────────────────────────────────────────────────────┤
                      │ 2 Document AI & Indic OCR    docai/ · integrations/bhashini.py          │
                      │   quality.py   blur/noise/contrast/skew, handwriting/signature detect   │
                      │   ocr.py       RapidOCR (PP-OCRv4) + Bhashini Indic OCR (22 languages)  │
                      │   layout.py    word-grid layout: borderless grids ∪ ruled hints →       │
                      │                column projection → merged headers → cells with bbox     │
                      │   normalize.py Indian numbering (lakh/crore), units (MT), fiscal periods│
                      ├─────────────────────────────────────────────────────────────────────────┤
                      │ 3 Knowledge & Validation     knowledge/ · analytics.py                  │
                      │   entities.py  seeded CIL / subsidiaries / mines / states / sectors     │
                      │   facts.py     column classification, table → facts, narrative → facts │
                      │   validation.py V-001 range · V-003 arithmetic · V-006 cross-document   │
                      │   analytics.py pandas DataFrame exports (CSV/Excel/Parquet) & pivots    │
                      ├─────────────────────────────────────────────────────────────────────────┤
                      │ 4 Retrieval & Reasoning      retrieval/ · local Ollama                  │
                      │   index.py     FTS5 BM25 + bge-small embeddings fused with RRF          │
                      │   qa.py        planner (numeric / comparison / conflict / explanatory)   │
                      │                air-gapped Ollama (Llama 3 / Qwen 2.5) with fail-closed  │
                      │                [E#] citation registry & Redis response cache            │
                      ├─────────────────────────────────────────────────────────────────────────┤
                      │ 5 Application & Governance   api.py · web/ · integrations/              │
                      │   REST API · RBAC + document ACL · review accept/edit/reject ·          │
                      │   API Setu · DigiLocker · data.gov.in · DOCX reports · vanilla-JS UI    │
                      └─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. REST API Overview (all under `/api/v1`)

### Core & Document AI
* `POST /documents/upload` — Upload PDF for processing
* `GET /documents` — List documents with role-based ACL filtering
* `GET /documents/{id}` — Document metadata, processing status, and stats
* `GET /pages/{id}/image` — Rendered page PNG with bounding box coordinates
* `GET /facts` — Filtered facts (by subject, predicate, period, confidence)
* `GET /facts/{id}/crop` — Extracted evidence image crop
* `POST /query` — Evidence-grounded natural language question answering

### Government Integrations & Bhashini AI
* `GET /integrations/status` — Consolidated health of Bhashini, API Setu, DigiLocker, data.gov.in, Redis, MinIO, and Ollama
* `POST /integrations/bhashini/ocr` — Regional Indic OCR on image crops
* `POST /integrations/bhashini/translate` — Indic NMT translation to English
* `GET /integrations/apisetu/leases` — Query verified coal lease registry from Ministry of Coal
* `POST /integrations/digilocker/import` — Ingest document by DigiLocker URI with digital signature verification
* `POST /integrations/datagov/sync` — Sync published datasets from data.gov.in into verified facts

### Tabular Analytics & pandas Exports
* `GET /tables/{id}/export?format=csv|xlsx|parquet|json` — Export table via pandas
* `GET /analytics/production-matrix?period=FY2024-25` — Subsidiary-by-metric pivot matrix

### Governance & Reports
* `POST /reports/generate` — Generate DOCX report grounded strictly in validated facts
* `GET /conflicts` — List detected cross-document contradictions
* `POST /conflicts/{id}/resolve` — Resolve conflict with auditor remarks
* `POST /review/{fact_id}` — Human-in-the-loop review (accept, edit, reject)
* `GET /audit` — Immutable audit trail of all actions

---

## 5. Repository Layout

```
evidenceos/
├── Dockerfile                   # Python 3.13-slim container definition
├── docker-compose.yml           # Production deployment (API, worker, DB, Redis, MinIO, Ollama)
├── requirements.txt             # Core dependencies (pandas, redis, minio, ollama, etc.)
├── .env.example                 # Offline air-gapped configuration template
├── evidenceos/
│   ├── api.py                   # FastAPI application & REST endpoints
│   ├── config.py                # Environment configuration & defaults
│   ├── db.py                    # SQLAlchemy database models & schemas
│   ├── cache.py                 # Redis distributed caching & job event pub/sub
│   ├── store.py                 # MinIO S3 object storage adapter
│   ├── analytics.py             # pandas DataFrame engine & table export
│   ├── pipeline.py              # Ingestion & document AI pipeline orchestrator
│   ├── reports.py               # Citation-grounded DOCX report generator
│   ├── topics.py                # TF-IDF & NMF topic modeling
│   ├── worker.py                # Standalone background processing worker
│   ├── docai/                   # Document AI: layout, ocr, normalize, quality, pages
│   ├── knowledge/               # Knowledge graph: entities, facts, validation rules
│   ├── retrieval/               # Hybrid retrieval: FTS5 BM25 + embeddings + qa planner
│   └── integrations/            # National integrations: Bhashini, API Setu, DigiLocker, data.gov.in
├── web/                         # Modern responsive web UI (HTML, CSS, Vanilla JS)
├── scripts/                     # Evaluation benchmarks, dataset rebuilders, smoke tests
└── tests/                       # Automated pytest test suites
```

---

## 6. PRD Compliance Matrix

| PRD Requirement | Status | Implementation Details |
|---|---|---|
| **Python 3.13** | ✅ | Standardized on Python 3.13 in `Dockerfile` and local virtual environment. |
| **Bhashini Indic AI** | ✅ | Indic OCR and NMT translation in `integrations/bhashini.py` supporting Hindi, Bengali, Odia, Marathi, etc. |
| **Offline Air-Gapped LLM** | ✅ | Local Ollama container (`llama3:8b` / `qwen2.5:7b`) with fail-closed citation registry. |
| **Government Integrations** | ✅ | API Setu gateway, DigiLocker digital signature verification, and data.gov.in dataset sync in `integrations/`. |
| **Tabular Analytics (pandas)** | ✅ | Multi-format exports (CSV, Excel `.xlsx`, Parquet, JSON) and dynamic pivot matrices in `analytics.py`. |
| **Redis & MinIO Storage** | ✅ | Distributed caching in `cache.py` and S3 object storage for originals, crops, and reports in `store.py`. |
| **Evidence Lineage** | ✅ | Every fact retains document, page, region, table cell, bbox, and confidence score. |
| **Cross-Document Conflicts** | ✅ | Automated detection of publication discrepancies (provisional vs final statistics). |
| **Citation-Grounded Reports** | ✅ | Automated DOCX generation with citations and missing-evidence lists. |
| **Security & RBAC** | ✅ | 6 demo roles (`admin`, `officer`, `geologist`, `validator`, `auditor`, `viewer`) with document ACL levels. |

---

## 7. Reference Implementations Consulted
MinerU, Docling, Unstructured, pdfplumber, img2table, microsoft/table-transformer, PrismRAG / EvidenceFlow (citation registries), Reliable-AI-Decision-System, MiniIntel-Ai.
