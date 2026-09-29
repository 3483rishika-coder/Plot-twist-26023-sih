"""Central configuration for EvidenceOS.

Everything is environment-overridable so the same code runs in the SIH sandbox
(SQLite + local filesystem) and in production (PostgreSQL/pgvector + MinIO).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EVIDENCEOS_DATA_DIR", ROOT_DIR / "data")).resolve()
STORE_DIR = Path(os.environ.get("EVIDENCEOS_STORE_DIR", DATA_DIR / "store")).resolve()
CACHE_DIR = Path(os.environ.get("EVIDENCEOS_CACHE_DIR", Path.home() / ".cache" / "evidenceos")).resolve()
TEMPLATE_DIR = ROOT_DIR / "templates"
WEB_DIR = ROOT_DIR / "web"

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR / 'evidenceos.db'}")

# --- Document AI -----------------------------------------------------------
OCR_DPI = int(os.environ.get("EVIDENCEOS_OCR_DPI", "170"))          # render DPI for OCR of scanned pages
VIEW_DPI = int(os.environ.get("EVIDENCEOS_VIEW_DPI", "110"))        # render DPI for page images in the viewer
SCANNED_TEXT_THRESHOLD = 40                                          # chars of text layer below which a page is "scanned"
MAX_PAGES_DEFAULT = int(os.environ.get("EVIDENCEOS_MAX_PAGES", "0"))  # 0 = no limit

# --- Confidence policy (PRD §11.4.5) ----------------------------------------
CONFIDENCE_WEIGHTS = {
    "ocr": 0.25,
    "layout": 0.10,
    "structure": 0.20,
    "entity_matching": 0.15,
    "validation": 0.15,
    "source_agreement": 0.15,
}
AUTO_ACCEPT_THRESHOLD = float(os.environ.get("EVIDENCEOS_AUTO_ACCEPT", "0.90"))
MANDATORY_REVIEW_THRESHOLD = float(os.environ.get("EVIDENCEOS_MANDATORY_REVIEW", "0.70"))

# --- Validation -------------------------------------------------------------
CONFLICT_REL_TOLERANCE = float(os.environ.get("EVIDENCEOS_CONFLICT_TOL", "0.00002"))  # floating-point noise only; rounding handled by decimals
ARITHMETIC_REL_TOLERANCE = 0.01   # 1 % for total = sum(rows) (rounded published figures)

# --- Distributed Storage & Cache (Redis & MinIO) -----------------------------
REDIS_URL = os.environ.get("REDIS_URL", "")                  # e.g. redis://redis:6379/0 (falls back to in-memory)
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "")        # e.g. minio:9000 or localhost:9000 (falls back to local filesystem)
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", os.environ.get("MINIO_ROOT_USER", "evidenceos"))
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", os.environ.get("MINIO_ROOT_PASSWORD", "evidenceos-secret"))
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "evidenceos")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "0") in ("1", "true", "True")

# --- Retrieval / LLM (Ollama offline default + OpenAI / vLLM fallback) -------
EMBEDDING_MODEL = os.environ.get("EVIDENCEOS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDINGS_ENABLED = os.environ.get("EVIDENCEOS_EMBEDDINGS", "1") not in ("0", "false", "False")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")      # "ollama" (local offline/air-gapped default), "openai", "vllm", "groq"
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434")  # Ollama default; overridable for Docker/cloud
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")             # optional for Ollama, required for OpenAI
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3:8b")         # default local model (e.g. llama3:8b, qwen2.5:7b)
RETRIEVAL_TOP_K = 20
EVIDENCE_TOP_K = 6

# --- Government Integrations & Bhashini Indic AI ------------------------------
BHASHINI_ENABLED = os.environ.get("BHASHINI_ENABLED", "1") not in ("0", "false", "False")
BHASHINI_API_KEY = os.environ.get("BHASHINI_API_KEY", "")
BHASHINI_USER_ID = os.environ.get("BHASHINI_USER_ID", "")
BHASHINI_PIPELINE_ID = os.environ.get("BHASHINI_PIPELINE_ID", "")
BHASHINI_INFERENCE_URL = os.environ.get("BHASHINI_INFERENCE_URL", "https://dhruva-api.bhashini.gov.in/services/inference/pipeline")

APISETU_CLIENT_ID = os.environ.get("APISETU_CLIENT_ID", "")
APISETU_API_KEY = os.environ.get("APISETU_API_KEY", "")
APISETU_BASE_URL = os.environ.get("APISETU_BASE_URL", "https://apisetu.gov.in/api/v1")

DIGILOCKER_CLIENT_ID = os.environ.get("DIGILOCKER_CLIENT_ID", "")
DIGILOCKER_CLIENT_SECRET = os.environ.get("DIGILOCKER_CLIENT_SECRET", "")
DIGILOCKER_BASE_URL = os.environ.get("DIGILOCKER_BASE_URL", "https://api.digitallocker.gov.in/public/oauth2/1")

DATAGOV_API_KEY = os.environ.get("DATAGOV_API_KEY", "")
DATAGOV_BASE_URL = os.environ.get("DATAGOV_BASE_URL", "https://api.data.gov.in/resource")

# --- Security ----------------------------------------------------------------
DEMO_API_KEYS = {
    # role -> api key (demo seeds; production uses SSO + hashed keys)
    "admin": "eos-admin-key",
    "officer": "eos-officer-key",
    "geologist": "eos-geologist-key",
    "validator": "eos-validator-key",
    "auditor": "eos-auditor-key",
    "viewer": "eos-viewer-key",
}
ROLE_PERMISSIONS = {
    "admin": {"*"},
    "officer": {"query", "report", "read"},
    "geologist": {"upload", "review", "query", "read", "report"},
    "report_writer": {"report", "read", "query"},
    "validator": {"review", "read", "query"},
    "auditor": {"audit", "read"},
    "viewer": {"read"},
}
ACL_ORDER = {"public": 0, "private": 1, "sensitive": 2}
ROLE_MAX_ACL = {"admin": 2, "officer": 1, "geologist": 2, "validator": 2, "auditor": 2, "viewer": 0, "report_writer": 1}

for _d in (DATA_DIR, STORE_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
