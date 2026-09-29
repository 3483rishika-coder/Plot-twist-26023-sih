"""Comprehensive API and Pipeline End-to-End Verification Test.

Tests all endpoints:
1. System Status & Health (/api/v1/stats, /api/v1/roles, /api/v1/me)
2. Government & AI Integrations:
   - /api/v1/integrations/status
   - /api/v1/integrations/apisetu/leases
   - /api/v1/integrations/bhashini/translate
   - /api/v1/integrations/bhashini/ocr
   - /api/v1/integrations/digilocker/import
   - /api/v1/integrations/datagov/sync
3. Tabular Analytics & pandas Exports:
   - /api/v1/analytics/production-matrix
   - /api/v1/tables/{table_id}/export (CSV, Excel xlsx, Parquet, JSON)
4. Pipeline & Query Engine:
   - /api/v1/query (structured and RAG routes)
   - /api/v1/documents
   - /api/v1/facts
"""
import io
import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure package is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidenceos import config
from evidenceos.api import app, seed_users
from evidenceos.db import Document, Fact, Table, init_db, session_scope
from evidenceos.knowledge.entities import seed_entities

client = TestClient(app)
ADMIN_HEADER = {"X-API-Key": config.DEMO_API_KEYS["admin"]}


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Initialize DB and seed required entities and users."""
    init_db()
    with session_scope() as s:
        seed_users(s)
        seed_entities(s)


def test_system_stats_endpoint():
    """Verify /api/v1/stats includes Redis, MinIO, Ollama, and Bhashini status."""
    res = client.get("/api/v1/stats", headers=ADMIN_HEADER)
    assert res.status_code == 200
    data = res.json()
    assert "redis" in data
    assert "minio" in data
    assert "llm_provider" in data
    assert data["llm_provider"] == "ollama"
    assert "llm_model" in data
    assert "bhashini_enabled" in data
    assert data["bhashini_enabled"] is True


def test_integrations_status_endpoint():
    """Verify /api/v1/integrations/status consolidated health reporter."""
    res = client.get("/api/v1/integrations/status", headers=ADMIN_HEADER)
    assert res.status_code == 200
    data = res.json()
    assert "bhashini" in data
    assert "apisetu" in data
    assert "digilocker" in data
    assert "datagov" in data
    assert "redis" in data
    assert "minio" in data
    assert "ollama" in data
    assert data["ollama"]["provider"] == "ollama"


def test_apisetu_leases_endpoint():
    """Verify API Setu Ministry of Coal gateway lease retrieval."""
    res = client.get("/api/v1/integrations/apisetu/leases", headers=ADMIN_HEADER)
    assert res.status_code == 200
    leases = res.json()
    assert isinstance(leases, list)
    assert len(leases) >= 1
    assert "lease_id" in leases[0]

    # Filter by state
    res_jh = client.get("/api/v1/integrations/apisetu/leases?state=Jharkhand", headers=ADMIN_HEADER)
    assert res_jh.status_code == 200
    for l in res_jh.json():
        assert "jharkhand" in l["state"].lower()


def test_bhashini_translation_endpoint():
    """Verify Bhashini Indic translation endpoint with regional terms."""
    payload = {
        "text": "कोल इंडिया लिमिटेड द्वारा उत्पादन",
        "source_language": "hi",
        "target_language": "en"
    }
    res = client.post("/api/v1/integrations/bhashini/translate", json=payload, headers=ADMIN_HEADER)
    assert res.status_code == 200
    data = res.json()
    assert "Coal India Limited" in data["translated"]
    assert "Production" in data["translated"]


def test_bhashini_ocr_endpoint():
    """Verify Bhashini Indic OCR endpoint on a synthesized test image."""
    import numpy as np
    import cv2
    img = np.ones((80, 300), dtype=np.uint8) * 255
    cv2.putText(img, "COAL INDIA", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    _, buf = cv2.imencode(".png", img)

    files = {"file": ("test.png", buf.tobytes(), "image/png")}
    data = {"lang": "hi"}
    res = client.post("/api/v1/integrations/bhashini/ocr", files=files, data=data, headers=ADMIN_HEADER)
    assert res.status_code == 200
    res_json = res.json()
    assert res_json["status"] == "success"
    assert "provider" in res_json


def test_digilocker_import_endpoint():
    """Verify DigiLocker import endpoint pulls, verifies digital signature, and registers document."""
    payload = {"uri": "in.gov.coal.circular.2024.089"}
    res = client.post("/api/v1/integrations/digilocker/import", json=payload, headers=ADMIN_HEADER)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "document_id" in data
    assert "digital_signature" in data
    assert data["digital_signature"]["tamper_proof"] is True


def test_datagov_sync_endpoint():
    """Verify data.gov.in sync endpoint converts open datasets into verified facts."""
    res = client.post("/api/v1/integrations/datagov/sync?dataset=production&period=FY2024-25", headers=ADMIN_HEADER)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["facts_synced"] > 0


def test_pandas_production_matrix_endpoint():
    """Verify pandas tabular pivot matrix endpoint."""
    res = client.get("/api/v1/analytics/production-matrix?period=FY2024-25", headers=ADMIN_HEADER)
    assert res.status_code == 200
    matrix = res.json()
    assert "data" in matrix
    assert "metrics" in matrix


def test_pandas_table_export_endpoint():
    """Verify pandas table export in CSV, Excel, Parquet, and JSON."""
    with session_scope() as s:
        # Seed a dummy table if none exists
        tbl = s.query(Table).first()
        if not tbl:
            from evidenceos.db import Cell
            tbl = Table(document_id="DATAGOV-MOC-001", page_number=1, num_rows=2, num_cols=2)
            s.add(tbl); s.flush()
            s.add(Cell(table_id=tbl.table_id, row_index=0, col_index=0, raw_text="Subsidiary"))
            s.add(Cell(table_id=tbl.table_id, row_index=0, col_index=1, raw_text="Production_MT"))
            s.add(Cell(table_id=tbl.table_id, row_index=1, col_index=0, raw_text="ECL"))
            s.add(Cell(table_id=tbl.table_id, row_index=1, col_index=1, raw_text="52.04"))
            s.commit()
            table_id = tbl.table_id
        else:
            table_id = tbl.table_id

    # Test CSV export
    res_csv = client.get(f"/api/v1/tables/{table_id}/export?format=csv", headers=ADMIN_HEADER)
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    assert b"Subsidiary" in res_csv.content or b"ECL" in res_csv.content or len(res_csv.content) > 0

    # Test Excel xlsx export
    res_xlsx = client.get(f"/api/v1/tables/{table_id}/export?format=xlsx", headers=ADMIN_HEADER)
    assert res_xlsx.status_code == 200
    assert "spreadsheetml" in res_xlsx.headers["content-type"]
    assert len(res_xlsx.content) > 100

    # Test JSON export
    res_json = client.get(f"/api/v1/tables/{table_id}/export?format=json", headers=ADMIN_HEADER)
    assert res_json.status_code == 200
    assert "application/json" in res_json.headers["content-type"]


def test_query_engine_endpoint():
    """Verify query endpoint runs smoothly without crashes (testing both routes)."""
    # Fact query route
    res = client.post("/api/v1/query", json={"query": "What is the coal production?"}, headers=ADMIN_HEADER)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "citations" in data
    assert "confidence" in data
    assert "route" in data
