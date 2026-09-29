"""Unit tests for integrations (Bhashini, API Setu, DigiLocker, data.gov.in),
cache, MinIO storage, and pandas analytics.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidenceos import cache, config, store
from evidenceos.integrations import apisetu_client, bhashini_client, datagov_client, digilocker_client, get_all_integrations_status
from evidenceos.integrations.bhashini import detect_indic_script, is_indic_text, COAL_INDIAN_LANG_GLOSSARY
from evidenceos.integrations.digilocker import verify_pdf_digital_signature


def test_bhashini_indic_detection_and_translation():
    # Hindi text
    hindi_text = "कोल इंडिया लिमिटेड द्वारा कोयला उत्पादन"
    assert is_indic_text(hindi_text)
    assert detect_indic_script(hindi_text) == "devanagari"

    # English text
    eng_text = "Coal India Limited annual production"
    assert not is_indic_text(eng_text)
    assert detect_indic_script(eng_text) is None

    # Offline translation fallback via glossary
    translated = bhashini_client.translate("कोल इंडिया लिमिटेड", source_language="hi", target_language="en")
    assert "Coal India Limited" in translated


def test_apisetu_client():
    leases = apisetu_client.fetch_coal_leases()
    assert len(leases) >= 3
    # Check Jharkhand lease
    jh_leases = apisetu_client.fetch_coal_leases(state="Jharkhand")
    assert len(jh_leases) >= 1
    assert any("BCCL" in l["company"] for l in jh_leases)

    status = apisetu_client.get_status()
    assert status["name"].startswith("API Setu")


def test_digilocker_client():
    docs = digilocker_client.list_issued_documents()
    assert len(docs) >= 2
    assert any("Ministry of Coal" in d["issuer"] for d in docs)

    status = digilocker_client.get_status()
    assert "DigiLocker" in status["name"]


def test_datagov_client():
    records = datagov_client.fetch_resource("coal-production")
    assert len(records) >= 5
    assert any(r["subsidiary"] == "ECL" for r in records)

    reserves = datagov_client.fetch_resource("coal-reserves")
    assert len(reserves) >= 3
    assert any(r["state"] == "Jharkhand" for r in reserves)


def test_cache_memory_fallback():
    # In absence of running Redis, cache works with in-memory fallback
    key = "test_test_key_999"
    cache.cache_set(key, {"test": 123}, ttl_seconds=60)
    val = cache.cache_get(key)
    assert val == {"test": 123}
    cache.cache_delete(key)
    assert cache.cache_get(key) is None


def test_minio_status():
    status = store.get_minio_status()
    assert "configured" in status
    assert "bucket" in status
    assert status["bucket"] == "evidenceos"


def test_all_integrations_consolidated_status():
    status = get_all_integrations_status()
    assert "bhashini" in status
    assert "apisetu" in status
    assert "digilocker" in status
    assert "datagov" in status
    assert "redis" in status
    assert "minio" in status
    assert "ollama" in status
