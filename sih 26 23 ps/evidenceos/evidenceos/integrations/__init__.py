"""EvidenceOS Government & AI Integrations Suite (PRD §8, §13).

Provides connectors for:
- Bhashini Indic AI (MeitY) for Indic OCR & Translation
- API Setu (National API Gateway) for Ministry of Coal data exchange
- DigiLocker for tamper-proof digital signature verification & document pull
- data.gov.in (OGD Platform India) for open government mining datasets
"""
from __future__ import annotations

from .apisetu import apisetu_client
from .bhashini import bhashini_client
from .datagov import datagov_client
from .digilocker import digilocker_client


def get_all_integrations_status() -> dict:
    """Return consolidated status of all government and national platform integrations."""
    from .. import cache, config, store

    return {
        "bhashini": bhashini_client.get_status(),
        "apisetu": apisetu_client.get_status(),
        "digilocker": digilocker_client.get_status(),
        "datagov": datagov_client.get_status(),
        "redis": cache.get_redis_status(),
        "minio": store.get_minio_status(),
        "ollama": {
            "name": "Ollama Local AI (Air-Gapped / Offline)",
            "provider": config.LLM_PROVIDER,
            "model": config.LLM_MODEL,
            "base_url": config.LLM_BASE_URL,
            "mode": "air_gapped_offline",
        }
    }
