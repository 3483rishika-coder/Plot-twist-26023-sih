"""Distributed Redis cache and job notification layer for EvidenceOS.

Falls back gracefully to a thread-safe in-memory cache when Redis is unavailable or unconfigured,
ensuring zero-setup local runs while taking full advantage of Redis in docker-compose/production.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

from . import config

logger = logging.getLogger(__name__)

_redis_client = None
_redis_checked = False
_lock = threading.Lock()

# Thread-safe in-memory fallback
_mem_cache: dict[str, tuple[float, Any]] = {}
_mem_lock = threading.Lock()


def get_redis_client():
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    with _lock:
        if _redis_checked:
            return _redis_client
        _redis_checked = True
        if not config.REDIS_URL:
            logger.info("REDIS_URL not configured; using thread-safe in-memory cache")
            return None
        try:
            import redis
            client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True, socket_connect_timeout=2)
            client.ping()
            _redis_client = client
            logger.info(f"Connected to Redis at {config.REDIS_URL}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis at {config.REDIS_URL} ({e}); falling back to in-memory cache")
            _redis_client = None
        return _redis_client


def cache_get(key: str) -> Optional[Any]:
    client = get_redis_client()
    if client:
        try:
            val = client.get(key)
            if val is not None:
                try:
                    return json.loads(val)
                except Exception:
                    return val
        except Exception as e:
            logger.warning(f"Redis get error for {key}: {e}")
    # In-memory fallback
    with _mem_lock:
        entry = _mem_cache.get(key)
        if entry:
            expiry, val = entry
            if expiry == 0 or expiry > time.time():
                return val
            _mem_cache.pop(key, None)
    return None


def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> None:
    client = get_redis_client()
    val_str = json.dumps(value) if not isinstance(value, str) else value
    if client:
        try:
            if ttl_seconds > 0:
                client.setex(key, ttl_seconds, val_str)
            else:
                client.set(key, val_str)
            return
        except Exception as e:
            logger.warning(f"Redis set error for {key}: {e}")
    # In-memory fallback
    with _mem_lock:
        expiry = time.time() + ttl_seconds if ttl_seconds > 0 else 0
        _mem_cache[key] = (expiry, value)


def cache_delete(key: str) -> None:
    client = get_redis_client()
    if client:
        try:
            client.delete(key)
        except Exception:
            pass
    with _mem_lock:
        _mem_cache.pop(key, None)


def publish_job_event(channel: str, message: dict) -> None:
    """Notify workers or UI via Redis pub/sub."""
    client = get_redis_client()
    if client:
        try:
            client.publish(channel, json.dumps(message))
        except Exception as e:
            logger.warning(f"Redis publish error: {e}")


def get_redis_status() -> dict:
    client = get_redis_client()
    configured = bool(config.REDIS_URL)
    connected = False
    details = {}
    if client:
        try:
            client.ping()
            info = client.info(section="server")
            connected = True
            details = {
                "version": info.get("redis_version"),
                "connected_clients": client.info(section="clients").get("connected_clients"),
                "used_memory_human": client.info(section="memory").get("used_memory_human"),
            }
        except Exception as e:
            details = {"error": str(e)}
    return {
        "configured": configured,
        "connected": connected,
        "url": config.REDIS_URL if configured else "in-memory (default)",
        "details": details,
    }
