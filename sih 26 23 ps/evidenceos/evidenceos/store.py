"""Evidence store (PRD §11.2): immutable originals + derived artifacts.

Local filesystem layout (S3/MinIO adapter keeps the same keys):
  originals/{document_id}/{filename}      immutable original
  crops/{document_id}/{region_or_cell}.png evidence crops (generated once, cached)
  reports/{report_id}.docx                 generated reports
Page images are rendered on demand from the original and cached under CACHE_DIR
(they are fully derivable, so they are not part of the persistent snapshot).
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import threading
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

_minio_client = None
_minio_checked = False
_lock = threading.Lock()


def get_minio_client():
    global _minio_client, _minio_checked
    if _minio_checked:
        return _minio_client
    with _lock:
        if _minio_checked:
            return _minio_client
        _minio_checked = True
        if not config.MINIO_ENDPOINT:
            return None
        try:
            from minio import Minio
            client = Minio(
                config.MINIO_ENDPOINT,
                access_key=config.MINIO_ACCESS_KEY,
                secret_key=config.MINIO_SECRET_KEY,
                secure=config.MINIO_SECURE,
            )
            # Ensure bucket exists
            if not client.bucket_exists(config.MINIO_BUCKET):
                client.make_bucket(config.MINIO_BUCKET)
                logger.info(f"Created MinIO bucket '{config.MINIO_BUCKET}'")
            _minio_client = client
            logger.info(f"Connected to MinIO at {config.MINIO_ENDPOINT}, bucket='{config.MINIO_BUCKET}'")
        except Exception as e:
            logger.warning(f"Could not connect to MinIO at {config.MINIO_ENDPOINT}: {e}; continuing with local filesystem")
            _minio_client = None
        return _minio_client


def upload_to_minio(local_path: str | Path, object_name: str, content_type: Optional[str] = None) -> bool:
    client = get_minio_client()
    if not client:
        return False
    local_path = Path(local_path)
    if not local_path.exists():
        return False
    try:
        client.fput_object(
            config.MINIO_BUCKET,
            object_name,
            str(local_path),
            content_type=content_type or "application/octet-stream",
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to upload {object_name} to MinIO: {e}")
        return False


def download_from_minio_if_needed(local_path: str | Path, object_name: str) -> bool:
    local_path = Path(local_path)
    if local_path.exists():
        return True
    client = get_minio_client()
    if not client:
        return False
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(config.MINIO_BUCKET, object_name, str(local_path))
        return True
    except Exception as e:
        logger.warning(f"Failed to download {object_name} from MinIO: {e}")
        return False


def get_minio_status() -> dict:
    client = get_minio_client()
    configured = bool(config.MINIO_ENDPOINT)
    connected = False
    buckets = []
    if client:
        try:
            connected = client.bucket_exists(config.MINIO_BUCKET)
            buckets = [b.name for b in client.list_buckets()]
        except Exception as e:
            return {"configured": configured, "connected": False, "endpoint": config.MINIO_ENDPOINT, "error": str(e)}
    return {
        "configured": configured,
        "connected": connected,
        "endpoint": config.MINIO_ENDPOINT if configured else "local-filesystem (default)",
        "bucket": config.MINIO_BUCKET,
        "all_buckets": buckets,
    }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_original(src: str | Path, document_id: str, filename: str, link: bool = False) -> str:
    src = Path(src)
    if link:
        return str(src.resolve())               # register in place (external / object-store URI analogue)
    dst_dir = config.STORE_DIR / "originals" / document_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / filename
    if not dst.exists():
        shutil.copy2(src, dst)
    try:
        dst.chmod(0o444)                        # immutable-ish: read only
    except Exception:
        pass
    # Persist immutable copy in MinIO S3 object store
    upload_to_minio(dst, f"originals/{document_id}/{filename}", content_type="application/pdf")
    return str(dst)


def page_image_path(document_id: str, page_number: int, dpi: int) -> Path:
    d = config.CACHE_DIR / "pages" / document_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"p{page_number}@{dpi}.png"


def render_page(pdf_path: str, document_id: str, page_number: int, dpi: int | None = None) -> Path:
    dpi = dpi or config.VIEW_DPI
    out = page_image_path(document_id, page_number, dpi)
    if out.exists():
        return out
    # Ensure source PDF is available (fetch from MinIO if missing locally)
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        download_from_minio_if_needed(pdf_file, f"originals/{document_id}/{pdf_file.name}")
    import pymupdf
    doc = pymupdf.open(pdf_path)
    try:
        pix = doc[page_number - 1].get_pixmap(dpi=dpi)
        pix.save(str(out))
    finally:
        doc.close()
    return out


def crop_path(document_id: str, key: str) -> Path:
    d = config.STORE_DIR / "crops" / document_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.png"


def render_crop(pdf_path: str, document_id: str, page_number: int, bbox: list[float], key: str, dpi: int = 160, pad: float = 6.0) -> Path:
    out = crop_path(document_id, key)
    if out.exists():
        return out
    # Check MinIO if crop was generated by another worker
    if download_from_minio_if_needed(out, f"crops/{document_id}/{key}.png"):
        return out
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        download_from_minio_if_needed(pdf_file, f"originals/{document_id}/{pdf_file.name}")
    import pymupdf
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_number - 1]
        r = pymupdf.Rect(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad) & page.rect
        if r.is_empty or r.width < 2 or r.height < 2:
            r = page.rect
        pix = page.get_pixmap(dpi=dpi, clip=r)
        pix.save(str(out))
        # Sync to MinIO S3 object store
        upload_to_minio(out, f"crops/{document_id}/{key}.png", content_type="image/png")
    finally:
        doc.close()
    return out


def reports_dir() -> Path:
    d = config.STORE_DIR / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_report_artifact(report_path: str | Path, report_id: str) -> None:
    """Save generated DOCX report to persistent store and MinIO."""
    upload_to_minio(report_path, f"reports/{report_id}.docx", content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

