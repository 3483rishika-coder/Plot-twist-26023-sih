"""Point document storage at data/originals (same SHA-256) and drop duplicate copies; checkpoint + VACUUM the SQLite DB.
Run after a rebuild to keep the workspace small:  python scripts/compact_store.py"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidenceos import config  # noqa: E402
from evidenceos.db import Document, SessionLocal, engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
orig = {}
for f in (ROOT / "data" / "originals").glob("*.pdf"):
    orig[hashlib.sha256(f.read_bytes()).hexdigest()] = f
s = SessionLocal()
moved = 0
for d in s.query(Document).all():
    src = orig.get(d.sha256)
    if src and Path(d.storage_location or "").resolve() != src.resolve():
        old = Path(d.storage_location) if d.storage_location else None
        d.storage_location = str(src.resolve())
        if old and old.exists() and str(config.STORE_DIR) in str(old.resolve()):
            try:
                os.chmod(old, 0o644); old.unlink(); old.parent.rmdir()
            except OSError:
                pass
        moved += 1
s.commit(); s.close()
print("documents re-pointed to data/originals:", moved)
if config.DATABASE_URL.startswith("sqlite"):
    with engine.connect() as c:
        c.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        c.exec_driver_sql("VACUUM")
    db = config.DATA_DIR / "evidenceos.db"
    print("db size MB:", round(db.stat().st_size / 1e6, 1))
