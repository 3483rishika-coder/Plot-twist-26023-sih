"""Ingest the real test corpus described in data/manifest.json.

Usage:
  python scripts/ingest_dataset.py                      # all available files, digital docs fully, scanned docs first 40 pages
  python scripts/ingest_dataset.py --ids 04 05 15       # subset
  python scripts/ingest_dataset.py --scanned-pages 0    # no page cap (full OCR; slow on CPU)
  python scripts/ingest_dataset.py --large-dir ~/.cache/evidenceos/large   # where the >100MB directories live
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidenceos import pipeline  # noqa: E402
from evidenceos.db import init_db, session_scope  # noqa: E402
from evidenceos.knowledge.entities import seed_entities  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--scanned-pages", type=int, default=40, help="page cap for scanned documents (0 = all)")
    ap.add_argument("--page-range", nargs=2, type=int, default=None, help="explicit page range for scanned docs")
    ap.add_argument("--large-dir", default=str(Path.home() / ".cache" / "evidenceos" / "large"))
    ap.add_argument("--no-process", action="store_true", help="register only; let the API worker process")
    ap.add_argument("--force", action="store_true", help="re-process documents that were already extracted")
    ap.add_argument("--copy", action="store_true", help="copy originals into data/store instead of registering them in place")
    args = ap.parse_args()

    init_db()
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    with session_scope() as s:
        seed_entities(s)
    for m in manifest["documents"]:
        if args.ids and m["id"] not in args.ids:
            continue
        if not m.get("file"):
            print(f"[{m['id']}] {m['title']}: {m.get('status')} (substitute: #{m.get('substitute')})")
            continue
        path = ROOT / "data" / "originals" / m["file"]
        link = not args.copy          # data/originals/ *is* the immutable store for the demo corpus; --copy duplicates into data/store
        if not path.exists():
            alt = Path(args.large_dir) / m["file"]
            if alt.exists():
                path, link = alt, True
            else:
                print(f"[{m['id']}] missing file {m['file']} — run scripts/fetch_dataset.py")
                continue
        with session_scope() as s:
            doc, created = pipeline.ingest_file(s, path, title=m["title"], source=m["source"], source_url=m["url"], doc_type=m["document_type"],
                                                period=m.get("period"), acl_level=m.get("acl_level", "public"), link=link,
                                                meta={"manifest_id": m["id"], "role": m.get("role")}, enqueue=False)
            doc_id, code, scanned, pages, status = doc.document_id, doc.code, doc.is_scanned, doc.page_count, doc.status
        print(f"[{m['id']}] {code} {'registered' if created else 'already present (' + str(status) + ')'}: {m['title']} ({pages} pages, scanned={scanned})")
        if args.no_process or (not created and status in ("extracted", "validated", "reviewed", "published") and not args.force):
            continue
        t = time.time()
        kw = {}
        if scanned or m.get("large"):
            if args.page_range:
                kw["page_range"] = tuple(args.page_range)
            elif args.scanned_pages:
                kw["max_pages"] = args.scanned_pages

        def prog(k, n, pr):
            if k % 10 == 0 or k == n:
                print(f"      page {k}/{n} {'OCR' if pr.is_scanned else 'digital'} regions={len(pr.regions)} tables={len(pr.tables)}", flush=True)

        try:
            stats = pipeline.process_document(doc_id, progress=prog, **kw)
            print(f"      done in {time.time() - t:.1f}s: {stats}")
        except Exception as e:                       # keep going with the next document; the failure is recorded on the document
            print(f"      FAILED after {time.time() - t:.1f}s: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
