#!/usr/bin/env bash
# Full rebuild of the demo corpus (digital + scanned documents). Run from the project root with the venv active.
set -u
cd "$(dirname "$0")/.."
rm -f data/evidenceos.db data/evidenceos.db-shm data/evidenceos.db-wal
rm -rf data/store
python scripts/ingest_dataset.py --ids 04 05 06 07 08 13 14 15
python scripts/ingest_dataset.py --ids 01 --page-range 27 110          # Coal Directory 2024-25: summary + production sections
python scripts/ingest_dataset.py --ids 11 --scanned-pages 0             # scanned promotion orders (full OCR, 21 pages)
python scripts/ingest_dataset.py --ids 12 --scanned-pages 12            # scanned PR agency contract (first 12 pages: signatures / stamps)
python scripts/show_conflicts.py
