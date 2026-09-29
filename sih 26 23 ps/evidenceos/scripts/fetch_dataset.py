"""Download the demo corpus listed in data/manifest.json into data/originals/ (large files into --large-dir).

    python scripts/fetch_dataset.py [--ids 04 05 ...] [--large-dir ~/.cache/evidenceos/large]

Government sites reject bare clients: a browser User-Agent is used and HEAD requests are avoided.  Files that are
already present (same size) are skipped.  cmpdi.co.in blocks some networks with HTTP 403 — the manifest marks the
substitutes used in that case.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--large-dir", default=str(Path.home() / ".cache" / "evidenceos" / "large"))
    args = ap.parse_args()
    manifest = json.load(open(ROOT / "data" / "manifest.json"))
    items = manifest["documents"] if isinstance(manifest, dict) else manifest
    out_dir = ROOT / "data" / "originals"; out_dir.mkdir(parents=True, exist_ok=True)
    large_dir = Path(args.large_dir); large_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for m in items:
        if args.ids and m["id"] not in args.ids:
            continue
        if not m.get("file") or not m.get("url"):
            print(f"[{m['id']}] skipped ({m.get('status', 'no file')})"); continue
        dest = (large_dir if m.get("large") else out_dir) / m["file"]
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[{m['id']}] present: {dest.name}"); ok += 1; continue
        print(f"[{m['id']}] downloading {m['url']}")
        try:
            with httpx.stream("GET", m["url"], headers=UA, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes(1 << 16):
                        f.write(chunk)
            print(f"      saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)"); ok += 1
        except Exception as e:
            print(f"      FAILED: {e}")
            if dest.exists():
                dest.unlink()
    print(f"{ok} files available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
