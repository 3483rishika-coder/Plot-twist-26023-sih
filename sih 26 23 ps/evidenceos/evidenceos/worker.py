"""Standalone job worker (used by docker-compose; the dev server runs the same loop in a background thread).

    python -m evidenceos.worker
"""
import time

from . import pipeline
from .db import init_db, session_scope
from .knowledge.entities import seed_entities


def main() -> None:
    init_db()
    with session_scope() as s:
        seed_entities(s)
    print("evidenceos worker started", flush=True)
    while True:
        try:
            n = pipeline.run_pending_jobs(limit=1, progress=lambda k, t, pr: print(f"  page {k}/{t}", flush=True) if k % 10 == 0 else None)
            if n == 0:
                time.sleep(2.0)
        except KeyboardInterrupt:
            break
        except Exception as e:  # keep the worker alive; the job is marked failed/dead by run_pending_jobs
            print("worker error:", e, flush=True)
            time.sleep(3.0)


if __name__ == "__main__":
    main()
