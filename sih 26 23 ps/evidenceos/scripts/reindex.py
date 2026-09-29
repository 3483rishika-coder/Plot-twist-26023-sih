"""Rebuild the retrieval index (chunks + FTS + embeddings) for all documents without re-running extraction."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidenceos.db import Document, SessionLocal
from evidenceos.retrieval import index as ri

s = SessionLocal()
for d in s.query(Document).order_by(Document.created_at).all():
    t = time.time()
    n = ri.index_document(s, d)
    print(f"{d.code}: {n} chunks in {time.time() - t:.1f}s", flush=True)
