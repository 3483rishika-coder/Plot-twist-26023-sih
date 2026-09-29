"""Word cloud + topic identification (PRD §11.6.3).

TF-IDF keyphrases with a coal-domain stopword list, NMF topics over page chunks, and topic
trends by document period / publisher.  Results are cached in the analytics table.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .db import Analytics, Document, Page

DOMAIN_STOP = set(ENGLISH_STOP_WORDS) | {
    "mt", "fig", "figs", "qty", "nos", "no", "sl", "sr", "ltd", "limited", "shri", "dated", "page", "annual", "report", "ministry", "coal", "india",
    "govt", "government", "total", "grand", "upto", "wise", "table", "chart", "million", "tonne", "tonnes", "lakh", "crore", "rs", "cum",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december", "fy", "year", "years", "month", "monthly", "provisional", "actual", "growth",
    "achmt", "achievement", "share", "particulars", "company", "companies", "subs", "subsidiary", "subsidiaries", "director", "directory",
    "et", "al", "also", "shall", "may", "per", "etc", "viz", "inter", "alia", "said", "various", "respectively", "following", "given", "including",
}
TOKEN_RE = r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"


def _corpus(session, filters: dict):
    q = session.query(Document).filter(Document.status.in_(["extracted", "validated", "reviewed", "published"]))
    if filters.get("doc_type"):
        q = q.filter(Document.document_type == filters["doc_type"])
    if filters.get("source"):
        q = q.filter(Document.source.ilike(f"%{filters['source']}%"))
    docs = q.all()
    if filters.get("year"):
        y = str(filters["year"])
        docs = [d for d in docs if d.period and (y in d.period or y[-2:] == (d.period[-2:] if d.period else ""))]
    items = []
    for d in docs:
        pages = session.query(Page).filter(Page.document_id == d.document_id).all()
        for p in pages:
            if p.text and len(p.text) > 200:
                items.append({"document_id": d.document_id, "doc_code": d.code, "title": d.title, "period": d.period, "source": d.source,
                              "doc_type": d.document_type, "page": p.page_number, "text": p.text})
    return docs, items


def compute(session, filters: dict | None = None, n_topics: int = 8, n_words: int = 80) -> dict:
    filters = filters or {}
    key = "topics:" + hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()
    docs, items = _corpus(session, filters)
    stamp = hashlib.md5(json.dumps([[d.code, str(d.updated_at)] for d in docs], default=str).encode()).hexdigest()
    cached = session.get(Analytics, key)
    if cached and cached.payload.get("stamp") == stamp:
        return cached.payload
    if len(items) < 3:
        payload = {"stamp": stamp, "words": [], "topics": [], "trend": [], "n_pages": len(items), "n_docs": len(docs), "summary": "Not enough processed text for topic analysis."}
        _save(session, key, payload)
        return payload
    texts = [it["text"] for it in items]
    vec = TfidfVectorizer(stop_words=list(DOMAIN_STOP), token_pattern=TOKEN_RE, ngram_range=(1, 2), min_df=2, max_df=0.85, max_features=6000, sublinear_tf=True)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    # word cloud: corpus-level tf-idf mass
    mass = np.asarray(X.sum(axis=0)).ravel()
    top = np.argsort(-mass)[: n_words * 2]
    words = []
    seen = set()
    for i in top:
        w = vocab[i]
        if any(w in s or s in w for s in seen):
            continue
        seen.add(w)
        words.append({"text": w, "weight": round(float(mass[i]), 4)})
        if len(words) >= n_words:
            break
    # topics via NMF
    k = max(2, min(n_topics, X.shape[0] // 3, 12))
    nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=400)
    W = nmf.fit_transform(X)
    H = nmf.components_
    topics = []
    for t in range(k):
        idx = np.argsort(-H[t])[:10]
        kws = [vocab[i] for i in idx]
        weight = float(W[:, t].sum())
        top_pages = np.argsort(-W[:, t])[:3]
        topics.append({"topic_id": t, "label": _label(kws), "keywords": kws, "weight": round(weight, 3),
                       "examples": [{"doc_code": items[i]["doc_code"], "page": items[i]["page"], "title": items[i]["title"]} for i in top_pages]})
    # trend: mean topic weight per period label (FY or month → FY) and per publisher
    by_period: dict[str, list] = defaultdict(list)
    for i, it in enumerate(items):
        per = _period_bucket(it["period"])
        by_period[per].append(W[i] / (W[i].sum() + 1e-9))
    trend = []
    for per in sorted(by_period):
        arr = np.array(by_period[per])
        trend.append({"period": per, "n_pages": len(arr), "weights": [round(float(x), 4) for x in arr.mean(axis=0)]})
    for t in topics:
        series = [(tr["period"], tr["weights"][t["topic_id"]]) for tr in trend]
        t["trend"] = _arrow(series)
    total_w = sum(t["weight"] for t in topics) or 1
    topics.sort(key=lambda t: -t["weight"])
    summary_lines = [f"{t['label']} ({t['weight'] / total_w * 100:.0f}% of topical mass, trend {t['trend']}): {', '.join(t['keywords'][:5])}" for t in topics[:6]]
    payload = {"stamp": stamp, "words": words, "topics": topics, "trend": trend, "n_pages": len(items), "n_docs": len(docs),
               "summary": f"Top topics across {len(docs)} documents / {len(items)} pages:\n" + "\n".join(summary_lines)}
    _save(session, key, payload)
    return payload


def _save(session, key, payload):
    a = session.get(Analytics, key)
    if a is None:
        session.add(Analytics(key=key, payload=payload))
    else:
        a.payload = payload
    session.commit()


def _label(kws: list[str]) -> str:
    return " / ".join(w.title() for w in kws[:3])


def _period_bucket(period: str | None) -> str:
    if not period:
        return "unknown"
    m = re.match(r"FY(\d{4})-(\d{2})", period)
    if m:
        return f"FY{m.group(1)}-{m.group(2)}"
    m = re.match(r"(\d{4})-(\d{2})$", period)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        sy = y if mo >= 4 else y - 1
        return f"FY{sy}-{(sy + 1) % 100:02d}"
    return period


def _arrow(series):
    vals = [v for _, v in series if v is not None]
    if len(vals) < 2:
        return "→ stable"
    d = vals[-1] - vals[0]
    if d > 0.03:
        return "↑ rising"
    if d < -0.03:
        return "↓ falling"
    return "→ stable"


def topics_summary(session, filters: dict | None = None) -> dict:
    return compute(session, filters)
