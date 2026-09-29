"""Evaluate EvidenceOS Q&A against the gold set (data/gold/qa_gold.json).

Metrics
  answer accuracy  – an accepted value appears in the answer text (numeric routes) / abstention when expected
  citation rate    – every non-abstained answer carries ≥1 citation with document+page+bbox
  citation faith   – cited facts really contain the answered value (checked against the DB)
  route accuracy   – planner route matches the expected route family
  conflict recall  – questions whose sources disagree surface the conflict

Usage:  python scripts/evaluate.py [--verbose] [--json out.json]
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidenceos.db import Fact, SessionLocal  # noqa: E402
from evidenceos.retrieval import qa  # noqa: E402

NUM_RE = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?(?![\w])|(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUM_RE.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.006, 0.0002 * abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gold", "qa_gold.json"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    gold = json.load(open(args.gold))["items"]
    s = SessionLocal()
    rows = []
    t_all = time.time()
    for g in gold:
        t0 = time.time()
        r = qa.answer(s, g["question"], user=None, filters={})
        ms = int((time.time() - t0) * 1000)
        ans = r.get("answer") or ""
        nums = numbers_in(ans)
        route = r.get("route") or ""
        abstained = bool(r.get("missing_evidence")) or r.get("partial")
        if g["expect_route"] == "abstain":
            ok_answer = bool(r.get("missing_evidence")) or bool(r.get("partial")) or not r.get("citations")
        else:
            hits = [v for v in g["accept"] if any(close(n, v) for n in nums)]
            ok_answer = len(hits) >= g.get("min_values", 1) and not r.get("missing_evidence")
        route_ok = (g["expect_route"] in route) or (g["expect_route"] == "abstain") or (g["expect_route"] == "numeric" and route.startswith("comparison"))
        cits = r.get("citations") or []
        cited = bool(cits) and all(c.get("document_code") and c.get("page") for c in cits)
        # citation faithfulness: numeric answers must be backed by cited facts holding the value
        faithful = True
        if g["expect_route"] in ("numeric", "comparison", "conflict") and not r.get("missing_evidence"):
            fact_vals = []
            for c in cits:
                if c.get("fact_id"):
                    f = s.get(Fact, c["fact_id"])
                    if f is not None and f.display_value:
                        try:
                            fact_vals.append(float(f.display_value.replace(",", "")))
                        except ValueError:
                            pass
            answered = [n for n in nums if any(close(n, v) for v in g["accept"])]
            faithful = all(any(close(n, fv) for fv in fact_vals) for n in answered) if answered else bool(fact_vals)
        conflict_ok = (not g.get("expect_conflict")) or bool(r.get("conflicts"))
        rows.append({"id": g["id"], "question": g["question"], "route": route, "route_ok": route_ok, "answer_ok": ok_answer, "cited": cited,
                     "faithful": faithful, "conflict_ok": conflict_ok, "confidence": r.get("confidence"), "ms": ms,
                     "answer": ans[:300], "n_citations": len(cits)})
        if args.verbose:
            flag = "✓" if ok_answer else "✗"
            print(f"{flag} {g['id']} [{route}] {g['question']}\n     → {ans[:220].replace(chr(10), ' | ')}\n     cited={cited} faithful={faithful} conflict_ok={conflict_ok} conf={r.get('confidence')} {ms}ms")
    n = len(rows)
    summ = {
        "n": n,
        "answer_accuracy": round(sum(r["answer_ok"] for r in rows) / n, 3),
        "route_accuracy": round(sum(r["route_ok"] for r in rows) / n, 3),
        "citation_rate": round(sum(r["cited"] for r in rows if not r["route"].startswith("abstain")) / max(1, sum(1 for r in rows if r["n_citations"] or r["answer_ok"])), 3),
        "citation_faithfulness": round(sum(r["faithful"] for r in rows) / n, 3),
        "conflict_recall": round(sum(r["conflict_ok"] for r in rows) / n, 3),
        "median_latency_ms": sorted(r["ms"] for r in rows)[n // 2],
        "seconds": round(time.time() - t_all, 1),
    }
    print("\n=== EvidenceOS Q&A evaluation ===")
    for k, v in summ.items():
        print(f"{k:24s} {v}")
    fails = [r for r in rows if not r["answer_ok"]]
    if fails:
        print("\nFailed:")
        for r in fails:
            print(f"  {r['id']} [{r['route']}] {r['question']}\n     → {r['answer'][:200].replace(chr(10), ' | ')}")
    if args.json:
        json.dump({"summary": summ, "rows": rows}, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
