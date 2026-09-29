"""Debug helper: run layout + column classification on one PDF page and print tables/specs.
Usage: python scripts/debug_page.py <pdf> <page_number>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymupdf
from evidenceos.docai import pages as P
from evidenceos.knowledge.facts import classify_columns, DocContext
from evidenceos.pipeline import _intro_text


def show(path, pno, maxrows=6):
    doc = pymupdf.open(path)
    res = P.analyze_page(doc, pno - 1)
    for t in res.tables:
        print(f"--- {os.path.basename(path)} p{pno} caption={t.caption!r} unit_hint={t.unit_hint!r} rows={len(t.rows)} cols={len(t.headers)} bbox={t.bbox}")
        print("   headers:", t.headers)
        for r in t.rows[:maxrows]:
            print("   ", r)
        ctx = DocContext(document_id='x', doc_code='X', doc_type='annual_report', period=None)
        treg = next((r for r in res.regions if r.type == "table" and abs(r.bbox[1] - t.bbox[1]) < 3 and abs(r.bbox[0] - t.bbox[0]) < 3), None)
        intro = _intro_text(res.regions, treg.bbox if treg else t.bbox)
        if intro:
            print("   intro:", repr(intro[-160:]))
        specs, ec, pc, tp = classify_columns(t.headers, t.rows, t.caption or '', t.unit_hint or '', ctx, intro=intro, page_text=res.text or '')
        print("   entity_col", ec, "parent_col", pc, "specs:", [(s.index, s.kind, s.predicate, s.qualifier, s.period.key if s.period else None) for s in specs if s.kind != 'ignore'])


if __name__ == "__main__":
    show(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 6)
