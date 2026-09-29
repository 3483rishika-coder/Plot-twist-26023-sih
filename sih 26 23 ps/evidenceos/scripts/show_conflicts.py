"""Print open conflicts and summary stats (debug helper)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import func
from evidenceos.db import SessionLocal, Conflict, Fact, Table, Entity, Chunk, Validation

s = SessionLocal()
cs = s.query(Conflict).filter(Conflict.status == 'open').order_by(Conflict.code).all()
print("OPEN CONFLICTS:", len(cs))
for c in cs:
    print(f"{c.code} {c.severity:8s} | {c.subject_text} | {c.predicate} | {c.period} | " +
          str([(v['value'], v['document_code'], v['page'], v.get('qualifier')) for v in c.values]))
print()
print("facts:", s.query(Fact).count(), "tables:", s.query(Table).count(), "entities:", s.query(Entity).count(), "chunks:", s.query(Chunk).count())
for st, n in s.query(Fact.review_status, func.count()).group_by(Fact.review_status).all():
    print(' review_status', st, n)
for rid, st, n in s.query(Validation.rule_id, Validation.status, func.count()).group_by(Validation.rule_id, Validation.status).all():
    print(' validation', rid, st, n)
