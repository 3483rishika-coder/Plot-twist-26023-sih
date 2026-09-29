#!/usr/bin/env bash
# Quick end-to-end check of the running API (default http://localhost:8000). Usage: bash scripts/smoke_api.sh [base_url]
B=${1:-http://localhost:8000}; K="X-API-Key: eos-geologist-key"
set -e
echo "== stats";      curl -sf -H "$K" $B/api/v1/stats | python3 -c "import sys,json; s=json.load(sys.stdin); print({k:s[k] for k in ('documents','pages','tables','facts','conflicts_open')})"
echo "== query";      curl -sf -H "$K" -X POST $B/api/v1/query -H 'Content-Type: application/json' -d '{"query":"What was India'"'"'s coal production in FY 2024-25?"}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['route'], r['confidence'], r['answer'][:160].replace(chr(10),' | '), '| citations:', len(r['citations']))"
echo "== conflicts";  curl -sf -H "$K" "$B/api/v1/conflicts?status=open" | python3 -c "import sys,json; c=json.load(sys.stdin); print(len(c), 'open;', [x['code']+' '+x['subject']+' '+x['predicate'] for x in c[:3]])"
echo "== review";     curl -sf -H "$K" "$B/api/v1/review/queue?limit=3" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['total'], 'queued')"
echo "== report";     curl -sf -H "$K" -X POST $B/api/v1/reports/generate -H 'Content-Type: application/json' -d '{"template_id":"production_summary","params":{"period":"FY2024-25"}}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['title'], r['facts_used'], 'facts', r['download_url'])"
echo "== topics";     curl -sf -H "$K" "$B/api/v1/topics" | python3 -c "import sys,json; t=json.load(sys.stdin); print(t['n_docs'],'docs', [w['text'] for w in t['words'][:8]])"
echo "== page image"; DOC=$(curl -sf -H "$K" $B/api/v1/documents | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['document_id'])"); PID=$(curl -sf -H "$K" $B/api/v1/documents/$DOC/pages | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['page_id'])"); curl -sf -H "$K" -o /tmp/p1.png $B/api/v1/pages/$PID/image && file /tmp/p1.png | cut -c1-80
echo "== acl (viewer must not see private/sensitive docs)"; curl -sf -H "X-API-Key: eos-viewer-key" $B/api/v1/documents | python3 -c "import sys,json; print(sorted({d['acl_level'] for d in json.load(sys.stdin)}))"
echo "== audit (auditor)"; curl -sf -H "X-API-Key: eos-auditor-key" "$B/api/v1/audit?limit=3" | python3 -c "import sys,json; print([a['action'] for a in json.load(sys.stdin)])"
echo OK
