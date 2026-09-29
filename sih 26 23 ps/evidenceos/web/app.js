/* EvidenceOS web UI — vanilla JS single-page app talking to /api/v1 (no build step).
   Views: dashboard · documents/upload · evidence viewer (page image + bbox overlays) · ask · conflicts · review · reports · topics · audit */
(function () {
  "use strict";
  const $ = (sel, el = document) => el.querySelector(sel);
  const app = $("#app");
  const state = { apiKey: localStorage.getItem("eos.apiKey") || "eos-officer-key", me: null, roles: [], docs: null };

  // ------------------------------------------------------------------ helpers
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function toast(msg, bad) { const t = $("#toast"); t.textContent = msg; t.hidden = false; t.className = "toast" + (bad ? " bad" : ""); clearTimeout(t._h); t._h = setTimeout(() => t.hidden = true, 3500); }
  async function api(path, opts = {}) {
    const headers = Object.assign({ "X-API-Key": state.apiKey }, opts.headers || {});
    if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); opts.method = opts.method || "POST"; }
    const r = await fetch(path, Object.assign({}, opts, { headers }));
    if (!r.ok) { let d = ""; try { d = (await r.json()).detail; } catch (e) { d = r.statusText; } throw new Error(typeof d === "string" ? d : JSON.stringify(d)); }
    const ct = r.headers.get("content-type") || "";
    return ct.includes("json") ? r.json() : r;
  }
  function imgUrl(path) { return path + (path.includes("?") ? "&" : "?") + "k=" + encodeURIComponent(state.apiKey); }
  function fmtNum(v, d = 2) { if (v == null || isNaN(v)) return "—"; return Number(v).toLocaleString("en-IN", { maximumFractionDigits: d }); }
  function confBar(c) { const p = Math.round((c || 0) * 100); const cls = p >= 90 ? "" : p >= 70 ? "mid" : "low"; return `<span class="conf"><span class="bar ${cls}"><i style="width:${p}%"></i></span><small>${p}%</small></span>`; }
  function statusPill(s) { const m = { valid: "ok", verified: "ok", conflict: "warn", low_confidence: "warn", rejected: "bad", pending: "", extracted: "ok", processing: "info", queued: "info", failed: "bad", open: "warn", resolved: "ok", ignored: "" }; return `<span class="pill ${m[s] || ""}">${esc(s || "—")}</span>`; }
  function periodLabel(p) { if (!p) return "—"; if (p.startsWith("asof:")) return "as on " + p.slice(5).split("-").reverse().join("."); if (p.includes(":ytd:")) { const [fy, , m] = p.split(":"); return `${fy} (upto ${monthName(m)})`; } if (p.includes(":range:")) { const [fy, , r] = p.split(":"); const [a, b] = r.split(".."); return `${monthName(a)}–${monthName(b)}`; } if (/^\d{4}-\d{2}$/.test(p)) return monthName(p); if (p.startsWith("Y")) return p.slice(1); return p; }
  function monthName(ym) { const [y, m] = ym.split("-"); const n = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][parseInt(m, 10) - 1]; return `${n}'${y.slice(2)}`; }
  function viewerLink(c, label) { const q = []; if (c.region_id) q.push("region=" + c.region_id); if (c.fact_id) q.push("fact=" + c.fact_id); if (c.table_id) q.push("table=" + c.table_id); return `#/viewer/${c.document_id}/${c.page || c.page_number || 1}${q.length ? "?" + q.join("&") : ""}`; }
  function debounce(fn, ms) { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; }

  // ------------------------------------------------------------------ auth / roles
  async function loadMe() {
    try { state.me = await api("/api/v1/me"); } catch (e) { state.apiKey = "eos-viewer-key"; state.me = await api("/api/v1/me"); }
    $("#whoami").textContent = `${state.me.username} · ${state.me.permissions.join(", ")}`;
    try { state.roles = await api("/api/v1/roles"); } catch (e) { state.roles = []; }
    const sel = $("#roleSelect"); sel.innerHTML = state.roles.map(r => `<option value="${esc(r.api_key)}" ${r.api_key === state.apiKey ? "selected" : ""}>${esc(r.role)}</option>`).join("");
    sel.onchange = () => { state.apiKey = sel.value; localStorage.setItem("eos.apiKey", sel.value); state.docs = null; loadMe().then(route); };
  }
  function can(perm) { return state.me && (state.me.permissions.includes("*") || state.me.permissions.includes(perm)); }

  // ------------------------------------------------------------------ router
  const routes = { dashboard, documents, viewer, ask, conflicts, review, reports, topics, audit, document: documentDetail, fact: factDetail };
  function parseHash() { const h = location.hash.replace(/^#\/?/, "") || "dashboard"; const [pathPart, qs] = h.split("?"); const parts = pathPart.split("/"); const q = {}; (qs || "").split("&").filter(Boolean).forEach(kv => { const [k, v] = kv.split("="); q[decodeURIComponent(k)] = decodeURIComponent(v || ""); }); return { name: parts[0], args: parts.slice(1), q }; }
  let routeSeq = 0;
  async function route() {
    const token = ++routeSeq;
    const { name, args, q } = parseHash();
    document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", a.dataset.route === name || (name === "viewer" && a.dataset.route === "documents") || (name === "document" && a.dataset.route === "documents")));
    const fn = routes[name] || dashboard;
    app.innerHTML = '<div class="loading">Loading…</div>';
    try { await fn(args, q, () => token === routeSeq); } catch (e) { if (token === routeSeq) app.innerHTML = `<div class="panel"><h2>Error</h2><p class="muted">${esc(e.message)}</p></div>`; }
  }
  window.addEventListener("hashchange", route);

  // ------------------------------------------------------------------ dashboard
  async function dashboard() {
    const [s, docs, conflicts] = await Promise.all([api("/api/v1/stats"), api("/api/v1/documents"), api("/api/v1/conflicts?status=open")]);
    const byStatus = s.facts_by_status || {}; const byReview = s.facts_by_review || {};
    const needReview = (byReview.mandatory || 0) + (byReview.recommended || 0);
    app.innerHTML = `
      <div class="row between"><h1>Evidence overview</h1><div class="row"><span class="pill ${s.llm_configured ? "ok" : ""}">LLM ${s.llm_configured ? "configured" : "off · extractive answers"}</span><span class="pill ${s.embeddings ? "ok" : ""}">embeddings ${s.embeddings ? "on" : "off"}</span></div></div>
      <div class="grid cols-6">
        ${stat(s.documents, "documents")}${stat(s.pages, "pages processed", `${s.scanned_pages} OCR`)}${stat(s.tables, "tables reconstructed")}
        ${stat(s.facts, "evidence-linked facts")}${stat(s.conflicts_open, "open conflicts", "", s.conflicts_open ? "warn" : "")}${stat(needReview, "facts awaiting review", "", needReview ? "warn" : "")}
      </div>
      <div class="grid cols-2" style="margin-top:14px">
        <div class="panel"><h3>Validation status</h3>${Object.keys(byStatus).length ? Object.entries(byStatus).map(([k, v]) => `<div class="row between"><span>${statusPill(k)}</span><b>${fmtNum(v, 0)}</b></div>`).join("") : '<div class="empty">No facts yet — upload a document.</div>'}
          <h3 style="margin-top:14px">Confidence policy</h3><small>&gt; 0.90 auto-accept · 0.70–0.90 review recommended · &lt; 0.70 mandatory review. Confidence = OCR × layout × structure × entity × validation × source agreement.</small></div>
        <div class="panel"><h3>Latest open conflicts</h3>${conflicts.length ? conflicts.slice(0, 6).map(c => `<div class="row between clickable" onclick="location.hash='#/conflicts'"><span>${esc(c.code)} <b>${esc(c.subject)}</b> — ${esc(c.predicate)} ${esc(periodLabel(c.period))}</span><span>${c.values.map(v => esc(v.value)).join(" vs ")} <span class="pill ${c.severity === "critical" ? "bad" : "warn"}">${esc(c.severity)}</span></span></div>`).join("") : '<div class="empty">No open conflicts — all cross-document comparisons agree within tolerance.</div>'}</div>
      </div>
      <div class="panel" style="margin-top:14px"><div class="row between"><h3>Documents</h3><a href="#/documents">manage / upload →</a></div>
        <table class="tbl"><thead><tr><th>Code</th><th>Title</th><th>Type</th><th>Period</th><th>ACL</th><th class="num">Pages</th><th class="num">Tables</th><th class="num">Facts</th><th>Status</th></tr></thead><tbody>
        ${docs.map(d => `<tr class="clickable" onclick="location.hash='#/document/${d.document_id}'"><td class="mono">${esc(d.code)}</td><td>${esc(d.title)} ${d.is_scanned ? '<span class="badge-scan">scanned</span>' : ""}</td><td>${esc(d.document_type)}</td><td>${esc(d.period || "")}</td><td>${esc(d.acl_level)}</td><td class="num">${d.pages_processed}/${d.page_count}</td><td class="num">${d.tables}</td><td class="num">${d.facts}</td><td>${statusPill(d.status)}</td></tr>`).join("")}
        </tbody></table></div>
      <div class="panel" style="margin-top:14px"><h3>Try the demos</h3><div class="chips">
        ${["What was India's coal production in FY 2024-25?", "Compare CIL coal production in FY 2023-24 and FY 2024-25", "Which documents disagree on India's dispatch figures?", "What are the total coal resources of Jharkhand as on 1.4.2024?", "How much coal was produced by Gevra mine in 2024-25?", "What was the coal dispatch to the power sector during March 2025?"].map(q => `<span class="chip" onclick="location.hash='#/ask?q=${encodeURIComponent(q)}'">${esc(q)}</span>`).join("")}
      </div></div>`;
    function stat(n, l, sub, cls) { return `<div class="panel stat"><div class="n ${cls ? "" : ""}" style="${cls === "warn" && n ? "color:var(--warn)" : ""}">${fmtNum(n, 0)}</div><div class="l">${l}${sub ? ` · <span>${sub}</span>` : ""}</div></div>`; }
  }

  // ------------------------------------------------------------------ documents + upload
  async function documents() {
    const docs = await api("/api/v1/documents");
    app.innerHTML = `
      <div class="row between"><h1>Documents</h1><small>Originals are immutable (SHA-256 fingerprinted); every fact links back to page + region + bbox.</small></div>
      ${can("upload") ? `<div class="panel" id="uploadPanel"><h3>Upload a PDF (digital or scanned)</h3>
        <div class="dropzone" id="drop">Drop a PDF here or <label style="color:var(--brand);cursor:pointer"><input type="file" id="file" accept="application/pdf" hidden>browse</label>
        <div class="row" style="justify-content:center;margin-top:10px">
          <input id="upTitle" placeholder="Title (optional)" style="width:260px"><select id="upType"><option value="">auto type</option><option>monthly_statistics</option><option>annual_report</option><option>coal_directory</option><option>inventory</option><option>geological_report</option><option>administrative</option></select>
          <input id="upPeriod" placeholder="Period e.g. FY2024-25" style="width:150px"><select id="upAcl"><option value="public">public</option><option value="private">private</option><option value="sensitive">sensitive</option></select>
          <input id="upMax" type="number" min="0" placeholder="max pages (0=all)" style="width:150px" title="Cap pages for very large scanned documents"></div></div>
        <div id="upStatus" class="muted" style="margin-top:8px"></div></div>` : '<div class="panel muted">Your role cannot upload documents (RBAC). Switch to geologist/admin to try.</div>'}
      <div class="grid cols-3" style="margin-top:14px" id="docGrid">${docs.map(docCard).join("") || '<div class="empty">No documents yet.</div>'}</div>`;
    if (can("upload")) wireUpload();
    // poll processing docs
    if (docs.some(d => ["queued", "processing"].includes(d.status))) setTimeout(() => { if (parseHash().name === "documents") documents(); }, 4000);
  }
  function docCard(d) {
    const pct = d.page_count ? Math.round(100 * d.pages_processed / d.page_count) : 0;
    return `<div class="panel doc-card" onclick="location.hash='#/document/${d.document_id}'">
      <div class="row between"><span class="mono muted">${esc(d.code)} · v${d.version}</span>${statusPill(d.status)}</div>
      <div class="t" style="margin:6px 0">${esc(d.title)}</div>
      <div class="row"><span class="tag">${esc(d.document_type)}</span>${d.period ? `<span class="tag">${esc(d.period)}</span>` : ""}<span class="tag">${esc(d.acl_level)}</span>${d.is_scanned ? '<span class="tag handwriting">scanned · OCR</span>' : ""}</div>
      <div class="progress" style="margin:8px 0 4px"><i style="width:${pct}%"></i></div>
      <small>${d.pages_processed}/${d.page_count} pages · ${d.tables} tables · ${d.facts} facts${d.job && d.job.status === "running" ? " · processing…" : ""}${d.error ? ` · <span style="color:var(--bad)">${esc(d.error)}</span>` : ""}</small></div>`;
  }
  function wireUpload() {
    const drop = $("#drop"), file = $("#file");
    ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", e => { if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
    file.addEventListener("change", () => { if (file.files[0]) upload(file.files[0]); });
    async function upload(f) {
      const fd = new FormData(); fd.append("file", f);
      if ($("#upTitle").value) fd.append("title", $("#upTitle").value);
      if ($("#upType").value) fd.append("document_type", $("#upType").value);
      if ($("#upPeriod").value) fd.append("period", $("#upPeriod").value);
      fd.append("acl_level", $("#upAcl").value); fd.append("max_pages", $("#upMax").value || "0");
      $("#upStatus").textContent = `Uploading ${f.name}…`;
      try {
        const r = await api("/api/v1/documents/upload", { method: "POST", body: fd });
        $("#upStatus").innerHTML = r.duplicate ? `Duplicate of existing document <b>${esc(r.code)}</b> (same SHA-256) — not re-ingested.` : `Registered <b>${esc(r.code)}</b> (${r.page_count} pages${r.is_scanned ? ", scanned → OCR" : ""}). Processing in background…`;
        setTimeout(documents, 1500);
      } catch (e) { $("#upStatus").textContent = "Upload failed: " + e.message; toast(e.message, true); }
    }
  }

  async function documentDetail(args) {
    const id = args[0];
    const [d, pages] = await Promise.all([api(`/api/v1/documents/${id}`), api(`/api/v1/documents/${id}/pages`)]);
    const facts = await api(`/api/v1/facts?document_id=${id}&limit=25`);
    app.innerHTML = `
      <div class="row between"><h1>${esc(d.title)}</h1><div class="row"><button onclick="location.hash='#/viewer/${id}/1'" class="primary">Open evidence viewer</button>${can("upload") ? `<button id="reproc">Re-process</button>` : ""}</div></div>
      <div class="grid cols-2">
        <div class="panel"><h3>Document</h3><dl class="kv">
          <dt>Code</dt><dd class="mono">${esc(d.code)} (v${d.version})</dd><dt>File</dt><dd>${esc(d.filename)} · ${fmtNum((d.file_size || 0) / 1e6, 1)} MB</dd>
          <dt>SHA-256</dt><dd class="mono" style="word-break:break-all">${esc(d.sha256)}</dd><dt>Source</dt><dd>${esc(d.source || "—")} ${d.source_url ? `· <a href="${esc(d.source_url)}" target="_blank" rel="noopener">original URL</a>` : ""}</dd>
          <dt>Type / period</dt><dd>${esc(d.document_type)} · ${esc(d.period || "—")}</dd><dt>ACL</dt><dd>${esc(d.acl_level)}</dd>
          <dt>Pages</dt><dd>${pages.length}/${d.page_count} processed · ${pages.filter(p => p.is_scanned).length} scanned (OCR)</dd><dt>Status</dt><dd>${statusPill(d.status)} ${d.error ? `<span style="color:var(--bad)">${esc(d.error)}</span>` : ""}</dd>
          <dt>Pipeline stats</dt><dd class="mono">${esc(JSON.stringify((d.meta || {}).stats || {}))}</dd></dl></div>
        <div class="panel"><h3>Pages</h3><div class="scroll" style="max-height:320px"><table class="tbl"><thead><tr><th>#</th><th>Mode</th><th>Quality</th><th class="num">Regions</th><th class="num">Tables</th><th class="num">Facts</th></tr></thead><tbody>
          ${pages.map(p => `<tr class="clickable" onclick="location.hash='#/viewer/${id}/${p.page_number}'"><td>${p.page_number}</td><td>${p.is_scanned ? `<span class="tag handwriting">OCR ${p.ocr_confidence != null ? Math.round(p.ocr_confidence * 100) + "%" : ""}</span>` : '<span class="tag">digital</span>'}</td><td><small>${qualityText(p.quality)}</small></td><td class="num">${p.n_regions}</td><td class="num">${p.n_tables}</td><td class="num">${p.n_facts}</td></tr>`).join("")}
        </tbody></table></div></div>
      </div>
      <div class="panel" style="margin-top:14px"><div class="row between"><h3>Recent facts (${facts.total})</h3><a href="#/review">review queue →</a></div>${factTable(facts.items)}</div>`;
    if ($("#reproc")) $("#reproc").onclick = async () => { await api(`/api/v1/documents/${id}/reprocess`, { method: "POST" }); toast("Re-processing queued"); };
  }
  function qualityText(q) { if (!q) return ""; const parts = []; if (q.blank) parts.push("blank page"); if (q.digital === false) { if (q.blur != null) parts.push(`blur ${Number(q.blur).toFixed(2)}`); if (q.noise != null) parts.push(`noise ${Number(q.noise).toFixed(2)}`); if (q.contrast != null) parts.push(`contrast ${Number(q.contrast).toFixed(2)}`); if (q.skew != null && Math.abs(q.skew) > 0.3) parts.push(`skew ${Number(q.skew).toFixed(1)}°`); if (q.handwriting_probability != null) parts.push(`handwriting p=${Number(q.handwriting_probability).toFixed(2)}`); } if (q.handwriting_regions) parts.push(`${q.handwriting_regions} handwriting/stamp region${q.handwriting_regions > 1 ? "s" : ""} flagged`); if (q.actions && q.actions.length) parts.push(`actions: ${q.actions.join(", ")}`); return parts.join(" · "); }
  function factTable(items) {
    if (!items.length) return '<div class="empty">No facts.</div>';
    return `<div class="scroll"><table class="tbl"><thead><tr><th>Subject</th><th>Metric</th><th>Period</th><th class="num">Value</th><th>Qualifier</th><th>Confidence</th><th>Status</th><th>Evidence</th></tr></thead><tbody>
      ${items.map(f => `<tr><td><b>${esc(f.subject)}</b>${f.dimension ? ` <small>→ ${esc(f.dimension)}</small>` : ""}</td><td>${esc(f.predicate)}</td><td>${esc(periodLabel(f.period))}</td><td class="num">${esc(f.display)}</td><td>${esc(f.qualifier)}</td><td>${confBar(f.overall_confidence)}</td><td>${statusPill(f.validation_status)}</td><td><a href="${viewerLink(f)}">${esc(f.document_code)} p${f.page_number}</a> · <a href="#/fact/${f.fact_id}">detail</a></td></tr>`).join("")}
    </tbody></table></div>`;
  }

  // ------------------------------------------------------------------ evidence viewer
  async function viewer(args, q) {
    const docId = args[0]; let pno = parseInt(args[1] || "1", 10);
    const data = await api(`/api/v1/documents/${docId}/pages/${pno}`);
    if (!data.processed) { app.innerHTML = `<div class="panel"><h2>Page ${pno} not processed yet</h2><p class="muted">The document has ${data.page_count} pages; this page has not been processed (page cap or still running). <a href="#/document/${docId}">Back to document</a></p></div>`; return; }
    const regions = data.regions, tables = data.tables, facts = data.facts;
    const hlRegion = q.region || null, hlFact = q.fact || null, hlTable = q.table || null;
    const W = data.width, H = data.height;
    const rect = (b, cls, title, extra = "") => b ? `<rect x="${b[0]}" y="${b[1]}" width="${Math.max(b[2] - b[0], 1)}" height="${Math.max(b[3] - b[1], 1)}" class="${cls}" ${extra}><title>${esc(title)}</title></rect>` : "";
    const scannedPill = data.is_scanned ? `<span class="pill warn">OCR page · ${data.ocr_confidence != null ? Math.round(data.ocr_confidence * 100) + "% mean confidence" : ""}</span>` : '<span class="pill ok">digital text layer</span>';
    app.innerHTML = `
      <div class="row between" style="margin-bottom:10px">
        <div class="row"><a href="#/document/${docId}">← ${esc(data.document_code)}</a><h1 style="margin:0;font-size:16px">${esc(data.title)}</h1>${scannedPill}${qualityText(data.quality) ? `<small>${qualityText(data.quality)}</small>` : ""}</div>
        <div class="pager"><button id="prev" ${pno <= 1 ? "disabled" : ""}>‹</button><input id="pnum" type="number" min="1" max="${data.page_count}" value="${pno}"><span class="muted">/ ${data.page_count}</span><button id="next" ${pno >= data.page_count ? "disabled" : ""}>›</button>
          <label style="margin-left:10px"><input type="checkbox" id="showText" checked> text regions</label><label><input type="checkbox" id="showCells"> cells</label><label><input type="checkbox" id="showFacts" checked> facts</label></div>
      </div>
      <div class="viewer">
        <div class="page-wrap"><div class="page-inner" id="pageInner"><img id="pageImg" src="${imgUrl(data.image_url)}" alt="page ${pno}">
          <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" id="overlay">
            ${regions.map(r => rect(r.bbox, "reg " + r.type + (r.region_id === hlRegion ? " hl" : ""), `${r.type} (${Math.round((r.confidence || 0) * 100)}%) ${(r.text || "").slice(0, 120)}`, `data-region="${r.region_id}"`)).join("")}
            ${tables.map(t => t.cells.map(c => rect(c.bbox, "cell" + (t.table_id === hlTable ? " hl" : ""), c.text, `data-table="${t.table_id}"`)).join("")).join("")}
            ${facts.map(f => rect(f.bbox, "fact" + (f.fact_id === hlFact ? " hl" : ""), `${f.subject} ${f.predicate} ${periodLabel(f.period)} = ${f.display}`, `data-fact="${f.fact_id}"`)).join("")}
          </svg></div></div>
        <div class="side">
          <div class="panel tight legend"><span><i style="border-color:#1d4ed8"></i>table</span><span><i style="border-color:#64748b"></i>text</span><span><i style="border-color:#7c3aed"></i>title</span><span><i style="border-color:#be185d"></i>figure/chart</span><span><i style="border-color:#b45309;border-style:dashed"></i>handwriting</span><span><i style="border-color:#0f766e;background:rgba(15,118,110,.15)"></i>fact</span><span><i style="border-color:#c2410c"></i>highlighted</span></div>
          <div class="panel tight" id="selBox"><h3>Selection</h3><div class="muted">Click a box on the page, or a fact below. ${hlFact ? "Highlighted: cited fact." : hlRegion ? "Highlighted: cited region." : ""}</div></div>
          <div class="panel tight"><h3>Facts on this page (${facts.length})</h3><div class="scroll" style="max-height:34vh" id="factList">
            ${facts.length ? facts.map(f => `<div class="fact-line ${f.fact_id === hlFact ? "active" : ""}" data-fact="${f.fact_id}"><b>${esc(f.subject)}</b>${f.dimension ? ` → ${esc(f.dimension)}` : ""} · ${esc(f.predicate)} · ${esc(periodLabel(f.period))}${f.qualifier && f.qualifier !== "actual" ? ` (${esc(f.qualifier)})` : ""}<br><span class="v">${esc(f.display)}</span> <small>orig “${esc(f.original_value)}” ${esc(f.original_unit || "")}</small> ${confBar(f.overall_confidence)} ${statusPill(f.validation_status)}</div>`).join("") : '<div class="muted">No numeric facts extracted from this page.</div>'}
          </div></div>
          <div class="panel tight"><h3>Tables (${tables.length})</h3>${tables.map(t => `<details><summary>${esc(t.caption || "(no caption)")} · ${t.n_rows}×${t.n_cols} · ${esc(t.method)} ${t.table_group_id ? '<span class="tag">multi-page group</span>' : ""}</summary><div class="scroll" style="max-height:260px"><table class="tbl"><thead><tr>${(t.headers || []).map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${(t.rows || []).map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div><small><a href="/api/v1/tables/${t.table_id}" target="_blank">JSON</a></small></details>`).join("") || '<div class="muted">No tables on this page.</div>'}</div>
          <div class="panel tight"><details><summary>Page text (${data.is_scanned ? "OCR" : "text layer"})</summary><pre style="white-space:pre-wrap;font-size:12px">${esc(data.text || "")}</pre></details></div>
        </div>
      </div>`;
    const go = n => { location.hash = `#/viewer/${docId}/${n}`; };
    $("#prev").onclick = () => go(pno - 1); $("#next").onclick = () => go(pno + 1);
    $("#pnum").onchange = () => go(Math.min(data.page_count, Math.max(1, parseInt($("#pnum").value, 10) || 1)));
    const svg = $("#overlay");
    const toggle = (cls, on) => svg.querySelectorAll("rect." + cls).forEach(r => r.style.display = on ? "" : "none");
    $("#showText").onchange = e => ["text", "title", "header", "footer", "page_number", "caption"].forEach(c => toggle(c, e.target.checked));
    $("#showCells").onchange = e => toggle("cell", e.target.checked); toggle("cell", false);
    $("#showFacts").onchange = e => toggle("fact", e.target.checked);
    svg.addEventListener("click", ev => {
      const r = ev.target.closest("rect"); if (!r) return;
      svg.querySelectorAll("rect.hl").forEach(x => x.classList.remove("hl")); r.classList.add("hl");
      if (r.dataset.fact) showFact(r.dataset.fact); else if (r.dataset.region) showRegion(r.dataset.region); else if (r.dataset.table) showTable(r.dataset.table);
    });
    $("#factList").addEventListener("click", ev => { const l = ev.target.closest(".fact-line"); if (!l) return; $("#factList").querySelectorAll(".active").forEach(x => x.classList.remove("active")); l.classList.add("active"); svg.querySelectorAll("rect.hl").forEach(x => x.classList.remove("hl")); const rr = svg.querySelector(`rect[data-fact="${l.dataset.fact}"]`); if (rr) { rr.classList.add("hl"); rr.scrollIntoView({ block: "center", behavior: "smooth" }); } showFact(l.dataset.fact); });
    if (hlFact) showFact(hlFact); else if (hlRegion) showRegion(hlRegion); else if (hlTable) showTable(hlTable);
    const hlEl = svg.querySelector("rect.hl"); if (hlEl) setTimeout(() => hlEl.scrollIntoView({ block: "center" }), 200);

    async function showFact(fid) {
      const f = await api(`/api/v1/facts/${fid}`);
      $("#selBox").innerHTML = `<h3>Fact ${esc(f.code)}</h3>
        <div><b>${esc(f.subject)}</b>${f.dimension ? ` → ${esc(f.dimension)}` : ""} · ${esc(f.predicate)} · ${esc(periodLabel(f.period))} (${esc(f.qualifier)})</div>
        <div style="font-size:22px;font-weight:700;margin:4px 0">${esc(f.display)}</div>
        <img src="${imgUrl(f.crop_url)}" style="width:100%;border:1px solid var(--line);border-radius:4px" alt="evidence crop">
        <small>original “${esc(f.original_value)}” ${esc(f.original_unit || "")} · ${esc(f.extraction_method)} · row/cell ${f.cell_id ? "linked" : "n/a"}</small>
        <div style="margin-top:6px">${confBar(f.overall_confidence)} ${statusPill(f.validation_status)} ${f.review_status !== "none" ? `<span class="pill warn">review ${esc(f.review_status)}</span>` : ""}</div>
        <details style="margin-top:6px"><summary>confidence breakdown</summary><dl class="kv">${Object.entries(f.confidence || {}).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${typeof v === "number" ? v.toFixed(2) : esc(v)}</dd>`).join("")}</dl></details>
        <details open style="margin-top:6px"><summary>validations (${f.validations.length})</summary>${f.validations.map(v => `<div><span class="pill ${v.status === "pass" ? "ok" : v.status === "fail" ? "bad" : "warn"}">${esc(v.rule_id)}</span> <small>${esc(v.message)}</small></div>`).join("") || "<small>none</small>"}</details>
        ${f.conflicts.length ? `<details open style="margin-top:6px"><summary style="color:var(--warn)">conflicts (${f.conflicts.length})</summary>${f.conflicts.map(c => `<div><b>${esc(c.code)}</b> ${c.values.map(v => `${esc(v.value)} (${esc(v.document_code)} p${v.page})`).join(" vs ")}<br><small>${esc(c.explanation)}</small></div>`).join("")}</details>` : ""}
        ${can("review") ? `<div class="row" style="margin-top:8px"><button class="small primary" data-act="accept">Accept</button><button class="small" data-act="edit">Edit value</button><button class="small danger" data-act="reject">Reject</button></div>` : ""}
        <div style="margin-top:6px"><a href="#/fact/${f.fact_id}">full detail →</a></div>`;
      $("#selBox").querySelectorAll("button[data-act]").forEach(b => b.onclick = () => reviewAction(f, b.dataset.act, () => showFact(fid)));
    }
    function showRegion(rid) {
      const r = regions.find(x => x.region_id === rid); if (!r) return;
      $("#selBox").innerHTML = `<h3>Region · ${esc(r.type)}</h3><img src="${imgUrl(`/api/v1/regions/${rid}/crop`)}" style="width:100%;border:1px solid var(--line);border-radius:4px"><small>reading order ${r.reading_order} · confidence ${Math.round((r.confidence || 0) * 100)}% · bbox [${r.bbox.map(v => v.toFixed(0)).join(", ")}] pt${r.meta && r.meta.needs_review ? " · <b style='color:var(--warn)'>handwriting — needs human review</b>" : ""}</small><pre style="white-space:pre-wrap;font-size:12px;max-height:200px;overflow:auto">${esc(r.text || "")}</pre>`;
    }
    function showTable(tid) { const t = tables.find(x => x.table_id === tid); if (!t) return; $("#selBox").innerHTML = `<h3>Table</h3><div>${esc(t.caption || "(no caption)")}</div><small>${t.n_rows} rows × ${t.n_cols} cols · ${esc(t.method)} · confidence ${Math.round((t.confidence || 0) * 100)}%${t.table_group_id ? " · part of a multi-page table group" : ""}</small>`; }
  }

  async function reviewAction(f, act, done) {
    let body = { action: act };
    if (act === "edit") { const v = prompt(`New value for ${f.subject} ${f.predicate} ${periodLabel(f.period)} (in ${f.original_unit || f.unit}):`, f.display_value); if (v == null) return; body.value = v; body.note = prompt("Note (optional)") || ""; }
    if (act === "reject") { body.note = prompt("Reason for rejection (optional)") || ""; }
    try { await api(`/api/v1/review/${f.fact_id}`, { json: body }); toast(`Fact ${act}ed — audit logged`); done && done(); } catch (e) { toast(e.message, true); }
  }

  async function factDetail(args) {
    const f = await api(`/api/v1/facts/${args[0]}`);
    app.innerHTML = `<div class="row between"><h1>Fact ${esc(f.code)}</h1><a href="${viewerLink(f)}" class="btn">open in viewer</a></div>
      <div class="grid cols-2"><div class="panel"><h3>Statement</h3><div style="font-size:18px"><b>${esc(f.subject)}</b>${f.dimension ? ` → ${esc(f.dimension)}` : ""} · ${esc(f.predicate)} · ${esc(periodLabel(f.period))} (${esc(f.qualifier)}) = <b>${esc(f.display)}</b></div>
        <dl class="kv" style="margin-top:10px"><dt>Canonical value</dt><dd>${esc(f.value)} ${esc(f.unit)}</dd><dt>Original</dt><dd>“${esc(f.original_value)}” ${esc(f.original_unit || "")}</dd><dt>Context</dt><dd>${esc(f.context)}</dd><dt>Extraction</dt><dd>${esc(f.extraction_method)}</dd><dt>Confidence</dt><dd>${confBar(f.overall_confidence)}</dd><dt>Status</dt><dd>${statusPill(f.validation_status)} review: ${esc(f.review_status)}</dd></dl>
        <h3 style="margin-top:12px">Confidence dimensions</h3><dl class="kv">${Object.entries(f.confidence || {}).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${typeof v === "number" ? v.toFixed(3) : esc(v)}</dd>`).join("")}</dl>
        ${can("review") ? `<div class="row" style="margin-top:10px"><button class="primary" data-act="accept">Accept</button><button data-act="edit">Edit value</button><button class="danger" data-act="reject">Reject</button></div>` : ""}</div>
      <div class="panel"><h3>Evidence</h3><img src="${imgUrl(f.crop_url)}" style="width:100%;border:1px solid var(--line);border-radius:4px"><dl class="kv" style="margin-top:8px"><dt>Document</dt><dd>${esc(f.evidence.document_code)} — ${esc(f.evidence.title)}</dd><dt>SHA-256</dt><dd class="mono" style="word-break:break-all">${esc(f.evidence.sha256)}</dd><dt>Page</dt><dd>${f.evidence.page}</dd><dt>Region / table / cell</dt><dd class="mono">${esc(f.evidence.region_id || "—")}<br>${esc(f.evidence.table_id || "—")}<br>${esc(f.evidence.cell_id || "—")}</dd><dt>bbox (pt)</dt><dd class="mono">${esc(JSON.stringify(f.bbox))}</dd></dl></div></div>
      <div class="grid cols-2" style="margin-top:14px"><div class="panel"><h3>Validations</h3>${f.validations.map(v => `<div style="padding:4px 0;border-bottom:1px solid var(--line)"><span class="pill ${v.status === "pass" ? "ok" : v.status === "fail" ? "bad" : "warn"}">${esc(v.rule_id)} ${esc(v.rule_type)}</span> ${esc(v.message)}</div>`).join("") || '<div class="muted">none</div>'}</div>
      <div class="panel"><h3>Conflicts & corrections</h3>${f.conflicts.map(c => `<div class="conflict-card panel tight ${c.severity}" style="margin-bottom:8px"><b>${esc(c.code)}</b> ${esc(c.severity)} · ${esc(c.status)}<div class="values" style="margin-top:6px">${c.values.map(v => `<div class="value-box" onclick="location.hash='${viewerLink(v)}'"><div class="v">${esc(v.value)}</div><small>${esc(v.document_code)} p${v.page} · ${esc(v.qualifier)}</small></div>`).join("")}</div><small>${esc(c.explanation)}</small></div>`).join("") || '<div class="muted">no conflicts</div>'}
        ${f.corrections.length ? `<h3 style="margin-top:10px">Corrections</h3>${f.corrections.map(c => `<div><small>${esc(c.at)}</small> <b>${esc(c.action)}</b> ${esc(c.old_value)} → ${esc(c.new_value)} ${c.note ? `<i>${esc(c.note)}</i>` : ""}</div>`).join("")}` : ""}</div></div>`;
    app.querySelectorAll("button[data-act]").forEach(b => b.onclick = () => reviewAction(f, b.dataset.act, () => factDetail(args)));
  }

  // ------------------------------------------------------------------ ask
  async function ask(args, q) {
    app.innerHTML = `<h1>Ask the evidence</h1>
      <div class="panel"><div class="ask-box"><input id="q" placeholder="e.g. What was CIL's coal production in FY 2024-25?" value="${esc(q.q || "")}"><button class="primary" id="go">Ask</button></div>
        <div class="chips" style="margin-top:10px">${["What was India's coal production in FY 2024-25?", "Compare CIL coal production in FY 2023-24 and FY 2024-25", "SECL vs MCL production in March 2025", "Which documents disagree on India's dispatch figures?", "What are the total coal resources of Jharkhand as on 1.4.2024?", "Lignite production of NLC India in 2025-26 upto December", "What is the fatality rate in coal mines?", "What were the main topics of the safety chapter?"].map(s => `<span class="chip">${esc(s)}</span>`).join("")}</div>
        <small class="muted">Routing: numeric → validated facts (SQL) · comparison · conflict → conflict register · explanatory → hybrid retrieval (BM25 + embeddings) with fail-closed citations. If evidence is missing the system says so instead of guessing.</small></div>
      <div id="result" style="margin-top:14px"></div>`;
    const run = async () => {
      const query = $("#q").value.trim(); if (!query) return;
      history.replaceState(null, "", `#/ask?q=${encodeURIComponent(query)}`);
      $("#result").innerHTML = '<div class="loading">Searching evidence…</div>';
      try { const r = await api("/api/v1/query", { json: { query } }); if ($("#result")) renderAnswer(r); } catch (e) { if ($("#result")) $("#result").innerHTML = `<div class="panel" style="color:var(--bad)">${esc(e.message)}</div>`; }
    };
    $("#go").onclick = run; $("#q").onkeydown = e => { if (e.key === "Enter") run(); };
    app.querySelectorAll(".chip").forEach(c => c.onclick = () => { $("#q").value = c.textContent; run(); });
    if (q.q) run();
  }
  function renderAnswer(r) {
    const cits = r.citations || [];
    let ans = esc(r.answer || "");
    ans = ans.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\[E(\d+)\]/g, (m, n) => `<span class="cite" data-c="${n}">E${n}</span>`);
    const abst = r.missing_evidence;
    $("#result").innerHTML = `
      <div class="panel"><div class="row between"><div class="qa-meta">route <b>${esc(r.route)}</b> · confidence ${confBar(r.confidence)} · ${r.elapsed_ms} ms ${r.mode ? "· " + esc(r.mode) : ""} ${r.partial ? '<span class="pill warn">partial evidence</span>' : ""}</div>${abst ? '<span class="pill warn">not enough evidence</span>' : '<span class="pill ok">evidence-backed</span>'}</div>
        <div class="answer ${abst ? "abstain" : ""}" style="margin-top:8px">${ans}</div>
        ${(r.conflicts || []).length ? `<div style="margin-top:10px">${r.conflicts.map(c => `<div class="conflict-card panel tight ${c.severity}"><b>${esc(c.code)}</b> ${esc(c.subject)} — ${esc(c.predicate)} ${esc(periodLabel(c.period))} <span class="pill warn">${esc(c.severity)}</span><div class="values" style="margin:6px 0">${c.values.map(v => `<div class="value-box" onclick="location.hash='${viewerLink(v)}'"><div class="v">${esc(v.display || v.value)}</div><small>${esc(v.document_code)} p${v.page} · ${esc(v.qualifier)}<br>${esc(v.document_title || "")}</small></div>`).join("")}</div><small>${esc(c.explanation)}</small></div>`).join("")}</div>` : ""}
      </div>
      <div class="panel" style="margin-top:12px"><h3>Citations (${cits.length}) — click to open the evidence</h3><div class="stack">
        ${cits.map((c, i) => `<div class="citation" onclick="location.hash='${viewerLink(c)}'"><span class="lbl">${esc(c.label || "E" + (i + 1))}</span>${c.fact_id ? `<img src="${imgUrl(c.crop_url || `/api/v1/facts/${c.fact_id}/crop`)}" alt="">` : c.region_id ? `<img src="${imgUrl(`/api/v1/regions/${c.region_id}/crop`)}" alt="">` : ""}<div><div><b>${esc(c.document_code)}</b> · ${esc(c.document_title || "")} · page ${c.page}${c.value ? ` · <b>${esc(c.value)}</b> (${esc(c.qualifier || "")}, ${esc(periodLabel(c.period))})` : ""}</div><small>${esc(c.snippet || c.context || "")}</small>${c.confidence != null ? `<div>${confBar(c.confidence)} ${statusPill(c.validation_status)}</div>` : ""}</div></div>`).join("") || '<div class="muted">No citations.</div>'}
      </div></div>
      <details class="panel" style="margin-top:12px"><summary>Raw JSON response (for auditors / integration)</summary><pre class="json">${esc(JSON.stringify(r, null, 2))}</pre></details>`;
    $("#result").querySelectorAll(".cite").forEach(el => el.onclick = () => { const c = cits[parseInt(el.dataset.c, 10) - 1] || cits.find(x => x.label === "E" + el.dataset.c); if (c) location.hash = viewerLink(c); });
  }

  // ------------------------------------------------------------------ conflicts
  async function conflicts(args, q) {
    const status = q.status || "open";
    const list = await api(`/api/v1/conflicts?status=${status}`);
    app.innerHTML = `<div class="row between"><h1>Cross-document conflicts</h1><div class="row"><select id="cstatus"><option ${status === "open" ? "selected" : ""}>open</option><option ${status === "resolved" ? "selected" : ""}>resolved</option><option ${status === "ignored" ? "selected" : ""}>ignored</option><option ${status === "all" ? "selected" : ""}>all</option></select><a class="btn" href="#/reports?tpl=conflict_register">Generate conflict register (DOCX)</a></div></div>
      <small class="muted">Same subject + metric + period compared across documents (rule V-006). Tolerance = published rounding precision; totals of partial mine lists, percentages and projections are excluded. Provisional monthly statistics vs final annual-report figures are the typical source of genuine discrepancies.</small>
      <div class="stack" style="margin-top:12px">${list.map(c => `
        <div class="panel conflict-card ${c.severity}"><div class="row between"><div><b>${esc(c.code)}</b> <span class="pill ${c.severity === "critical" ? "bad" : "warn"}">${esc(c.severity)}</span> ${statusPill(c.status)} &nbsp; <b>${esc(c.subject)}</b> — ${esc(c.predicate)} · ${esc(periodLabel(c.period))}</div>
          ${can("review") && c.status === "open" ? `<div class="row"><button class="small" data-id="${c.conflict_id}" data-act="ignore">Ignore</button><button class="small primary" data-id="${c.conflict_id}" data-act="resolve">Resolve…</button></div>` : ""}</div>
          <div class="values" style="margin:8px 0">${c.values.map(v => `<div class="value-box" onclick="location.hash='${viewerLink(v)}'"><div class="v">${esc(v.display || v.value)}</div><small>${esc(v.document_code)} · page ${v.page} · ${esc(v.qualifier)}<br>${esc(v.document_title || "")}</small><img src="${imgUrl(`/api/v1/facts/${v.fact_id}/crop`)}" alt="" loading="lazy"></div>`).join("")}</div>
          <small>${esc(c.explanation)}${c.resolution ? ` · <b>resolution:</b> ${esc(c.resolution)}` : ""}</small></div>`).join("") || '<div class="empty">No conflicts with this status.</div>'}</div>`;
    $("#cstatus").onchange = () => location.hash = `#/conflicts?status=${$("#cstatus").value}`;
    app.querySelectorAll("button[data-act]").forEach(b => b.onclick = async () => {
      const c = list.find(x => x.conflict_id === b.dataset.id);
      let body = { action: b.dataset.act };
      if (b.dataset.act === "resolve") {
        const opts = c.values.map((v, i) => `${i + 1}) ${v.value} — ${v.document_code} p${v.page} (${v.qualifier})`).join("\n");
        const pick = prompt(`Which value is authoritative?\n${opts}\n\nEnter a number (or leave blank to just mark resolved):`);
        if (pick === null) return; const idx = parseInt(pick, 10) - 1;
        if (!isNaN(idx) && c.values[idx]) body.preferred_fact_id = c.values[idx].fact_id;
        body.resolution = prompt("Resolution note", body.preferred_fact_id ? `Final figure from ${c.values[idx].document_code} preferred (later/final publication)` : "") || "";
      }
      try { await api(`/api/v1/conflicts/${c.conflict_id}/resolve`, { json: body }); toast("Conflict updated"); conflicts(args, q); } catch (e) { toast(e.message, true); }
    });
  }

  // ------------------------------------------------------------------ review queue
  async function review() {
    const r = await api("/api/v1/review/queue?limit=60");
    app.innerHTML = `<div class="row between"><h1>Review queue</h1><span class="pill warn">${r.total} facts need a human decision</span></div>
      <small class="muted">Facts with confidence &lt; 0.90, arithmetic or range warnings, or cross-document conflicts. Accept / edit / reject — every decision is recorded in the audit log and re-triggers cross-document validation.</small>
      <div class="panel" style="margin-top:12px">${r.items.map(f => `
        <div class="review-item"><div><div><b>${esc(f.subject)}</b>${f.dimension ? ` → ${esc(f.dimension)}` : ""} · ${esc(f.predicate)} · ${esc(periodLabel(f.period))} (${esc(f.qualifier)})</div>
          <div style="font-size:20px;font-weight:700">${esc(f.display)} <small class="muted">orig “${esc(f.original_value)}”</small></div>
          <div>${confBar(f.overall_confidence)} ${statusPill(f.validation_status)} <span class="pill ${f.review_status === "mandatory" ? "bad" : "warn"}">${esc(f.review_status)}</span> <small>${esc(f.document_code)} p${f.page_number} · ${esc((f.context || "").slice(0, 90))}</small></div>
          <div class="row" style="margin-top:6px"><a class="btn small" href="${viewerLink(f)}">view evidence</a>${can("review") ? `<button class="small primary" data-id="${f.fact_id}" data-act="accept">Accept</button><button class="small" data-id="${f.fact_id}" data-act="edit">Edit</button><button class="small danger" data-id="${f.fact_id}" data-act="reject">Reject</button>` : ""}</div></div>
          <img src="${imgUrl(f.crop_url)}" alt="evidence" loading="lazy"></div>`).join("") || '<div class="empty">Queue is empty 🎉</div>'}</div>`;
    app.querySelectorAll("button[data-act]").forEach(b => b.onclick = () => reviewAction(r.items.find(x => x.fact_id === b.dataset.id), b.dataset.act, review));
  }

  // ------------------------------------------------------------------ reports
  async function reports(args, q) {
    const [tpls, list, ents] = await Promise.all([api("/api/v1/reports/templates"), api("/api/v1/reports"), api("/api/v1/entities?type=company")]);
    const subs = await api("/api/v1/entities?type=subsidiary");
    const allEnts = ents.concat(subs).filter(e => e.facts > 0);
    app.innerHTML = `<h1>Report generator</h1>
      <div class="grid cols-2"><div class="panel"><h3>Template</h3>
        <select id="tpl" style="width:100%">${tpls.map(t => `<option value="${t.template_id}" ${q.tpl === t.template_id ? "selected" : ""}>${esc(t.title)}</option>`).join("")}</select>
        <p class="muted" id="tplDesc"></p>
        <div class="stack"><label>Period <input id="rPeriod" placeholder="FY2024-25 · 2025-03 · FY2025-26:ytd:2025-12 · asof:2025-04-01" style="width:100%"></label>
          <label>Entities (optional) <select id="rEnts" multiple size="6" style="width:100%">${allEnts.map(e => `<option value="${e.entity_id}">${esc(e.name)} (${e.facts})</option>`).join("")}</select></label>
          <label>Question (Parliament brief) <input id="rQ" style="width:100%" placeholder="What was India's coal production in FY 2024-25 and how did it compare with the target?"></label>
          <label>Title (optional) <input id="rTitle" style="width:100%"></label>
          ${can("report") ? '<button class="primary" id="gen">Generate DOCX</button>' : '<span class="muted">Your role cannot generate reports.</span>'}</div>
        <div id="genStatus" style="margin-top:8px"></div></div>
      <div class="panel"><h3>Generated reports</h3><div class="scroll" id="repList">${repRows(list)}</div>
        <small class="muted">Reports are assembled only from validated facts; each figure carries a footnote citation (document code, page, table). Facts that could not be found are listed under “Missing evidence” instead of being invented.</small></div></div>`;
    const desc = () => { const t = tpls.find(x => x.template_id === $("#tpl").value); $("#tplDesc").textContent = t ? `${t.description} Params: ${t.params.join(", ") || "none"}` : ""; };
    $("#tpl").onchange = desc; desc();
    if ($("#gen")) $("#gen").onclick = async () => {
      const params = { period: $("#rPeriod").value || undefined, entity_ids: [...$("#rEnts").selectedOptions].map(o => o.value), question: $("#rQ").value || undefined, title: $("#rTitle").value || undefined };
      $("#genStatus").textContent = "Generating…";
      try { const r = await api("/api/v1/reports/generate", { json: { template_id: $("#tpl").value, params } }); $("#genStatus").innerHTML = `Done: <a href="${imgUrl(r.download_url)}">download ${esc(r.title)}.docx</a> · ${r.facts_used} facts used${(r.missing || []).length ? ` · ${r.missing.length} missing items listed in the report` : ""}`; $("#repList").innerHTML = repRows(await api("/api/v1/reports")); }
      catch (e) { $("#genStatus").textContent = "Failed: " + e.message; }
    };
    function repRows(list) { return list.length ? `<table class="tbl"><thead><tr><th>Title</th><th>Template</th><th class="num">Facts</th><th>Missing</th><th>Created</th><th></th></tr></thead><tbody>${list.map(r => `<tr><td>${esc(r.title)}</td><td>${esc(r.template_id)}</td><td class="num">${r.facts_used}</td><td>${(r.missing || []).length}</td><td><small>${esc(r.created_at.replace("T", " ").slice(0, 16))}</small></td><td><a href="${imgUrl(r.download_url)}">DOCX</a></td></tr>`).join("")}</tbody></table>` : '<div class="empty">No reports yet.</div>'; }
  }

  // ------------------------------------------------------------------ topics
  async function topics(args, q) {
    const params = new URLSearchParams(); if (q.year) params.set("year", q.year); if (q.doc_type) params.set("doc_type", q.doc_type);
    const t = await api("/api/v1/topics" + (params.toString() ? "?" + params : ""));
    const words = t.words || []; const max = Math.max(...words.map(w => w.weight), 1e-9); const min = Math.min(...words.map(w => w.weight), max);
    const size = w => 12 + 30 * Math.sqrt((w - min) / (max - min + 1e-9));
    const hues = ["#0f766e", "#1d4ed8", "#b45309", "#be185d", "#4d7c0f", "#6d28d9", "#0e7490", "#9f1239"];
    app.innerHTML = `<div class="row between"><h1>Topics & word cloud</h1><div class="row"><select id="tYear"><option value="">all years</option>${["2022", "2023", "2024", "2025", "2026"].map(y => `<option ${q.year === y ? "selected" : ""}>${y}</option>`).join("")}</select><select id="tType"><option value="">all document types</option>${["monthly_statistics", "annual_report", "coal_directory", "administrative"].map(y => `<option ${q.doc_type === y ? "selected" : ""}>${y}</option>`).join("")}</select></div></div>
      <small class="muted">TF-IDF over ${t.n_pages} processed pages of ${t.n_docs} documents (domain stop-words removed) · NMF topic model · trend = topic share by fiscal year. ${t.summary ? "" : ""}</small>
      <div class="panel" style="margin-top:12px"><div class="cloud">${words.map((w, i) => `<span style="font-size:${size(w.weight).toFixed(0)}px;color:${hues[i % hues.length]};opacity:${(0.55 + 0.45 * (w.weight - min) / (max - min + 1e-9)).toFixed(2)}" title="${w.weight}">${esc(w.text)}</span>`).join("") || '<div class="empty">Not enough text yet.</div>'}</div></div>
      <div class="grid cols-4" style="margin-top:12px">${(t.topics || []).map((tp, i) => `<div class="topic"><b style="color:${hues[i % hues.length]}">${esc(tp.label)}</b> <span class="pill">${esc(tp.trend || "")}</span><div style="margin:6px 0"><small>${tp.keywords.map(k => esc(k)).join(" · ")}</small></div>
        <div class="trend" title="share by period">${(t.trend || []).map(tr => `<i style="height:${Math.max(3, Math.round(34 * (tr.weights[tp.topic_id] || 0) / Math.max(...(t.trend || []).map(x => x.weights[tp.topic_id] || 0), 1e-9)))}px" title="${esc(tr.period)}: ${((tr.weights[tp.topic_id] || 0) * 100).toFixed(0)}%"></i>`).join("")}</div>
        <small>e.g. ${tp.examples.map(e => `<a href="#/viewer/${e.document_id || ""}/${e.page}" onclick="event.preventDefault();window.__openTopicExample('${esc(e.doc_code)}',${e.page})">${esc(e.doc_code)} p${e.page}</a>`).join(", ")}</small></div>`).join("")}</div>
      <div class="panel" style="margin-top:12px"><h3>Summary</h3><div style="white-space:pre-wrap">${esc(t.summary || "")}</div></div>`;
    const nav = () => { const p = new URLSearchParams(); if ($("#tYear").value) p.set("year", $("#tYear").value); if ($("#tType").value) p.set("doc_type", $("#tType").value); location.hash = "#/topics" + (p.toString() ? "?" + p : ""); };
    $("#tYear").onchange = nav; $("#tType").onchange = nav;
    window.__openTopicExample = async (code, page) => { const docs = await api("/api/v1/documents"); const d = docs.find(x => x.code === code); if (d) location.hash = `#/viewer/${d.document_id}/${page}`; };
  }

  // ------------------------------------------------------------------ audit
  async function audit() {
    if (!can("audit")) { app.innerHTML = '<div class="panel"><h2>Audit log</h2><p class="muted">Your role cannot read the audit log. Switch to <b>auditor</b> or <b>admin</b>.</p></div>'; return; }
    const rows = await api("/api/v1/audit?limit=300");
    app.innerHTML = `<h1>Audit log</h1><small class="muted">Who accessed / queried / changed what, when — including the documents each answer relied on.</small>
      <div class="panel scroll" style="margin-top:12px"><table class="tbl"><thead><tr><th>Time (UTC)</th><th>User</th><th>Action</th><th>Object</th><th>Query / detail</th></tr></thead><tbody>
      ${rows.map(a => `<tr><td><small>${esc(a.timestamp.replace("T", " ").slice(0, 19))}</small></td><td>${esc(a.user)}</td><td><span class="tag">${esc(a.action)}</span></td><td class="mono"><small>${esc(a.entity_type || "")} ${esc((a.entity_id || a.document || "").slice(0, 12))}</small></td><td><small>${esc(a.query || "")} ${a.detail ? esc(JSON.stringify(a.detail).slice(0, 160)) : ""}</small></td></tr>`).join("")}
      </tbody></table></div>`;
  }

  // ------------------------------------------------------------------ boot
  loadMe().then(route).catch(e => { app.innerHTML = `<div class="panel">API not reachable: ${esc(e.message)}</div>`; });
})();
