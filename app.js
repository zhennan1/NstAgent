"use strict";

const LINKS = {
  paper: "https://arxiv.org/abs/2609.35759",
  code: "https://github.com/zhennan1/NstAgent",
  data: "https://huggingface.co/datasets/zhennan1/NstAgent",
};
const BIBTEX = `@article{wan2026nstagent,
  title={Scaling Long-Form Story Generation via Narrative State Tracking},
  author={Zhennan Wan and Jianfei Chen},
  year={2026}
}`;
const LENGTHS = ["10k", "20k", "50k", "100k"];
// Backend for live generation: same origin by default; config.js may point it elsewhere (e.g. from GitHub Pages).
const API = String(window.NST_API || "").replace(/\/+$/, "");
const api = path => (API ? `${API}/${path}` : path);

// ----------------------------------------------------------------------------- helpers
let LANG = (() => {
  try { const s = localStorage.getItem("lang"); if (s === "zh" || s === "en") return s; } catch { }
  return (navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en";
})();
function t(key, vars) {
  let s = (I18N[LANG] && I18N[LANG][key]) ?? I18N.en[key] ?? key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, v);
  return s;
}
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const isCJK = s => /[一-鿿]/.test(s || "");
// Chapters are stored 0-indexed (as in the paper and the data); the page shows them from 1.
const chNo = id => (id == null || id === "" || isNaN(+id) ? id : +id + 1);
const locFmt = s => String(s ?? "").replace(/(chapters?\s*(?:id\s*)?)(\d+)(\s*[-–]\s*)?(\d+)?/gi, (_, w, n, dash, m) => w + (+n + 1) + (dash && m ? dash + (+m + 1) : (dash || "") + (m || ""))).replace(/第\s*(\d+)\s*章/g, (_, n) => `第 ${+n + 1} 章`);
const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const cache = {};
async function getJSON(url) {
  if (cache[url]) return cache[url];
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status}`);
  return (cache[url] = await r.json());
}
function paragraphs(text) {
  return String(text || "").split(/\n\s*\n|\n/).map(p => p.trim()).filter(Boolean);
}
function applyI18n() {
  document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
  $$("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  $("#lang").textContent = t("lang");
}

// ----------------------------------------------------------------------------- state rendering
function stateHTML(state, prev, resolved) {
  if (!state || !Object.values(state).some(v => v && v.length)) return `<p class="muted small">${t("state.empty")}</p>`;
  const prevMap = b => new Map(((prev && prev[b]) || []).map(e => [e.id, e]));
  const block = (bucket, title, render) => {
    const items = state[bucket] || [], pm = prevMap(bucket);
    const rows = items.map(e => {
      const old = pm.get(e.id);
      const cls = !prev ? "" : !old ? "new" : JSON.stringify(old) !== JSON.stringify(e) ? "changed" : "";
      const tag = cls ? `<span class="tag ${cls}">${t("state." + cls)}</span>` : "";
      return { cls, html: `<div class="entry ${cls}">${render(e)}${tag}</div>` };
    });
    // Newly added or changed entries first, so each chapter's update is visible at a glance.
    rows.sort((a, b) => (b.cls ? 1 : 0) - (a.cls ? 1 : 0));
    return `<h4>${title}<span class="n">${items.length}</span></h4>${rows.map(r => r.html).join("")}`;
  };
  let html = block("character_states", t("state.characters"), e => `<b>${esc(e.name)}</b>: ${esc(e.description)}`);
  html += block("future_requirements", t("state.requirements"), e => `<span class="k">${esc(e.key)}</span> ${esc(e.description)}`);
  if (resolved && resolved.length) {
    html += `<h4>${t("state.resolved")}<span class="n">${resolved.length}</span></h4>` +
      resolved.map(e => `<div class="entry resolved"><span class="k">${esc(e.key)}</span> ${esc(e.description)}</div>`).join("");
  }
  const events = [...(state.past_events || [])].reverse();
  const pm = prevMap("past_events");
  html += `<h4>${t("state.events")}<span class="n">${events.length}</span></h4>` + events.map(e => {
    const cls = prev && !pm.has(e.id) ? "new" : "";
    return `<div class="entry ${cls}"><span class="k">${esc(e.key || "")}</span> ${esc(e.description)}${cls ? `<span class="tag new">${t("state.new")}</span>` : ""}</div>`;
  }).join("");
  return `<div class="state">${html}</div>`;
}

function toolHTML(ev) {
  const name = t("tool." + ev.name);
  let detail = "";
  if (ev.name === "read") detail = `#${chNo(ev.target)}`;
  else if (ev.name === "search") detail = `“${esc(ev.target)}”`;
  else if (ev.name === "write") detail = ev.words ? `${ev.words} ${t(isCJK(ev.info) ? "demo.chars" : "demo.words")}` : "";
  else if (ev.name === "correct") detail = ev.target != null ? `#${chNo(ev.target)}` : "";
  else if (ev.name === "update" && ev.ops) {
    const o = ev.ops;
    detail = t("tool.ops", { c: o.upsert_character_state, e: o.add_past_event, a: o.add_future_requirement, r: o.resolve_future_requirement });
  }
  const bad = !ev.ok ? ` <span class="muted">(${t("tool.rejected")})</span>` : "";
  return `<div class="t ${ev.ok ? ev.name : "bad"}" title="${esc(ev.info)}"><span class="nm">${esc(name)}</span><span>${detail}${bad}</span></div>`;
}

// ----------------------------------------------------------------------------- home
async function viewHome(root) {
  const btn = (key, href, cls = "ghost") => href ? `<a class="btn ${cls}" href="${href}" target="_blank" rel="noopener">${t(key)}</a>` : "";
  root.innerHTML = `
  <div class="narrow">
    <h1>${esc(t("title"))}</h1>
    <div class="buttons">
      ${btn("btn.paper", LINKS.paper)}${btn("btn.code", LINKS.code)}${btn("btn.data", LINKS.data)}
      <a class="btn blue" href="#demo">${t("btn.try")}</a>
    </div>
    <p class="tldr">${esc(t("tldr"))}</p>
    <h2>${t("h.overview")}</h2>
    <figure><img src="img/overview.png" alt="Overview of NstAgent"><figcaption>${esc(t("overview.caption"))}</figcaption></figure>
    <h2>${t("h.abstract")}</h2>
    <p class="abstract">${esc(t("abstract"))}</p>
    <h2>${t("h.results")}</h2>
    <div id="charts" class="charts"><p class="muted">${t("loading")}</p></div>
    <p class="note">${esc(t("results.note"))}</p>
    <h2>${t("h.cite")}</h2>
    <pre class="bib" id="bib">${esc(BIBTEX)}</pre>
  </div>`;
  try {
    const res = await getJSON("data/results.json");
    const charts = [];
    for (const bb of ["deepseek"]) {
      for (const [metric, label, d] of [["ins", "metric.ins", 1], ["wq", "metric.wq", 2]]) {
        const series = ["nstagent", "rollsum"].map(m => LENGTHS.map(l => (res.find(r => r.backbone === bb && r.method === m && r.length === l) || {})[metric]));
        charts.push(`<div class="chart"><h4>${t("b." + bb)} · ${t(label)}</h4>${lineChart(series, d)}
          <div class="legend"><span><i style="background:var(--blue)"></i>NstAgent</span><span><i style="background:var(--orange)"></i>RollSum</span></div></div>`);
      }
    }
    $("#charts").innerHTML = charts.join("");
  } catch { $("#charts").innerHTML = `<p class="muted">${t("error.load")}</p>`; }
}

function lineChart(series, digits) {
  const W = 300, H = 170, L = 40, R = 12, T = 12, B = 26;
  const vals = series.flat().filter(v => v != null);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.25 || 0.5; lo -= pad; hi += pad;
  const x = i => L + i * (W - L - R) / (LENGTHS.length - 1);
  const y = v => T + (hi - v) / (hi - lo) * (H - T - B);
  const colors = ["var(--blue)", "var(--orange)"];
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img">`;
  for (let k = 0; k <= 3; k++) {
    const v = lo + (hi - lo) * k / 3;
    svg += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#eceff2"/><text x="${L - 6}" y="${y(v) + 4}" font-size="10" text-anchor="end" fill="#8a949e">${v.toFixed(digits)}</text>`;
  }
  LENGTHS.forEach((l, i) => { svg += `<text x="${x(i)}" y="${H - 8}" font-size="11" text-anchor="middle" fill="#5c6773">${l.toUpperCase()}</text>`; });
  series.forEach((s, j) => {
    const pts = s.map((v, i) => v == null ? null : [x(i), y(v)]).filter(Boolean);
    svg += `<polyline fill="none" stroke="${colors[j]}" stroke-width="2.5" points="${pts.map(p => p.join(",")).join(" ")}"/>`;
    s.forEach((v, i) => { if (v != null) svg += `<circle cx="${x(i)}" cy="${y(v)}" r="3.5" fill="${colors[j]}"><title>${v.toFixed(3)}</title></circle>`; });
  });
  return svg + "</svg>";
}

// ----------------------------------------------------------------------------- demo
const Demo = { config: null, job: null, es: null, poll: null, data: null, sel: null, mode: null };

function newDemoData() {
  return { outline: [], lang: null, chapters: {}, tools: {}, states: {}, drafts: {}, thinking: {}, current: null, stage: null, stageVars: {}, done: null, error: null };
}

async function viewDemo(root) {
  root.innerHTML = `<p class="muted">${t("loading")}</p>`;
  try { Demo.config = await getJSON(api("api/config")); delete cache[api("api/config")]; }
  catch {
    root.innerHTML = `<div class="narrow"><h2 style="margin-top:0">${t("demo.title")}</h2><div class="status err">${esc(t("demo.offline"))}</div>
      <p class="muted">${t("demo.offline.local")} <a href="${LINKS.code}" target="_blank" rel="noopener">GitHub</a></p></div>`;
    return;
  }
  const saved = (() => { try { return localStorage.getItem("nst_job"); } catch { return null; } })();
  if (saved && !Demo.job) { Demo.job = saved; Demo.data = newDemoData(); connectJob(saved); }
  renderDemo(root);
}

// Rough time for any length, interpolated from the measured lengths.
function etaMinutes(words, opts) {
  const pts = (opts || []).map(o => [o.words, o.minutes]).sort((x, y) => x[0] - y[0]);
  if (!pts.length) return null;
  if (words <= pts[0][0]) return Math.max(5, Math.round(pts[0][1] * words / pts[0][0]));
  for (let i = 1; i < pts.length; i++) {
    const [w0, m0] = pts[i - 1], [w1, m1] = pts[i];
    if (words <= w1) return Math.round(m0 + (m1 - m0) * (words - w0) / (w1 - w0));
  }
  const [wl, ml] = pts[pts.length - 1];
  return Math.round(ml * words / wl);
}

function fmtMinutes(m) {
  if (m < 60) return t("time.min", { n: m });
  if (m > 60) return t("time.over");
  const h = m / 60;
  return t("time.hour", { n: Number.isInteger(h) ? h : h.toFixed(1) });
}

function renderDemo(root) {
  if (Demo.job) return renderJob(root);
  const c = Demo.config;
  if (!Demo.mode) Demo.mode = c.trial.available ? "trial" : "own";
  const own = Demo.mode === "own";
  const saved = (() => { try { return JSON.parse(sessionStorage.getItem("nst_own") || "{}"); } catch { return {}; } })();
  const opts = c.word_options || [];
  const minW = c.min_words || 1000, maxW = c.max_words || 100000;
  if (!Demo.words) Demo.words = (opts[0] || {}).words || 3000;
  const custom = Demo.customLen || !opts.some(o => o.words === Demo.words);
  const eta = w => { const m = etaMinutes(w, opts); return m == null ? "" : t("demo.eta", { t: fmtMinutes(m) }); };
  root.innerHTML = `
  <div class="narrow">
    <h2 style="margin-top:0">${t("demo.title")}</h2>
    <p class="muted">${esc(t("demo.intro"))}</p>
    <div class="card">
      <div class="field">
        <label for="prompt">${t("demo.prompt")}</label>
        <textarea id="prompt" maxlength="3000">${esc(Demo.lastPrompt || "")}</textarea>
        <div class="chips"><span class="muted small">${t("demo.examples")}</span>
          ${EXAMPLES[LANG].map((e, i) => `<button class="chip" data-ex="${i}" title="${esc(e)}">${esc(e.slice(0, LANG === "zh" ? 18 : 44))}…</button>`).join("")}
        </div>
        <div class="note">${esc(t("demo.lang.note"))}</div>
      </div>
      <div class="field"><label>${t("demo.length")}</label>
        <div class="lenrow">
          <div class="seg" id="len">${opts.map(o => `<button data-w="${o.words}" class="${!custom && o.words === Demo.words ? "on" : ""}">${o.words.toLocaleString()}</button>`).join("")}<button data-w="custom" class="${custom ? "on" : ""}">${t("demo.custom")}</button></div>
          <input type="number" id="customlen" class="customlen ${custom ? "" : "hidden"}" min="${minW}" max="${maxW}" step="500" value="${Demo.words}">
        </div>
        <div class="note" id="eta">${esc(eta(Demo.words))}</div>
      </div>
      <div class="field">
        <div class="seg" id="mode">
          <button data-m="trial" class="${own ? "" : "on"}">${t("demo.mode.trial")}</button>
          <button data-m="own" class="${own ? "on" : ""}">${t("demo.mode.own")}</button>
        </div>
      </div>
      ${own ? `
        <div class="field"><label>${t("demo.base")}</label><input type="text" id="base" placeholder="https://…/v1" value="${esc(saved.base || "")}"></div>
        <div class="row">
          <div class="field"><label>${t("demo.model")}</label><input type="text" id="model" value="${esc(saved.model || c.default_model || "")}"></div>
          <div class="field"><label>${t("demo.key")}</label><input type="password" id="key" autocomplete="off" placeholder="sk-…"></div>
        </div>
        <p class="note">${esc(t("demo.own.info"))}</p>`
      : `<p class="note">${esc(t("demo.trial.info"))}</p>`}
      <div id="formerr"></div>
      <button class="btn blue" id="start">${t("demo.start")}</button>
    </div>
  </div>`;
  $$(".chip", root).forEach(b => b.onclick = () => { $("#prompt").value = EXAMPLES[LANG][+b.dataset.ex]; });
  $$("#mode button", root).forEach(b => b.onclick = () => { Demo.lastPrompt = $("#prompt").value; Demo.mode = b.dataset.m; renderDemo(root); });
  const lenInput = $("#customlen");
  $$("#len button", root).forEach(b => b.onclick = () => {
    $$("#len button").forEach(x => x.classList.toggle("on", x === b));
    if (b.dataset.w === "custom") {
      Demo.customLen = true;
      lenInput.classList.remove("hidden");
      lenInput.focus();
      Demo.words = Math.min(maxW, Math.max(minW, +lenInput.value || Demo.words));
    } else {
      Demo.customLen = false;
      lenInput.classList.add("hidden");
      Demo.words = +b.dataset.w;
      lenInput.value = Demo.words;
    }
    $("#eta").textContent = eta(Demo.words);
  });
  lenInput.oninput = () => {
    const w = Math.round(+lenInput.value);
    if (w >= minW && w <= maxW) { Demo.words = w; $("#eta").textContent = eta(w); }
    else $("#eta").textContent = t("demo.len.range", { a: minW.toLocaleString(), b: maxW.toLocaleString() });
  };
  $("#start").onclick = startJob;
}

async function startJob() {
  const prompt = $("#prompt").value.trim();
  const err = m => { $("#formerr").innerHTML = `<div class="status err">${esc(m)}</div>`; };
  if (!prompt) return err(LANG === "zh" ? "请输入故事提示。" : "Please enter a story prompt.");
  const minW = Demo.config.min_words || 1000, maxW = Demo.config.max_words || 100000;
  if (Demo.customLen) {
    const w = Math.round(+$("#customlen").value);
    if (!(w >= minW && w <= maxW)) return err(t("demo.len.range", { a: minW.toLocaleString(), b: maxW.toLocaleString() }));
    Demo.words = w;
  }
  const body = { prompt, words: Demo.words };
  if (Demo.mode === "own") {
    body.api_base = $("#base").value.trim();
    body.model = $("#model").value.trim();
    body.api_key = $("#key").value.trim();
    if (!body.api_base) return err(LANG === "zh" ? "请输入 Base URL。" : "Please enter the base URL.");
    if (!body.api_key) return err(LANG === "zh" ? "请输入 API Key。" : "Please enter your API key.");
    try { sessionStorage.setItem("nst_own", JSON.stringify({ base: body.api_base, model: body.model })); } catch { }
  }
  $("#start").disabled = true;
  try {
    const r = await fetch(api("api/jobs"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error === "trial_unavailable" ? t("demo.trial.down") : (j.error || r.status));
    Demo.job = j.id; Demo.data = newDemoData(); Demo.data.trial = j.trial; Demo.sel = null; Demo.lastPrompt = prompt;
    try { localStorage.setItem("nst_job", j.id); } catch { }
    connectJob(j.id);
    renderDemo($("#view"));
  } catch (e) {
    $("#start").disabled = false;
    err(String(e.message || e));
  }
}

function connectJob(id) {
  if (Demo.es) Demo.es.close();
  if (Demo.poll) { clearTimeout(Demo.poll); Demo.poll = null; }
  const d = Demo.data;
  let seq = 0, gotAny = false;
  const take = ev => {
    if (ev.seq < seq) return;
    seq = ev.seq + 1;
    handleEvent(d, ev);
    scheduleJobRender();
  };
  const finished = () => d.done || d.error;
  // Polling fallback for proxies that buffer server-sent events (e.g. Cloudflare quick tunnels).
  const poll = async () => {
    if (Demo.job !== id) return;
    try {
      const r = await fetch(api(`api/jobs/${id}?since=${seq}`));
      if (r.status === 404) { d.error = LANG === "zh" ? "任务已过期或不存在。" : "This job has expired or does not exist."; scheduleJobRender(); return; }
      const j = await r.json();
      (j.events || []).forEach(take);
    } catch { }
    if (!finished()) Demo.poll = setTimeout(poll, 1000);
  };
  const startPolling = () => {
    if (Demo.es) { Demo.es.close(); Demo.es = null; }
    if (!Demo.poll && !finished()) poll();
  };
  if (!window.EventSource) return startPolling();
  const es = new EventSource(api(`api/jobs/${id}/events?since=0`));
  Demo.es = es;
  // Every job already has events when it is created, so silence means the stream is being buffered.
  const watchdog = setTimeout(() => { if (!gotAny) startPolling(); }, 3000);
  es.onmessage = m => { gotAny = true; take(JSON.parse(m.data)); };
  es.onerror = () => {
    es.close();
    clearTimeout(watchdog);
    if (!finished()) startPolling();
  };
}

function handleEvent(d, ev) {
  switch (ev.type) {
    case "stage": d.stage = ev.stage; d.stageVars = { n: ev.position }; break;
    case "outline": d.outline = ev.outline; d.lang = ev.lang; d.stage = "outlined"; break;
    case "chapter_start": d.current = ev.id; d.stage = "chapter"; d.tools[ev.id] = []; break;
    case "tool": (d.tools[ev.chapter] = d.tools[ev.chapter] || []).push(ev); if (ev.name === "write" && !ev.ok) d.drafts[ev.chapter + ":rejected"] = true; break;
    case "draft":
      if (ev.reset) { d.drafts[ev.chapter] = ""; delete d.drafts[ev.chapter + ":rejected"]; }
      d.drafts[ev.chapter] = (d.drafts[ev.chapter] || "") + (ev.text || "");
      d.thinking[ev.chapter] = false;
      break;
    case "thinking": d.thinking[ev.chapter] = true; break;
    case "chapter_done": d.chapters[ev.chapter.id] = ev.chapter; d.states[ev.chapter.id] = ev.state; break;
    case "done": d.done = ev; d.current = null; break;
    case "error": d.error = ev.trial ? t("demo.trial.down") : ev.message; d.current = null; break;
  }
}
const prevDone = d => { const ids = Object.keys(d.chapters).map(Number); return ids.length ? Math.max(...ids) : null; };

let renderTimer = null;
function scheduleJobRender() {
  if (renderTimer) return;
  renderTimer = setTimeout(() => { renderTimer = null; if (location.hash.startsWith("#demo") && Demo.job) renderJob($("#view"), true); }, 120);
}

function renderJob(root, incremental) {
  const d = Demo.data || newDemoData();
  const unit = t(d.lang === "zh" ? "demo.chars" : "demo.words");
  let status;
  if (d.error) status = `<div class="status err">${esc(t("stage.error", { m: d.error }))}</div>`;
  else if (d.done) {
    const w = Object.values(d.chapters).reduce((s, c) => s + (c.words || 0), 0);
    status = `<div class="status ${d.done.complete ? "ok" : "err"}">${esc(d.done.cancelled ? t("stage.cancelled") : d.done.complete ? t("stage.done", { w: w.toLocaleString() }) : t("stage.failed"))}</div>`;
  } else {
    const key = d.stage === "chapter" ? "stage.chapter" : d.stage === "queued" ? "stage.queued" : d.stage === "outline" || d.stage === "outlined" ? "stage.outline" : "stage.start";
    status = `<div class="status"><span class="spinner"></span>${esc(t(key, { n: d.stage === "chapter" ? chNo(d.current) : d.stageVars.n || 1 }))}</div>`;
  }
  const doneIds = Object.keys(d.chapters).map(Number);
  const sel = Demo.sel != null ? Demo.sel : (d.current != null && !d.chapters[d.current] ? d.current : prevDone(d));
  const ch = sel != null ? d.chapters[sel] : null;
  const outlineItem = d.outline.find(o => o.id === sel);
  const draft = sel != null && !ch ? d.drafts[sel] : null;
  const reader = ch
    ? `<h3>${esc(ch.name || "")}</h3>${paragraphs(ch.content).map(p => `<p>${esc(p)}</p>`).join("")}`
    : draft ? `<h3>${esc((outlineItem || {}).name || "")}</h3><div class="muted small" style="margin-bottom:8px">${d.drafts[sel + ":rejected"] ? t("draft.rejected") : `<span class="spinner"></span>${t("draft.writing")}`}</div>${paragraphs(draft).map(p => `<p>${esc(p)}</p>`).join("")}<span class="caret"></span>`
    : outlineItem ? `<h3>${esc(outlineItem.name || "")}</h3>${d.current === sel ? `<div class="muted small" style="margin-bottom:8px"><span class="spinner"></span>${esc(t(d.thinking[sel] ? "draft.thinking" : "draft.waiting"))}</div>` : ""}<div class="synopsis"><div class="label">${t("draft.synopsis")}</div>${esc(outlineItem.description || "")}</div>`
      : `<p class="placeholder">${d.stage && !d.error ? '<span class="spinner"></span>' : ""}${esc(Demo.lastPrompt || "")}</p>`;
  const prevId = doneIds.filter(i => i < sel).sort((a, b) => b - a)[0];
  // While a chapter is being written, show the state it was written from.
  const stateAfter = sel != null && d.states[sel] ? d.states[sel] : (prevId != null ? d.states[prevId] : null);
  const prevState = d.states[sel] ? (prevId != null ? d.states[prevId] : {}) : null;
  const resolved = stateAfter && prevState ? (prevState.future_requirements || []).filter(r => !(stateAfter.future_requirements || []).some(x => x.id === r.id)) : [];
  const tools = (d.tools[sel] || []).map(toolHTML).join("");
  const running = !d.done && !d.error;
  const html = `
    <div class="row" style="align-items:center">
      <div style="flex:3">${status}</div>
      <div style="flex:0;display:flex;gap:8px;white-space:nowrap">
        ${running ? `<button class="btn red" id="cancel">${t("demo.cancel")}</button>` : ""}
        ${d.done && doneIds.length ? `<a class="btn ghost" href="${api(`api/jobs/${Demo.job}/result`)}" download>${t("demo.download")}</a>` : ""}
        ${running ? "" : `<button class="btn blue" id="again">${t("demo.new")}</button>`}
      </div>
    </div>
    <div class="live">
      <div class="side card"><h4 style="margin:0 0 8px">${t("panel.outline")}</h4>
        <ul class="outline-list">${d.outline.map(o => `<li data-id="${o.id}" class="${d.chapters[o.id] ? "done" : ""} ${d.current === o.id ? "cur" : ""} ${sel === o.id ? "sel" : ""}">${chNo(o.id)}. ${esc(o.name || "")}<span class="w">${d.chapters[o.id] ? d.chapters[o.id].words : o.words || ""}</span></li>`).join("")}</ul>
      </div>
      <div class="card reader">${reader}</div>
      <div class="side">
        <div class="card"><h4 style="margin:0">${t("panel.state")}${sel != null ? ` · ${t("story.chapter", { n: chNo(sel) })}` : ""}</h4>${stateAfter ? stateHTML(stateAfter, prevState, resolved) : `<p class="muted small">${t("state.empty")}</p>`}</div>
        <div class="card" style="margin-top:12px"><h4 style="margin:0 0 6px">${t("panel.tools")}</h4><div class="tools">${tools || '<span class="muted small">–</span>'}</div></div>
      </div>
    </div>`;
  const scroll = incremental ? window.scrollY : 0;
  root.innerHTML = html;
  if (incremental) window.scrollTo(0, scroll);
  $$(".outline-list li", root).forEach(li => li.onclick = () => { Demo.sel = +li.dataset.id; renderJob(root); });
  const cancel = $("#cancel"); if (cancel) cancel.onclick = () => fetch(api(`api/jobs/${Demo.job}/cancel`), { method: "POST" });
  const again = $("#again"); if (again) again.onclick = () => {
    if (Demo.es) Demo.es.close();
    if (Demo.poll) { clearTimeout(Demo.poll); Demo.poll = null; }
    Demo.job = null; Demo.data = null; Demo.sel = null;
    try { localStorage.removeItem("nst_job"); } catch { }
    viewDemo(root);
  };
}

// ----------------------------------------------------------------------------- stories
const Browse = { filters: { lang: null, backbone: "deepseek", method: "nstagent", length: "10k", id: "" }, langTouched: false };

async function viewStories(root, file, chapter) {
  root.innerHTML = `<p class="muted">${t("loading")}</p>`;
  let index;
  try { index = await getJSON("data/index.json"); } catch { root.innerHTML = `<p>${t("error.load")}</p>`; return; }
  if (file) return viewStory(root, file, chapter);
  const f = Browse.filters;
  const opts = (key, values, label) => `<div><label class="small muted">${t(label)}</label><select data-f="${key}">${values.map(([v, l]) => `<option value="${v}" ${f[key] === v ? "selected" : ""}>${esc(l)}</option>`).join("")}</select></div>`;
  const ids = [...new Set(index.map(s => String(s.id)))];
  const langs = [...new Set(index.map(s => s.lang || "en"))];
  if (!Browse.langTouched) {
    f.lang = langs.includes(LANG) ? LANG : "en";
    if (f.lang !== "en") { f.method = "nstagent"; f.backbone = "deepseek"; }
  }
  const list = index.filter(s => (s.lang || "en") === f.lang && s.backbone === f.backbone && s.method === f.method && (!f.length || s.length === f.length) && (!f.id || String(s.id) === f.id));
  root.innerHTML = `
    <h2 style="margin-top:0">${t("stories.title")}</h2>
    <p class="muted">${esc(t("stories.intro"))}</p>
    <div class="filters">
      ${langs.length > 1 ? opts("lang", langs.map(l => [l, t("l." + l)]), "f.lang") : ""}
      ${opts("backbone", [["deepseek", t("b.deepseek")], ["luna", t("b.luna")]], "f.backbone")}
      ${opts("method", [["nstagent", "NstAgent"], ["rollsum", "RollSum"]], "f.method")}
      ${opts("length", [["", "—"], ...LENGTHS.map(l => [l, l.toUpperCase()])], "f.length")}
      ${opts("id", [["", "—"], ...ids.map(i => [i, "#" + i])], "f.prompt")}
    </div>
    <div class="list">${list.map(s => `
      <div class="card item" data-file="${s.file}">
        <div><div class="p">${esc(s.prompt)}</div>
          <div class="meta"><span class="badge ${s.method}">${t("m." + s.method)}</span><span class="badge">${s.length.toUpperCase()}</span>
          ${s.case ? `<span class="badge case">${t(s.case + ".title").split(/[:：]/)[0]}</span>` : ""}#${s.id} · ${s.chapters} ${LANG === "zh" ? "章" : "chapters"} · ${(s.word_count || 0).toLocaleString()} ${t(s.lang === "zh" ? "demo.chars" : "story.words")}</div></div>
        ${s.scores ? `<div class="scorebox"><div>${t("metric.ins")}<b>${fmt(s.scores.ins)}</b></div><div>${t("metric.wq")}<b>${fmt(s.scores.wq, 1)}</b></div></div>` : ""}
      </div>`).join("") || `<p class="muted">–</p>`}</div>`;
  $$("select[data-f]", root).forEach(sel => sel.onchange = () => {
    f[sel.dataset.f] = sel.value;
    if (sel.dataset.f === "lang") { Browse.langTouched = true; if (f.lang !== "en") { f.method = "nstagent"; f.backbone = "deepseek"; } }
    viewStories(root);
  });
  $$(".item", root).forEach(el => el.onclick = () => { location.hash = `#stories/${el.dataset.file}`; });
}

function chapterOf(loc) {
  const m = /chapters?\s*(?:id\s*)?(\d+)/i.exec(loc || "") || /第\s*(\d+)\s*章/.exec(loc || "");
  return m ? +m[1] : null;
}

function findSpan(text, quote) {
  if (!quote) return null;
  const norm = s => s.replace(/[“”]/g, '"').replace(/[‘’]/g, "'");
  const nt = norm(text), nq = norm(quote).trim();
  let i = nt.indexOf(nq);
  if (i >= 0) return [i, i + nq.length];
  for (const len of [80, 40]) {
    const head = nq.slice(0, len);
    if (head.length < 15) continue;
    i = nt.indexOf(head);
    if (i >= 0) return [i, i + Math.min(nq.length, len)];
  }
  return null;
}

function highlighted(text, marks) {
  const spans = [];
  for (const m of marks) {
    const s = findSpan(text, m.quote);
    if (s && !spans.some(x => s[0] < x[1] && x[0] < s[1])) spans.push([...s, m]);
  }
  spans.sort((a, b) => a[0] - b[0]);
  let out = "", pos = 0;
  for (const [a, b, m] of spans) {
    out += esc(text.slice(pos, a)) + `<mark class="${m.cls}" data-e="${m.idx}">` + esc(text.slice(a, b)) + "</mark>";
    pos = b;
  }
  out += esc(text.slice(pos));
  return out.split(/\n\s*\n|\n/).map(p => p.trim()).filter(Boolean).map(p => `<p>${p}</p>`).join("");
}

function stateAt(story, k) {
  const st = { character_states: new Map(), past_events: new Map(), future_requirements: new Map() };
  let prev = null;
  for (let i = 0; i <= k && i < story.chapters.length; i++) {
    if (i === k) prev = Object.fromEntries(Object.entries(st).map(([b, m]) => [b, [...m.values()]]));
    const diff = story.chapters[i].state_diff || {};
    for (const [b, d] of Object.entries(diff)) {
      for (const e of d.set || []) st[b].set(e.id, e);
      for (const id of d.remove || []) st[b].delete(id);
    }
  }
  const now = Object.fromEntries(Object.entries(st).map(([b, m]) => [b, [...m.values()].sort((x, y) => x.id - y.id)]));
  return { now, prev };
}

async function viewStory(root, file, chapter) {
  let s;
  try { s = await getJSON(`data/stories/${file}`); } catch { root.innerHTML = `<p>${t("error.load")}</p>`; return; }
  const k = Math.min(Math.max(chapter ?? 0, 0), s.chapters.length - 1);
  const ch = s.chapters[k];
  const errs = s.errors.map((e, idx) => ({ ...e, idx, at: chapterOf(e.location), pairAt: chapterOf(e.pair_location) }));
  const marks = [
    ...errs.filter(e => e.at === ch.id).map(e => ({ quote: e.quote, cls: "err", idx: e.idx })),
    ...errs.filter(e => e.pairAt === ch.id).map(e => ({ quote: e.pair, cls: "pair", idx: e.idx })),
  ];
  const inWindow = s.window && s.window.includes(ch.id);
  const unit = isCJK(ch.content) ? t("demo.chars") : t("story.words");
  // Left column: contents, scores, and this chapter's tool calls (one item per line).
  let left = `<div class="card"><ul class="outline-list">${s.chapters.map((c, i) => `<li data-i="${i}" class="done ${i === k ? "cur" : ""} ${s.window && s.window.includes(c.id) ? "check" : ""}" title="${esc(c.name)}">${chNo(c.id)}. ${esc(c.name || "")}<span class="w">${c.words}</span></li>`).join("")}</ul>
      ${s.window ? `<p class="note">● ${t("story.window")}</p>` : ""}</div>`;
  if (s.scores) left += `<div class="card kv"><div><span>${t("metric.ins")}</span><b>${fmt(s.scores.ins)}</b></div><div><span>${t("metric.sub")}</span><b>${fmt(s.scores.sub)}</b></div><div><span>${t("metric.wq")}</span><b>${fmt(s.scores.wq, 1)}</b></div></div>`;
  if (s.method === "nstagent" && ch.tools) {
    const tc = ch.tools;
    left += `<div class="card kv"><h4>${t("story.tools")}</h4>${["read", "search", "write", "correct", "update"].map(n => `<div><span>${t("tool." + n)}</span><b>${tc[n] || 0}</b></div>`).join("")}${tc.rejected ? `<div><span>${t("tool.rejected")}</span><b>${tc.rejected}</b></div>` : ""}</div>`;
  }
  // Right column: narrative state (or rolling summary), then the judge's contradictions.
  let right = "";
  if (s.method === "nstagent" && ch.state_diff) {
    const { now, prev } = stateAt(s, k);
    const resolved = (prev.future_requirements || []).filter(r => !now.future_requirements.some(x => x.id === r.id));
    right += `<div class="card"><h4 style="margin:0">${t("panel.state")} · ${t("story.chapter", { n: chNo(ch.id) })}</h4><div class="collapse state-box">${stateHTML(now, prev, resolved)}</div></div>`;
  } else if (s.method === "rollsum") {
    right += `<div class="card"><h4 style="margin:0 0 6px">${t("story.summary")}</h4>${ch.summary ? `<div class="small collapse state-box" style="white-space:pre-wrap">${esc(ch.summary)}</div>` : `<p class="muted small">${t("story.nosummary")}</p>`}</div>`;
  }
  right += `<div class="card"><h4 style="margin:0 0 8px">${t("story.errors")} (${errs.length})</h4><div class="errs collapse">${errs.length ? errs.map(e => `
      <div class="e" data-go="${e.at ?? ""}" data-pair="${e.pairAt ?? ""}" data-e="${e.idx}"><div class="h">${t("cat." + e.category)} · ${esc(e.subtype.replaceAll("_", " "))}</div>
      <q>${esc(e.quote)}</q><span class="muted small">${esc(locFmt(e.location))}</span>
      <div class="muted small" style="margin-top:4px">${t("story.pair")}: “${esc(e.pair)}” (${esc(locFmt(e.pair_location))})</div></div>`).join("")
    : `<p class="muted small">${t("story.noerrors")}</p>`}</div></div>`;
  root.innerHTML = `
    <p class="back"><a href="#stories">← ${t("story.back")}</a></p>
    <div class="card" style="margin-bottom:14px"><span class="badge ${s.method}">${t("m." + s.method)}</span><span class="badge">${t("b." + s.backbone)}</span><span class="badge">${s.length.toUpperCase()}</span>${s.lang === "zh" ? `<span class="badge case">${t("l.zh")}</span>` : ""} <span class="muted small">#${s.id}</span>
      <p style="margin:8px 0 0">${esc(s.prompt)}</p></div>
    <div class="live story">
      <div class="side stack">${left}</div>
      <div class="card reader">
        <div class="muted small">${t("story.chapter", { n: chNo(ch.id) })} ${t("story.of", { n: s.chapters.length })} · ${ch.words.toLocaleString()} ${unit}${inWindow ? ` · <span style="color:var(--orange)">● ${t("story.window")}</span>` : ""}</div>
        <h3>${esc(ch.name || "")}</h3>
        ${highlighted(ch.content, marks)}
        <div class="chnav">${k > 0 ? `<a href="#stories/${file}/${k - 1}">← ${esc(s.chapters[k - 1].name || "")}</a>` : "<span></span>"}${k < s.chapters.length - 1 ? `<a href="#stories/${file}/${k + 1}">${esc(s.chapters[k + 1].name || "")} →</a>` : ""}</div>
      </div>
      <div class="side stack">${right}</div>
    </div>`;
  $$(".outline-list li", root).forEach(li => li.onclick = () => { location.hash = `#stories/${file}/${li.dataset.i}`; });
  // Passages are only highlighted when their contradiction is selected, so the text reads normally.
  const show = idx => {
    $$("mark.on", root).forEach(m => m.classList.remove("on"));
    $$(".errs .e.sel", root).forEach(e => e.classList.remove("sel"));
    const ms = $$(`mark[data-e="${idx}"]`, root);
    ms.forEach(m => m.classList.add("on"));
    const item = $(`.errs .e[data-e="${idx}"]`, root);
    if (item) item.classList.add("sel");
    const target = ms.find(m => m.classList.contains("err")) || ms[0];
    if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
  };
  $$(".errs .e", root).forEach(el => el.onclick = () => {
    const idx = el.dataset.e;
    const here = $(`mark[data-e="${idx}"]`, root);
    if (!here) {
      for (const at of [el.dataset.go, el.dataset.pair]) {
        const i = s.chapters.findIndex(c => String(c.id) === at);
        if (i >= 0 && i !== k) { Browse.flash = idx; location.hash = `#stories/${file}/${i}`; return; }
      }
    }
    show(idx);
  });
  if (Browse.flash != null) {
    const idx = Browse.flash;
    Browse.flash = null;
    setTimeout(() => show(idx), 50);
  } else window.scrollTo(0, 0);
}

// ----------------------------------------------------------------------------- cases
async function viewCases(root) {
  root.innerHTML = `<p class="muted">${t("loading")}</p>`;
  let cases, index;
  try { [cases, index] = await Promise.all([getJSON("data/cases.json"), getJSON("data/index.json")]); }
  catch { root.innerHTML = `<p>${t("error.load")}</p>`; return; }
  const meta = f => index.find(s => s.file === f) || {};
  root.innerHTML = `<div class="narrow">
    <h2 style="margin-top:0">${t("cases.title")}</h2>
    <p class="muted">${esc(t("cases.intro"))}</p>
    <div class="cases">${cases.map(c => {
      const side = (m, cls) => {
        const s = meta(c[m]);
        return `<div class="side-card ${cls}"><b>${t("m." + m)}</b><div class="big">${s.scores ? s.scores.errors : "–"}</div>
          <div class="muted small">${t("cases.errors")}</div>${c[m] ? `<a href="#stories/${c[m]}">${t("cases.open")} →</a>` : ""}</div>`;
      };
      return `<div class="card case"><h3>${esc(t(c.key + ".title"))}</h3><p>${esc(t(c.key + ".text"))}</p>
        <div class="vs">${side("nstagent", "nst")}${side("rollsum", "roll")}</div></div>`;
    }).join("")}</div></div>`;
}

// ----------------------------------------------------------------------------- router
const Route = { last: null };
async function route() {
  const root = $("#view");
  const [page, a, b] = (location.hash.slice(1) || "home").split("/");
  root.classList.toggle("tight", page === "stories" && !!a);
  $$("#nav a").forEach(x => x.classList.toggle("active", x.getAttribute("href") === "#" + page));
  if (page !== "demo" && Demo.es && (!Demo.data || Demo.data.done || Demo.data.error)) { Demo.es.close(); Demo.es = null; }
  const key = location.hash.split("/").slice(0, 2).join("/");
  const fresh = Route.last !== key;
  Route.last = key;
  if (page === "demo") await viewDemo(root);
  else if (page === "stories") await viewStories(root, a, b != null ? +b : undefined);
  else if (page === "cases") await viewCases(root);
  else await viewHome(root);
  if (fresh) {
    // Entrance animations play once; later re-renders (e.g. live progress) must not replay them.
    root.classList.remove("enter"); void root.offsetWidth; root.classList.add("enter");
    clearTimeout(Route.enterTimer);
    Route.enterTimer = setTimeout(() => root.classList.remove("enter"), 1000);
    const h = $("h1", root) || $("h2", root);
    if (h && !h.closest(".live")) revealTitle(h);
  }
}

// Reveal a heading character by character (the method name moves as one piece).
function revealTitle(el) {
  if (!el || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const text = el.textContent;
  el.textContent = "";
  el.setAttribute("aria-label", text);
  let i = 0;
  for (const part of text.split(/(NstAgent)/)) {
    const chunks = part === "NstAgent" ? [part] : [...part];
    for (const ch of chunks) {
      const s = document.createElement("span");
      s.className = "rv" + (ch === "NstAgent" ? " nst" : "");
      s.textContent = ch;
      s.setAttribute("aria-hidden", "true");
      s.style.animationDelay = `${i * 38}ms`;
      el.appendChild(s);
      i++;
    }
  }
}

// Typeset every "NstAgent" in page text like the paper (small caps, Times).
function markMethodName(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: n => {
      const p = n.parentElement;
      if (!p || !n.nodeValue.includes("NstAgent")) return NodeFilter.FILTER_REJECT;
      if (p.closest(".nst, script, style, textarea, input, select, option, pre, .reader")) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const n of nodes) {
    const frag = document.createDocumentFragment();
    n.nodeValue.split(/(NstAgent)/).forEach(part => {
      if (!part) return;
      if (part === "NstAgent") {
        const s = document.createElement("span");
        s.className = "nst"; s.textContent = part;
        frag.appendChild(s);
      } else frag.appendChild(document.createTextNode(part));
    });
    n.parentNode.replaceChild(frag, n);
  }
}
new MutationObserver(muts => {
  for (const m of muts) m.addedNodes.forEach(n => {
    if (n.nodeType === 1) markMethodName(n);
    else if (n.nodeType === 3 && n.parentElement) markMethodName(n.parentElement);
  });
}).observe(document.body, { childList: true, subtree: true, characterData: false });
markMethodName(document.body);

$("#lang").onclick = () => {
  LANG = LANG === "zh" ? "en" : "zh";
  try { localStorage.setItem("lang", LANG); } catch { }
  applyI18n();
  Route.last = null;
  route();
};
window.addEventListener("hashchange", route);
applyI18n();
route();
