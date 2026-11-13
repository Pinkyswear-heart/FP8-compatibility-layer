/* benchlab -- bar charts of one metric across GPUs, inline SVG, no dependencies.

   Chart rules this file implements (from the dataviz method):
   - one metric per chart, so there is never a second y-axis;
   - values are shown as bars with a single baseline, 4px rounded data-end, <= 24px thick;
   - identity is carried by a swatch in the legend and by a value label at the tip, never by
     color alone;
   - hover gets a tooltip on the whole band, not just the bar;
   - a missing value renders as "未测", never as zero.
*/
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";
const SERIES = 8;
const COLORSLOTS = ["--series-1", "--series-2", "--series-3", "--series-4",
                    "--series-5", "--series-6", "--series-7", "--series-8"];

const state = {
  meta: null,
  records: [],
  selected: [],
  jobTimer: null,
};

/* ------------------------------------------------------------------ helpers */
function el(tag, attrs, kids) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined) continue;
    n.setAttribute(k, v);
  }
  for (const k of kids || []) n.appendChild(typeof k === "string" ? document.createTextNode(k) : k);
  return n;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmt(v, unit) {
  if (v === null || v === undefined) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 100) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function slugOf(rec) {
  return rec.gpu && rec.gpu.name ? rec.gpu.name : rec.id.split("/")[0];
}

/* Two records of the same card are two different measurements, and comparing 11-08 with 11-13 is
   exactly what this page is for -- so the display name disambiguates by time when the model
   repeats. The color slot follows the record for the same reason. */
function labelOf(rec) {
  const name = slugOf(rec);
  const same = state.records.filter((r) => slugOf(r) === name);
  if (same.length < 2) return name;
  const stamp = (rec.when || "").slice(5, 16).replace("T", " ").replace(/-/g, "/");
  return `${name.replace(/^NVIDIA GeForce /, "")} · ${stamp}`;
}

/* Each record keeps one color slot for the whole page. Assignment is by first appearance in the
   full record list, so selecting or deselecting a card never repaints the others. */
function colorSlot(rec) {
  return Math.max(0, state.records.findIndex((r) => r.id === rec.id)) % SERIES;
}

function colorOf(rec) {
  return cssVar(COLORSLOTS[colorSlot(rec)]);
}

function niceTicks(max, count) {
  if (!(max > 0)) return [0];
  const raw = max / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(v);
  return ticks;
}

/* Tooltip: one shared node, positioned next to the pointer and flipped near the edges. */
let tipNode = null;
function tip() {
  if (!tipNode) {
    tipNode = document.createElement("div");
    tipNode.className = "tip";
    document.body.appendChild(tipNode);
  }
  return tipNode;
}
function showTip(evt, html) {
  const t = tip();
  t.innerHTML = html;
  t.classList.add("on");
  const pad = 14;
  const r = t.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  t.style.left = Math.max(4, x) + "px";
  t.style.top = Math.max(4, y) + "px";
}
function hideTip() {
  if (tipNode) tipNode.classList.remove("on");
}

/* ------------------------------------------------------------------ api */
async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

/* ------------------------------------------------------------------ charts */
/* One chart per metric. Vertical bars, grouped by GPU: the tallest bar is the reference, so a
   card that is 2x slower has a bar half as long -- a linear scale, not a log one, because the
   question is "how much slower", not "how many orders of magnitude". */
function renderChart(metric, recs) {
  const host = document.createElement("div");
  host.className = "chart";

  const h3 = document.createElement("h3");
  h3.textContent = metric.label;
  const unit = document.createElement("span");
  unit.className = "unit";
  unit.textContent = metric.unit;
  h3.appendChild(unit);
  host.appendChild(h3);

  const note = document.createElement("p");
  note.className = "note";
  note.textContent = metric.hint || "";
  host.appendChild(note);

  const vals = recs.map((r) => ({ rec: r, v: r.metrics[metric.key] }));
  const present = vals.filter((x) => typeof x.v === "number" && isFinite(x.v) && x.v > 0);
  if (!present.length) {
    const e = document.createElement("div");
    e.className = "empty";
    e.textContent = "这些记录里没有这一项。";
    host.appendChild(e);
    return host;
  }

  const W = 470, rowH = 30, padL = 158, padR = 56, padT = 6, padB = 24;
  const H = padT + padB + rowH * vals.length;
  const plotW = W - padL - padR;
  const max = Math.max(...present.map((x) => x.v));
  const ticks = niceTicks(max, 4);
  const top = ticks[ticks.length - 1] || max;
  const x = (v) => padL + (v / top) * plotW;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
                          "aria-label": `${metric.label}（${metric.unit}）按 GPU 对比` });

  // gridlines + ticks (recessive, hairline, solid)
  for (const t of ticks) {
    svg.appendChild(el("line", { x1: x(t), x2: x(t), y1: padT, y2: H - padB,
                                 stroke: cssVar("--grid"), "stroke-width": 1 }));
    svg.appendChild(el("text", { x: x(t), y: H - padB + 14, "text-anchor": "middle",
                                 fill: cssVar("--text-muted"), "font-size": 10.5 }, [fmt(t, metric.unit)]));
  }

  vals.forEach((item, i) => {
    const y = padT + i * rowH;
    const barH = Math.min(18, rowH - 10);
    const cy = y + (rowH - barH) / 2;
    const name = labelOf(item.rec);

    const band = el("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" });
    svg.appendChild(band);

    // Name lives in its own gutter left of the baseline, so a long bar never runs under it.
    // Truncation is measured, not cursed: 11px system sans averages ~5.6px per glyph here.
    const room = Math.max(8, Math.floor((padL - 12) / 5.6));
    svg.appendChild(el("text", {
      x: padL - 10, y: cy + barH / 2 + 3.5, "text-anchor": "end",
      fill: cssVar("--text-secondary"), "font-size": 11,
    }, [name.length > room ? name.slice(0, room - 1) + "…" : name]));

    const hasVal = typeof item.v === "number" && isFinite(item.v) && item.v > 0;
    if (!hasVal) {
      svg.appendChild(el("text", { x: padL + 4, y: cy + barH / 2 + 3.5,
                                   fill: cssVar("--text-muted"), "font-size": 11 }, ["未测"]));
      band.addEventListener("mousemove", (e) => showTip(e,
        `<div class="t-key">${escapeHtml(name)}</div><div>${metric.label}：未测</div>`));
      band.addEventListener("mouseleave", hideTip);
      return;
    }

    const x0 = padL, w = Math.max(2, x(item.v) - x0);
    const path = roundedRightBar(x0, cy, w, barH, 4);
    const bar = el("path", { d: path, fill: colorOf(item.rec) });
    svg.appendChild(bar);

    // The value sits just outside the tip, always inside the right padding.
    svg.appendChild(el("text", {
      x: Math.min(x(item.v) + 6, W - padR + 4), y: cy + barH / 2 + 3.5,
      fill: cssVar("--text-primary"), "font-size": 11, "font-weight": 560,
      "font-variant-numeric": "tabular-nums",
    }, [`${fmt(item.v, metric.unit)}`]));

    band.addEventListener("mousemove", (e) => showTip(e,
      `<div class="t-key"><span class="t-dot" style="background:${colorOf(item.rec)}"></span>` +
      `${escapeHtml(name)}</div>` +
      `<div>${metric.label}：<span class="t-val">${fmt(item.v, metric.unit)}</span> ${metric.unit}</div>` +
      `<div class="t-key">${item.rec.when ? item.rec.when.replace("T", " ") : ""}` +
      `${item.rec.note ? " · " + escapeHtml(item.rec.note) : ""}</div>`));
    band.addEventListener("mouseleave", hideTip);
  });

  // baseline
  svg.appendChild(el("line", { x1: padL, x2: padL, y1: padT, y2: H - padB,
                               stroke: cssVar("--axis"), "stroke-width": 1 }));
  host.appendChild(svg);
  return host;
}

/* Bar growing from a square left edge to a 4px rounded right end. */
function roundedRightBar(x, y, w, h, r) {
  r = Math.min(r, w, h / 2);
  return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r}` +
         ` V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ------------------------------------------------------------------ sections */
function renderLegend(recs) {
  const host = document.getElementById("legend");
  host.innerHTML = "";
  if (recs.length < 2) return;
  for (const r of recs) {
    const it = document.createElement("span");
    it.className = "item";
    it.innerHTML = `<span class="swatch" style="background:${colorOf(r)}"></span>` +
                   escapeHtml(labelOf(r));
    host.appendChild(it);
  }
}

function renderCharts() {
  const host = document.getElementById("charts");
  host.innerHTML = "";
  const recs = state.selected.map((id) => state.records.find((r) => r.id === id)).filter(Boolean);
  if (!recs.length) {
    host.innerHTML = '<div class="empty">选中左侧的任意一张卡开始对比。</div>';
    renderLegend([]);
    return;
  }
  renderLegend(recs);
  for (const g of state.meta.groups) {
    const ms = state.meta.metrics.filter((m) => m.group === g.key);
    if (!ms.length) continue;
    for (const m of ms) host.appendChild(renderChart(m, recs));
  }
}

function renderTable() {
  const host = document.getElementById("table-host");
  host.innerHTML = "";
  const recs = state.selected.map((id) => state.records.find((r) => r.id === id)).filter(Boolean);
  if (!recs.length) return;
  const base = recs[0];

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  hr.appendChild(Object.assign(document.createElement("th"), { textContent: "指标" }));
  for (const r of recs) {
    hr.appendChild(Object.assign(document.createElement("th"), { textContent: labelOf(r) }));
  }
  thead.appendChild(hr);
  table.appendChild(thead);

  const tb = document.createElement("tbody");
  for (const m of state.meta.metrics) {
    const tr = document.createElement("tr");
    const first = document.createElement("td");
    first.textContent = `${m.label} (${m.unit})`;
    tr.appendChild(first);
    for (const r of recs) {
      const td = document.createElement("td");
      const v = r.metrics[m.key];
      if (typeof v !== "number" || !isFinite(v)) {
        td.innerHTML = '<span class="num-none">未测</span>';
      } else {
        const b = base.metrics[m.key];
        let sub = "";
        if (r !== base && typeof b === "number" && b > 0) {
          const rel = v / b;
          // "1.42x" reads as "this record's value is 1.42 of the first record's" -- spelled out
          // because a bare ratio column is easy to read backwards when lower is better.
          sub = `<div class="sub">为 ${escapeHtml(labelOf(base))} 的 ${rel.toFixed(2)}×</div>`;
        }
        td.innerHTML = `${fmt(v, m.unit)}${sub}`;
      }
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }

  // GPU identity rows, kept separate from the metric rows
  for (const [label, get] of [
    ["显存", (r) => r.gpu.vram_MB ? (r.gpu.vram_MB / 1024).toFixed(1) + " GB" : "—"],
    ["SM 数", (r) => r.gpu.sm_count || "—"],
    ["驱动", (r) => r.gpu.driver || "—"],
    ["CUDA", (r) => r.gpu.cuda || "—"],
    ["测量时间", (r) => (r.when || "").replace("T", " ")],
    ["来源", (r) => r.source === "import" ? "导入" : "本机"],
  ]) {
    const tr = document.createElement("tr");
    const t0 = document.createElement("td");
    t0.innerHTML = `<span class="muted">${label}</span>`;
    tr.appendChild(t0);
    for (const r of recs) {
      const td = document.createElement("td");
      td.innerHTML = `<span class="muted">${escapeHtml(String(get(r)))}</span>`;
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  table.appendChild(tb);
  host.appendChild(table);
}

function renderChips() {
  const host = document.getElementById("chips");
  host.innerHTML = "";
  if (!state.records.length) {
    host.innerHTML = '<div class="empty muted">还没有记录。先跑一次，或导入别的卡上的 JSON。</div>';
    return;
  }
  for (const r of state.records) {
    const sel = state.selected.includes(r.id);
    const chip = document.createElement("div");
    chip.className = "chip" + (sel ? " sel" : "");
    chip.tabIndex = 0;
    chip.setAttribute("role", "checkbox");
    chip.setAttribute("aria-checked", sel ? "true" : "false");
    const p = r.metrics;
    const bits = [];
    if (typeof p.decode_gelem_s === "number") bits.push(`${p.decode_gelem_s.toFixed(0)} Gelem/s`);
    if (r.gpu.vram_MB) bits.push(`${(r.gpu.vram_MB / 1024).toFixed(0)} GB`);
    chip.innerHTML =
      `<span class="swatch" style="background:${colorOf(r)}"></span>` +
      `<span>${escapeHtml(labelOf(r))}</span>` +
      `<span class="meta">${escapeHtml(bits.join(" · "))}</span>` +
      `<span class="x" title="删除这条记录">×</span>`;
    const toggle = (e) => {
      if (e.target.classList.contains("x")) return;
      const i = state.selected.indexOf(r.id);
      if (i >= 0) state.selected.splice(i, 1);
      else state.selected.push(r.id);
      renderAll();
    };
    chip.addEventListener("click", toggle);
    chip.addEventListener("keydown", (e) => {
      if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(e); }
    });
    chip.querySelector(".x").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`删除 ${labelOf(r)} 的这条记录？`)) return;
      await api(`/api/records/${r.id}`, { method: "DELETE" });
      state.records = state.records.filter((x) => x.id !== r.id);
      state.selected = state.selected.filter((x) => x !== r.id);
      renderAll();
    });
    host.appendChild(chip);
  }
}

function renderAll() {
  renderChips();
  renderCharts();
  renderTable();
}

/* ------------------------------------------------------------------ job */
function setJobView(j) {
  const box = document.getElementById("job");
  const st = document.getElementById("job-status");
  const logEl = document.getElementById("job-log");
  box.hidden = j.status === "idle" && !j.log_tail;
  st.className = "status " + j.status;
  st.innerHTML = `<span class="dot"></span>` +
    ({ idle: "空闲", running: `运行中 ${j.elapsed_s || 0}s`,
       done: `完成 ${j.elapsed_s || 0}s`, failed: "失败" }[j.status] || j.status);
  logEl.textContent = j.log_tail || "";
  logEl.scrollTop = logEl.scrollHeight;
  if (j.error) {
    const e = document.getElementById("job-error");
    e.textContent = j.error;
    e.hidden = false;
  }
  document.getElementById("btn-cancel").hidden = j.status !== "running";
}

async function pollJob() {
  try {
    const j = await api("/api/job");
    setJobView(j);
    if (j.status === "running") {
      clearTimeout(state.jobTimer);
      state.jobTimer = setTimeout(pollJob, 1200);
    } else if (j.status === "done" && j.record_id) {
      await loadRecords();
      if (!state.selected.includes(j.record_id)) state.selected.push(j.record_id);
      renderAll();
    }
  } catch (e) {
    // the server went away; stop polling and say so
    document.getElementById("job-status").innerHTML =
      `<span class="dot"></span>与服务的连接中断`;
  }
}

async function startRun() {
  const only = Array.from(document.querySelectorAll("#sections input:checked"))
    .map((i) => i.value);
  if (!only.length) { alert("至少选一个基准段落。"); return; }
  const note = document.getElementById("note").value.trim();
  const btn = document.getElementById("btn-run");
  btn.disabled = true;
  document.getElementById("job-error").hidden = true;
  try {
    const j = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ only, note }),
    });
    setJobView(j);
    clearTimeout(state.jobTimer);
    state.jobTimer = setTimeout(pollJob, 800);
  } catch (e) {
    const el2 = document.getElementById("job-error");
    el2.textContent = String(e.message || e);
    el2.hidden = false;
  } finally {
    btn.disabled = false;
  }
}

async function cancelRun() {
  await api("/api/job/cancel", { method: "POST" });
  pollJob();
}

async function importJson() {
  const ta = document.getElementById("import-text");
  const err = document.getElementById("import-error");
  err.hidden = true;
  let raw;
  try {
    raw = JSON.parse(ta.value);
  } catch (e) {
    err.textContent = "不是合法的 JSON：" + e.message;
    err.hidden = false;
    return;
  }
  try {
    const rec = await api("/api/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw, note: document.getElementById("import-note").value.trim() }),
    });
    ta.value = "";
    document.getElementById("import-note").value = "";
    await loadRecords();
    if (!state.selected.includes(rec.id)) state.selected.push(rec.id);
    renderAll();
  } catch (e) {
    err.textContent = String(e.message || e);
    err.hidden = false;
  }
}

/* Dropping a file straight onto the panel is the normal way to move a record between machines. */
function wireDrop() {
  const ta = document.getElementById("import-text");
  const drop = document.getElementById("import-drop");
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", async (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    ta.value = await f.text();
  });
}

/* ------------------------------------------------------------------ boot */
async function loadRecords() {
  const j = await api("/api/records");
  state.records = j.records;
}

async function boot() {
  state.meta = await api("/api/meta");
  await loadRecords();

  // nothing selected yet: pick the newest two, so the page is useful on first open
  state.selected = state.records.slice(0, 2).map((r) => r.id);

  document.getElementById("btn-run").addEventListener("click", startRun);
  document.getElementById("btn-cancel").addEventListener("click", cancelRun);
  document.getElementById("btn-import").addEventListener("click", importJson);
  document.getElementById("root-path").textContent = state.meta.root;
  document.querySelectorAll("#sections input").forEach((i) => {
    const sync = () => i.closest("label").classList.toggle("on", i.checked);
    i.addEventListener("change", sync);
    sync();
  });
  document.getElementById("btn-all").addEventListener("click", () => {
    document.querySelectorAll("#sections input").forEach((i) => {
      i.checked = true;
      i.closest("label").classList.add("on");
    });
  });
  document.getElementById("btn-fast").addEventListener("click", () => {
    const fast = new Set(["bw", "decode", "encode", "floor"]);
    document.querySelectorAll("#sections input").forEach((i) => {
      i.checked = fast.has(i.value);
      i.closest("label").classList.toggle("on", i.checked);
    });
  });
  wireDrop();

  renderAll();
  pollJob();

  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", () => renderAll());
}

boot().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<div style="padding:16px;color:#d03b3b">加载失败：${escapeHtml(String(e.message || e))}</div>`);
});
