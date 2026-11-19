/* Trace player for the laser + conveyor simulator.
 *
 * This file contains NO simulation logic. It replays the frames produced by
 * the Python engine (window.SIM_TRACE or a loaded trace.json), interpolating
 * positions between frames. The visual speed only changes how fast the
 * replay clock advances, never the simulated results.
 */
(() => {
  "use strict";

  const SPEEDS = [1, 2, 5, 10, 50];
  const STATUS = {
    W: { name: "WAITING", token: "--st-waiting" },
    M: { name: "MOVING", token: "--st-moving" },
    R: { name: "READY", token: "--st-ready" },
    P: { name: "PROCESSING", token: "--st-processing" },
    C: { name: "COMPLETED", token: "--st-completed" },
  };
  const PART = 3.0;            // drawn part size [cm]
  const BELT_HALF = 3.0;       // half belt width [cm]
  const QUEUE_STEP = 3.6;      // spacing of WAITING parts before the entry [cm]

  const $ = (id) => document.getElementById(id);
  const cell = $("cell"), cctx = cell.getContext("2d");
  const qc = $("queueChart"), qctx = qc.getContext("2d");

  let traces = {};        // policy name -> trace
  let trace = null, frames = [], T = 0, simT = 0, playing = false, speed = 5, lastTs = null;
  let cumRun = [];       // cumulative running time per conveyor at each frame (belt animation)
  let colors = {};
  let view = null;       // world -> screen transform
  let lastLogIdx = -1;

  // ------------------------------------------------------------- theme
  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    colors = {
      bg: g("--panel"), ink: g("--ink"), muted: g("--muted"), line: g("--line"),
      accent: g("--accent"), belt: g("--belt"), stripe: g("--belt-stripe"),
      zEntry: g("--zone-entry"), zActive: g("--zone-active"), zExit: g("--zone-exit"),
      gantry: g("--gantry"), laser: g("--laser"), stop: g("--stop"),
      conv: ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6"].map(g),
      fontData: g("--font-data") || "monospace", fontUi: g("--font-ui") || "sans-serif",
    };
    for (const k in STATUS) colors[k] = g(STATUS[k].token);
  }
  const convColor = (i) => colors.conv[i % colors.conv.length];

  // ------------------------------------------------------------- loading
  function setTraces(map, selected) {
    traces = map;
    const sel = $("policySelect");
    sel.innerHTML = Object.keys(traces).map((k) =>
      `<option value="${k}">${k}</option>`).join("");
    sel.value = selected;
    sel.disabled = Object.keys(traces).length < 2;
    load(traces[selected]);
  }

  function load(tr, keepTime = false) {
    const prevT = simT, wasPlaying = playing;
    const msg = document.getElementById("emptyMsg");
    if (msg) msg.remove();
    cell.hidden = false;
    trace = tr;
    frames = tr.frames;
    T = frames[frames.length - 1].t;
    simT = keepTime ? Math.min(prevT, T) : 0;
    playing = keepTime ? wasPlaying : false;
    lastLogIdx = -1;
    const nC = tr.geometry.conveyors.length;
    cumRun = [new Array(nC).fill(0)];
    for (let i = 1; i < frames.length; i++) {
      const dt = frames[i].t - frames[i - 1].t;
      cumRun.push(cumRun[i - 1].map((v, c) => v + (frames[i - 1].C[c][0] ? dt : 0)));
    }
    const m = tr.meta, s = tr.summary || {};
    $("meta").textContent = `policy=${m.policy} · arrivals=${m.arrival_mode} · seed=${m.seed} · ` +
      `zone end=${m.unprocessed_at_zone_end} · stop rule=${m.conveyor_stop_policy} · ` +
      `motion=${m.motion_model} · dt=${m.dt}s · makespan=${fmt(s.makespan, 1)} s`;
    $("scrub").max = T.toFixed(1);
    $("tEnd").textContent = `${T.toFixed(1)} s`;
    buildQueues(nC);
    resize();
    updatePlayBtn();
    render();
  }

  function buildQueues(nC) {
    const box = $("queues");
    box.innerHTML = "";
    for (let c = 0; c < nC; c++) {
      const row = document.createElement("div");
      row.className = "qrow";
      row.innerHTML = `<span>C${c + 1}</span><span class="qbar"><b id="qb${c}"></b></span>` +
        `<span id="qn${c}">0</span><span class="qstate" id="qs${c}">RUN</span>`;
      box.appendChild(row);
      $(`qb${c}`).style.background = convColor(c);
    }
  }

  function buildLegend() {
    $("legend").innerHTML = Object.keys(STATUS).map((k) =>
      `<span><i style="background:${colors[k]}"></i>${STATUS[k].name}</span>`).join("") +
      `<span><i style="background:${colors.laser};border-radius:50%"></i>Laser head</span>` +
      `<span><i style="background:${colors.stop}"></i>Stopped conveyor</span>`;
  }

  // ------------------------------------------------------------- frames
  function frameIndex(t) {           // last frame with frame.t <= t (binary search)
    let lo = 0, hi = frames.length - 1;
    if (t >= frames[hi].t) return hi;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (frames[mid].t <= t) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  function stateAt(t) {
    const i = frameIndex(t);
    const f0 = frames[i], f1 = frames[Math.min(i + 1, frames.length - 1)];
    const a = f1.t > f0.t ? Math.min(1, Math.max(0, (t - f0.t) / (f1.t - f0.t))) : 0;
    const next = new Map(f1.P.map((p) => [p[0], p]));
    const parts = f0.P.map((p) => {
      const q = next.get(p[0]);
      const x = q && q[3] !== "W" && p[3] !== "W" ? p[2] + (q[2] - p[2]) * a : p[2];
      return { n: p[0], c: p[1], x, s: p[3] };
    });
    const lx = f0.L[0] + (f1.L[0] - f0.L[0]) * a, ly = f0.L[1] + (f1.L[1] - f0.L[1]) * a;
    const run = cumRun[i].map((v, c) => v + (f0.C[c][0] ? (t - f0.t) : 0));
    return { f: f0, parts, laser: { x: lx, y: ly, state: f0.L[2], target: f0.L[3] }, run };
  }

  // ------------------------------------------------------------- canvas
  function resize() {
    if (!trace) return;
    const g = trace.geometry, ys = g.conveyors.map((c) => c.y);
    const dpr = window.devicePixelRatio || 1;
    const w = cell.clientWidth || 800;
    const marginL = 64, marginR = 12, marginT = 34, marginB = 46;
    const xMin = -(QUEUE_STEP * 4 + 1), xMax = g.total_length + 2;
    const yMin = Math.min(...ys) - BELT_HALF - 4, yMax = Math.max(...ys) + BELT_HALF + 4;
    const s = Math.max(2, (w - marginL - marginR) / (xMax - xMin));
    const h = Math.round(marginT + marginB + (yMax - yMin) * s);
    cell.width = Math.round(w * dpr); cell.height = Math.round(h * dpr);
    cell.style.height = `${h}px`;
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    view = { s, X: (x) => marginL + (x - xMin) * s, Y: (y) => marginT + (y - yMin) * s,
             w, h, yMin, yMax, marginL, marginT };
    const qw = qc.clientWidth || 600;
    qc.width = Math.round(qw * dpr); qc.height = Math.round(180 * dpr);
    qctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function render() {
    if (!trace) return;
    const st = stateAt(simT);
    drawCell(st);
    drawQueueChart();
    updateDashboard(st);
    updateLog();
  }

  function drawCell(st) {
    const g = trace.geometry, ws = g.workspace, v = view, s = v.s, ctx = cctx;
    ctx.clearRect(0, 0, v.w, v.h);
    ctx.fillStyle = colors.bg; ctx.fillRect(0, 0, v.w, v.h);
    const top = v.Y(v.yMin), bot = v.Y(v.yMax);
    const x0 = 0, x1 = g.entry_length, x2 = x1 + g.active_length, x3 = g.total_length;

    // zones
    ctx.fillStyle = colors.zEntry; ctx.fillRect(v.X(x0), top, (x1 - x0) * s, bot - top);
    ctx.fillStyle = colors.zActive; ctx.fillRect(v.X(x1), top, (x2 - x1) * s, bot - top);
    ctx.fillStyle = colors.zExit; ctx.fillRect(v.X(x2), top, (x3 - x2) * s, bot - top);
    ctx.font = `600 11px ${colors.fontUi}`; ctx.fillStyle = colors.muted; ctx.textAlign = "center";
    ctx.fillText("ENTRY", v.X((x0 + x1) / 2), top - 8);
    ctx.fillText("LASER ZONE", v.X((x1 + x2) / 2), top - 8);
    ctx.fillText("EXIT", v.X((x2 + x3) / 2), top - 8);
    ctx.fillText("QUEUE", v.X(-QUEUE_STEP * 2.2), top - 8);

    // laser workspace outline
    ctx.save(); ctx.setLineDash([6, 4]); ctx.strokeStyle = colors.accent; ctx.lineWidth = 1.2;
    ctx.strokeRect(v.X(ws.x_min), v.Y(ws.y_min - BELT_HALF - 1.5), (ws.x_max - ws.x_min) * s,
                   (ws.y_max - ws.y_min + 2 * BELT_HALF + 3) * s);
    ctx.restore();

    // belts
    g.conveyors.forEach((c, i) => {
      const running = st.f.C[i][0] === 1, reason = st.f.C[i][1];
      const yTop = v.Y(c.y - BELT_HALF), hgt = 2 * BELT_HALF * s;
      ctx.fillStyle = colors.belt; ctx.fillRect(v.X(0), yTop, x3 * s, hgt);
      // moving stripes: offset = distance travelled by the belt (scrub-safe)
      const off = (st.run[i] * c.speed) % 2.5;
      ctx.strokeStyle = colors.stripe; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = off; x < x3; x += 2.5) { ctx.moveTo(v.X(x), yTop + 2); ctx.lineTo(v.X(x), yTop + hgt - 2); }
      ctx.stroke();
      ctx.lineWidth = running ? 1 : 2.5;
      ctx.strokeStyle = running ? colors.line : colors.stop;
      ctx.strokeRect(v.X(0), yTop, x3 * s, hgt);
      // direction arrow at the exit
      ctx.fillStyle = colors.muted; ctx.beginPath();
      ctx.moveTo(v.X(x3 + 0.4), v.Y(c.y) - 5); ctx.lineTo(v.X(x3 + 1.8), v.Y(c.y)); ctx.lineTo(v.X(x3 + 0.4), v.Y(c.y) + 5);
      ctx.fill();
      // label
      ctx.textAlign = "left"; ctx.font = `600 12px ${colors.fontData}`; ctx.fillStyle = convColor(i);
      ctx.fillText(`C${c.id}`, 8, v.Y(c.y) - 2);
      ctx.font = `600 9px ${colors.fontData}`;
      ctx.fillStyle = running ? colors.muted : colors.stop;
      ctx.fillText(running ? "RUN" : (reason.includes("processing") ? "STOP·LASER" : "STOP·BLOCK"), 8, v.Y(c.y) + 11);
    });

    // parts
    const waitingIdx = {};
    const procId = st.laser.state === "PROCESSING" ? st.laser.target : null;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const p of st.parts) {
      const conv = g.conveyors[p.c - 1];
      let x = p.x;
      if (p.s === "W") { waitingIdx[p.c] = (waitingIdx[p.c] || 0) + 1; x = -QUEUE_STEP * waitingIdx[p.c] + PART / 2 - 0.6; }
      if (x < -QUEUE_STEP * 4) continue;   // long upstream queues: draw the first 4 only
      const px = v.X(x - PART / 2), py = v.Y(conv.y - PART / 2), sz = PART * s;
      ctx.fillStyle = colors[p.s] || colors.muted;
      ctx.fillRect(px, py, sz, sz);
      if (p.n === procId || p.n === st.laser.target) {
        ctx.strokeStyle = colors.ink; ctx.lineWidth = 2; ctx.strokeRect(px - 1, py - 1, sz + 2, sz + 2);
      }
      if (s >= 9) {
        ctx.fillStyle = "#fff"; ctx.font = `600 ${Math.min(10, s * 0.9)}px ${colors.fontData}`;
        ctx.fillText(String(p.n), px + sz / 2, py + sz / 2 + 0.5);
      }
    }
    // long queues: counter
    for (const c in waitingIdx) if (waitingIdx[c] > 4) {
      ctx.fillStyle = colors.stop; ctx.font = `600 10px ${colors.fontData}`;
      ctx.fillText(`+${waitingIdx[c] - 4}`, v.X(-QUEUE_STEP * 4 - 0.5), v.Y(g.conveyors[c - 1].y) - BELT_HALF * s - 6);
    }
    ctx.textBaseline = "alphabetic";

    // gantry: X rails + Y bridge + head
    const L = st.laser, railTop = v.Y(ws.y_min - BELT_HALF - 1.5), railBot = v.Y(ws.y_max + BELT_HALF + 1.5);
    ctx.strokeStyle = colors.gantry; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(v.X(L.x), railTop); ctx.lineTo(v.X(L.x), railBot); ctx.stroke();
    ctx.fillStyle = colors.gantry;
    ctx.fillRect(v.X(L.x) - 7, railTop - 4, 14, 6); ctx.fillRect(v.X(L.x) - 7, railBot - 2, 14, 6);
    if (L.state === "MOVING" || L.state === "WAITING_FOR_PART") {
      const tp = st.parts.find((p) => p.n === L.target);
      if (tp) {
        const ty = g.conveyors[tp.c - 1].y;
        ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = colors.laser; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(v.X(L.x), v.Y(L.y)); ctx.lineTo(v.X(tp.x), v.Y(ty)); ctx.stroke(); ctx.restore();
      }
    }
    const hx = v.X(L.x), hy = v.Y(L.y);
    if (L.state === "PROCESSING") {
      const pulse = 0.6 + 0.4 * Math.sin(simT * 12);
      const r = (2.4 + pulse) * s;
      const grad = ctx.createRadialGradient(hx, hy, 0, hx, hy, r);
      grad.addColorStop(0, colors.laser); grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.globalAlpha = 0.55; ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(hx, hy, r, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = colors.laser; ctx.strokeStyle = colors.bg; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(hx, hy, 0.9 * s, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = colors.ink; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(hx - 1.6 * s, hy); ctx.lineTo(hx + 1.6 * s, hy);
    ctx.moveTo(hx, hy - 1.6 * s); ctx.lineTo(hx, hy + 1.6 * s); ctx.stroke();

    // dimension line
    const dy = bot + 18;
    ctx.strokeStyle = colors.muted; ctx.fillStyle = colors.muted; ctx.lineWidth = 1;
    ctx.font = `11px ${colors.fontData}`; ctx.textAlign = "center";
    [[x0, x1, `${g.entry_length} cm entry`], [x1, x2, `${g.active_length} cm laser zone`],
     [x2, x3, `${g.exit_length} cm exit`]].forEach(([a, b, label]) => {
      ctx.beginPath(); ctx.moveTo(v.X(a) + 2, dy); ctx.lineTo(v.X(b) - 2, dy);
      ctx.moveTo(v.X(a), dy - 5); ctx.lineTo(v.X(a), dy + 5); ctx.moveTo(v.X(b), dy - 5); ctx.lineTo(v.X(b), dy + 5);
      ctx.stroke(); ctx.fillText(label, v.X((a + b) / 2), dy + 16);
    });
  }

  function drawQueueChart() {
    const ctx = qctx, w = qc.clientWidth || 600, h = 180, nC = trace.geometry.conveyors.length;
    const padL = 30, padR = 8, padT = 10, padB = 22;
    ctx.clearRect(0, 0, w, h);
    let qMax = 1;
    for (const f of frames) for (const q of f.K.queues) if (q > qMax) qMax = q;
    const X = (t) => padL + (t / Math.max(T, 1e-9)) * (w - padL - padR);
    const Y = (q) => padT + (1 - q / qMax) * (h - padT - padB);
    ctx.strokeStyle = colors.line; ctx.fillStyle = colors.muted; ctx.lineWidth = 1;
    ctx.font = `10px ${colors.fontData}`; ctx.textAlign = "right";
    const step = Math.max(1, Math.ceil(qMax / 4));
    for (let q = 0; q <= qMax; q += step) {
      ctx.beginPath(); ctx.moveTo(padL, Y(q)); ctx.lineTo(w - padR, Y(q)); ctx.stroke();
      ctx.fillText(String(q), padL - 4, Y(q) + 3);
    }
    ctx.textAlign = "center";
    const tStep = niceStep(T / 6);
    for (let t = 0; t <= T + 1e-9; t += tStep) ctx.fillText(`${Math.round(t)}s`, X(t), h - 6);
    const stride = Math.max(1, Math.floor(frames.length / (w * 2)));
    for (let c = 0; c < nC; c++) {
      ctx.strokeStyle = convColor(c); ctx.lineWidth = 1.5; ctx.beginPath();
      for (let i = 0; i < frames.length; i += stride) {
        const f = frames[i];
        if (i === 0) ctx.moveTo(X(f.t), Y(f.K.queues[c])); else ctx.lineTo(X(f.t), Y(f.K.queues[c]));
      }
      ctx.stroke();
      ctx.fillStyle = convColor(c); ctx.textAlign = "left";
      ctx.fillText(`C${c + 1}`, padL + 6 + c * 34, padT + 10);
    }
    ctx.strokeStyle = colors.ink; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(X(simT), padT); ctx.lineTo(X(simT), h - padB); ctx.stroke(); ctx.setLineDash([]);
  }

  function niceStep(x) {
    const p = Math.pow(10, Math.floor(Math.log10(Math.max(x, 1e-9))));
    for (const m of [1, 2, 5, 10]) if (m * p >= x) return m * p;
    return 10 * p;
  }

  // ------------------------------------------------------------- dashboard
  function fmt(v, d) { return v === null || v === undefined ? "–" : Number(v).toFixed(d); }

  function updateDashboard(st) {
    const k = st.f.K, L = st.laser;
    $("kTime").textContent = `${simT.toFixed(1)} s`;
    $("kCompleted").textContent = k.completed;
    $("kGen").textContent = `${k.generated} / ${k.exited}`;
    $("kThr").textContent = `${fmt(k.thr, 2)} /min`;
    $("kUtil").textContent = `${fmt(100 * k.util, 1)} %`;
    $("kPos").textContent = `${L.x.toFixed(1)}, ${L.y.toFixed(1)}`;
    $("kDist").textContent = `${fmt(k.dist, 0)} cm`;
    const pill = $("kLaserState");
    pill.textContent = L.state.replace("_FOR_PART", "");
    pill.style.color = L.state === "PROCESSING" ? colors.laser
      : L.state === "IDLE" ? colors.muted : colors.accent;
    const qMax = Math.max(4, ...k.queues);
    k.queues.forEach((q, c) => {
      $(`qn${c}`).textContent = q;
      $(`qb${c}`).style.width = `${(100 * q) / qMax}%`;
      const running = st.f.C[c][0] === 1;
      const qs = $(`qs${c}`);
      qs.textContent = running ? "RUN" : "STOP";
      qs.style.color = running ? colors.muted : colors.stop;
    });
    $("tNow").textContent = `${simT.toFixed(1)} s`;
    $("scrub").value = simT.toFixed(1);
  }

  function updateLog() {
    const ev = trace.events;
    let lo = 0, hi = ev.length;          // first event with t > simT
    while (lo < hi) { const mid = (lo + hi) >> 1; if (ev[mid].t <= simT + 1e-9) lo = mid + 1; else hi = mid; }
    if (lo === lastLogIdx) return;
    lastLogIdx = lo;
    const from = Math.max(0, lo - 60), ul = $("log");
    ul.innerHTML = ev.slice(from, lo).filter((e) => e.k !== "READY").map((e) =>
      `<li class="k-${e.k}"><span>${e.t.toFixed(2)}</span><span>${escapeHtml(e.m)}</span></li>`).join("");
    ul.scrollTop = ul.scrollHeight;
  }
  function escapeHtml(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

  // ------------------------------------------------------------- playback
  function tick(ts) {
    if (playing && trace) {
      if (lastTs !== null) {
        simT = Math.min(T, simT + ((ts - lastTs) / 1000) * speed);
        if (simT >= T) { playing = false; updatePlayBtn(); }
      }
      lastTs = ts;
      render();
    } else {
      lastTs = null;
    }
    requestAnimationFrame(tick);
  }

  function updatePlayBtn() { $("playBtn").textContent = playing ? "❚❚ Pause" : "▶ Play"; }

  function setup() {
    readColors();
    buildLegend();
    const sg = $("speedGroup");
    sg.innerHTML = SPEEDS.map((v) => `<button type="button" data-speed="${v}" aria-pressed="${v === speed}">${v}x</button>`).join("");
    sg.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-speed]"); if (!b) return;
      speed = Number(b.dataset.speed);
      sg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    });
    $("playBtn").addEventListener("click", () => {
      if (!trace) return;
      if (simT >= T) simT = 0;
      playing = !playing; updatePlayBtn();
    });
    $("resetBtn").addEventListener("click", () => {
      simT = 0; playing = false; lastLogIdx = -1; updatePlayBtn(); render();
    });
    $("scrub").addEventListener("input", (e) => { simT = Number(e.target.value); render(); });
    $("fileInput").addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0]; if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const tr = JSON.parse(reader.result);
          const key = `${tr.meta.policy} (${file.name})`;
          setTraces(Object.assign({}, traces, { [key]: tr }), key);
          $("footer").textContent = `Data: ${file.name}`; }
        catch (err) { $("footer").textContent = `Could not read ${file.name}: it is not a valid trace.json (${err.message}).`; }
      };
      reader.readAsText(file);
    });
    document.addEventListener("keydown", (e) => {
      if (e.code === "Space" && e.target === document.body) { e.preventDefault(); $("playBtn").click(); }
    });
    window.addEventListener("resize", () => { resize(); render(); });
    const retheme = () => { readColors(); buildLegend(); if (trace) { buildQueues(trace.geometry.conveyors.length); render(); } };
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
    new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    $("policySelect").addEventListener("change", (e) => load(traces[e.target.value], true));
    const data = window.SIM_TRACES || (window.SIM_TRACE ? { [window.SIM_TRACE.meta.policy]: window.SIM_TRACE } : null);
    if (data && Object.keys(data).length) {
      setTraces(data, Object.keys(data)[0]);
      simT = 0; playing = true; updatePlayBtn();   // start playing the demo straight away
    } else {
      const msg = Object.assign(document.createElement("div"), {
        id: "emptyMsg", className: "empty",
        innerHTML: "No data yet. Run <code>python run_simulation.py</code> and reload the page, " +
                   "or use “Open trace.json”.",
      });
      cell.hidden = true;
      cell.before(msg);
    }
    requestAnimationFrame(tick);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", setup);
  else setup();
})();
