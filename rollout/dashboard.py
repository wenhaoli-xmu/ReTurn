DASHBOARD_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rollout dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0d1117; color: #e6edf3;
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { padding: 14px 20px; border-bottom: 1px solid #21262d;
           display: flex; align-items: baseline; gap: 16px; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .agg { color: #8b949e; }
  header .agg b { color: #58a6ff; font-weight: 600; }
  #grid { display: grid; gap: 16px; padding: 20px;
          grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }
  .card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 14px 16px; }
  .card h2 { font-size: 14px; margin: 0 0 10px; }
  .row { display: flex; justify-content: space-between; margin: 3px 0; }
  .row .k { color: #8b949e; }
  .row .v { font-variant-numeric: tabular-nums; }
  .charts { margin-top: 10px; }
  .chart-label { display: flex; justify-content: space-between; margin-top: 8px;
                 font-size: 12px; color: #8b949e; }
  .chart-label .decode { color: #3fb950; }
  .chart-label .prefill { color: #f0883e; }
  .chart-label .power { color: #58a6ff; }
  .chart-label .kv { color: #bc8cff; }
  canvas { display: block; width: 100%; height: 70px; }
  .stale { opacity: .4; }
</style>
</head>
<body>
<header>
  <h1>rollout dashboard</h1>
  <span class="agg" id="agg"></span>
  <span class="agg" style="margin-left:auto" id="clock"></span>
</header>
<div id="grid"></div>
<script>
const N = 120;
const hist = {};
const grid = document.getElementById("grid");

function fmt(x) { return x == null ? "—" : x.toFixed(1); }

function push(arr, v) { arr.push(v); if (arr.length > N) arr.shift(); }

function draw(canvas, series, color, fixedMax, noClear) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!noClear) { canvas.width = w * dpr; canvas.height = h * dpr; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!noClear) ctx.clearRect(0, 0, w, h);
  const vals = series.filter(v => v != null);
  const max = fixedMax || Math.max(1, ...vals);
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
  series.forEach((v, i) => {
    const x = (i / (N - 1)) * w;
    const y = h - (v == null ? 0 : v / max) * (h - 4) - 2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  if (!noClear) {
    ctx.fillStyle = color; ctx.font = "11px monospace";
    ctx.fillText("max " + max.toFixed(0), 4, 12);
  }
}

function ensureCard(id) {
  let card = document.getElementById("card-" + id);
  if (card) return card;
  hist[id] = { decode: [], prefill: [], prefilling: [], decoding: [], reside: [], pending: [], power: [], dcduty: [], pfduty: [] };
  card = document.createElement("div");
  card.className = "card"; card.id = "card-" + id;
  card.innerHTML = `
    <h2>engine ${id}</h2>
    <div class="row"><span class="k">prefilling</span><span class="v" id="pfc-${id}">0</span></div>
    <div class="row"><span class="k">decoding</span><span class="v" id="dec-${id}">0</span></div>
    <div class="row"><span class="k">reside</span><span class="v" id="res-${id}">0</span></div>
    <div class="row"><span class="k">pending</span><span class="v" id="pend-${id}">0</span></div>
    <div class="row"><span class="k">decode tok/s</span><span class="v" id="dc-${id}">—</span></div>
    <div class="row"><span class="k">prefill tok/s</span><span class="v" id="pftps-${id}">—</span></div>
    <div class="row"><span class="k">decode streak</span><span class="v" id="dstreak-${id}">—</span></div>
    <div class="row"><span class="k">GPU power</span><span class="v" id="pw-${id}">—</span></div>
    <div class="row"><span class="k">kv usage</span><span class="v" id="kv-${id}">—</span></div>
    <div class="row"><span class="k">duty dc/pf/idle</span><span class="v" id="duty-${id}">—</span></div>
    <div class="charts">
      <div class="chart-label"><span class="decode">decode tok/s</span><span id="dc2-${id}"></span></div>
      <canvas id="cdc-${id}"></canvas>
      <div class="chart-label"><span class="prefill">prefill tok/s</span><span id="pf2-${id}"></span></div>
      <canvas id="cpf-${id}"></canvas>
      <div class="chart-label"><span><span class="decode">decode</span> / <span class="prefill">prefill</span> duty (%)</span><span id="duty2-${id}"></span></div>
      <canvas id="cduty-${id}"></canvas>
      <div class="chart-label"><span class="power">GPU power (W)</span><span id="pw2-${id}"></span></div>
      <canvas id="cpw-${id}"></canvas>
    </div>`;
  grid.appendChild(card);
  return card;
}

async function tick() {
  let data;
  try { data = await (await fetch("/stats")).json(); }
  catch (e) { document.getElementById("clock").textContent = "offline"; return; }

  let totPf = 0, totDec = 0, totRes = 0, totPend = 0, totDc = 0, totPfTps = 0, totPw = 0, kvUsed = 0, kvTotal = 0;
  let dutyDcSum = 0, dutyPfSum = 0, dutyN = 0;
  for (const id of Object.keys(data)) {
    ensureCard(id);
    const s = data[id], hs = hist[id];
    const kvPct = s.kv_total ? s.kv_used / s.kv_total * 100 : null;
    const dcDuty = s.decode_duty == null ? null : s.decode_duty * 100;
    const pfDuty = s.prefill_duty == null ? null : s.prefill_duty * 100;
    push(hs.decode, s.decode_tps); push(hs.prefill, s.prefill_tps);
    push(hs.prefilling, s.prefilling); push(hs.decoding, s.decoding);
    push(hs.reside, s.reside); push(hs.pending, s.pending);
    push(hs.power, s.power);
    push(hs.dcduty, dcDuty); push(hs.pfduty, pfDuty);
    document.getElementById("pfc-" + id).textContent = s.prefilling;
    document.getElementById("dec-" + id).textContent = s.decoding;
    document.getElementById("res-" + id).textContent = s.reside;
    document.getElementById("pend-" + id).textContent = s.pending;
    document.getElementById("dc-" + id).textContent = fmt(s.decode_tps);
    document.getElementById("pftps-" + id).textContent = fmt(s.prefill_tps);
    document.getElementById("dstreak-" + id).textContent = fmt(s.decode_streak);
    document.getElementById("duty-" + id).textContent = (dcDuty == null && pfDuty == null) ? "—"
      : `${(dcDuty || 0).toFixed(0)}/${(pfDuty || 0).toFixed(0)}/${(100 - (dcDuty || 0) - (pfDuty || 0)).toFixed(0)}%`;
    document.getElementById("pw-" + id).textContent = s.power == null ? "—"
      : `${s.power.toFixed(0)} W`;
    document.getElementById("kv-" + id).textContent = kvPct == null ? "—"
      : `${kvPct.toFixed(0)}%`;
    draw(document.getElementById("cdc-" + id), hs.decode, "#3fb950");
    draw(document.getElementById("cpf-" + id), hs.prefill, "#f0883e");
    const cduty = document.getElementById("cduty-" + id);
    draw(cduty, hs.dcduty, "#3fb950", 100);
    draw(cduty, hs.pfduty, "#f0883e", 100, true);
    draw(document.getElementById("cpw-" + id), hs.power, "#58a6ff");
    totPf += s.prefilling; totDec += s.decoding; totRes += s.reside; totPend += s.pending;
    totDc += s.decode_tps || 0; totPfTps += s.prefill_tps || 0; totPw += s.power || 0;
    if (dcDuty != null) { dutyDcSum += dcDuty; dutyPfSum += pfDuty || 0; dutyN++; }
    if (s.kv_total) { kvUsed += s.kv_used; kvTotal += s.kv_total; }
  }
  const kvAgg = kvTotal ? (kvUsed / kvTotal * 100).toFixed(0) + "%" : "—";
  const dutyAgg = dutyN ? `${(dutyDcSum / dutyN).toFixed(0)}/${(dutyPfSum / dutyN).toFixed(0)}%` : "—";
  document.getElementById("agg").innerHTML =
    `prefilling <b>${totPf}</b> · decoding <b>${totDec}</b> · reside <b>${totRes}</b> · pending <b>${totPend}</b> · ` +
    `decode <b>${totDc.toFixed(0)}</b> tok/s · prefill <b>${totPfTps.toFixed(0)}</b> tok/s · duty <b>${dutyAgg}</b> · power <b>${totPw.toFixed(0)}</b> W · ` +
    `KV <b>${kvAgg}</b>`;
  document.getElementById("clock").textContent = new Date().toLocaleTimeString();
}

tick();
setInterval(tick, 1000);
</script>
</body>
</html>"""
