import argparse
import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, quote


def load(path):
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    normalized = []
    for record in records:
        trajectory = record.get("trajectory")
        if isinstance(trajectory, dict):
            metadata = dict(record)
            metadata.pop("trajectory")
            record = dict(trajectory)
            record.update(metadata)
        normalized.append(record)
    return normalized


_TOK = None
_TOK_PATH = None


def _decode(ids):


    global _TOK
    if isinstance(ids, str):
        return ids
    if not ids or not _TOK_PATH:
        return None
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(_TOK_PATH)
    return _TOK.decode(ids)


def enrich(rec):

    paras = rec.get("paras")
    if not paras:
        return rec
    ntok = len(rec.get("tokens") or [])
    ends = [p["start"] for p in paras[1:]] + [ntok]

    seg = []
    for p, b in zip(paras, ends):
        s = p.get("summary")
        seg.append({
            "kind": p["kind"], "pid": p.get("pid"), "uid": p.get("uid"),
            "len": b - p["start"],
            "sum_len": len(s) if isinstance(s, list) else None,
            "sum_text": _decode(s),
            "folded_end": bool(p.get("folded")),
            "n_fold": 0, "n_unfold": 0, "first_fold": None,
        })
    rec["_seg"] = seg

    by_pid = {s["pid"]: s for s in seg if s["pid"] is not None}
    real = [s for s in seg if s["pid"] is not None]
    rows, folded = [], set()
    nf = nu = t = 0

    for p in paras:


        for pid in (p.get("swap") or []):
            s = by_pid.get(pid)
            if pid in folded:
                folded.discard(pid)
                nu += 1
                if s:
                    s["n_unfold"] += 1
            else:
                folded.add(pid)
                nf += 1
                if s:
                    s["n_fold"] += 1
                    s["first_fold"] = t if s["first_fold"] is None else s["first_fold"]

        if p["kind"] != "assistant":
            continue


        cut = p.get("pid")
        cur = [s for s in real if cut is not None and s["pid"] < cut]
        full = sum(s["len"] for s in cur)

        act = sum((s["sum_len"] or 0) if s["pid"] in folded else s["len"] for s in cur)
        rows.append({
            "t": t, "npara": len(cur), "nfold": sum(1 for s in cur if s["pid"] in folded),
            "full": full, "act": act, "save": (1 - act / full) if full else 0.0,
            "op_f": nf, "op_u": nu})
        nf = nu = 0
        t += 1

    tf = sum(r["full"] for r in rows)
    ta = sum(r["act"] for r in rows)
    sums = [s for s in seg if s["sum_len"]]
    probe_tok = sum(s["len"] for s in seg if s["kind"] == "probe")
    rec["_stats"] = {
        "turns": len(rows), "rows": rows, "ntok": ntok,
        "mean_save": (sum(r["save"] for r in rows) / len(rows)) if rows else 0.0,
        "agg_save": (1 - ta / tf) if tf else 0.0,
        "saved_tok": tf - ta,
        "peak_full": max((r["full"] for r in rows), default=0),
        "peak_act": max((r["act"] for r in rows), default=0),
        "n_para": len(real), "n_probe": sum(1 for s in seg if s["kind"] == "probe"),
        "probe_tok": probe_tok, "probe_frac": probe_tok / ntok if ntok else 0.0,
        "n_sum": len(sums),
        "sum_ratio": (sum(s["sum_len"] for s in sums) / sum(s["len"] for s in sums)) if sums else None,
        "op_f": sum(r["op_f"] for r in rows), "op_u": sum(r["op_u"] for r in rows),
        "folded_end": sum(1 for s in seg if s["folded_end"]),
    }
    return rec


_CACHE = {}


def payload(path):


    st = os.stat(path)
    sig = (st.st_mtime, st.st_size)
    hit = _CACHE.get(path)
    if hit and hit[0] == sig:
        return hit[1]
    body = json.dumps({"path": path, "records": [enrich(r) for r in load(path)]},
                      ensure_ascii=False).encode("utf-8")
    _CACHE[path] = (sig, body)
    return body


_STYLE = """
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0d1117; color: #e6edf3;
         font: 14px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; }
  a { color: #58a6ff; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 12px;
           font-weight: 600; }
  .r1 { background: #17351f; color: #3fb950; }
  .r0 { background: #3d1a1d; color: #f85149; }
  .rp { background: #3a3016; color: #d29922; }
  .rn { background: #21262d; color: #8b949e; }
"""

HOME_HTML = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>trace visualizer</title><style>{_STYLE}
  .box {{ max-width: 720px; margin: 12vh auto; padding: 0 20px; }}
  h1 {{ font-size: 18px; }}
  .hint {{ color: #8b949e; margin: 8px 0 20px; }}
  input {{ width: 100%; padding: 10px 12px; background: #161b22; color: #e6edf3;
           border: 1px solid #30363d; border-radius: 6px; font: inherit; }}
  button {{ margin-top: 12px; padding: 9px 18px; background: #238636; color: #fff;
            border: 0; border-radius: 6px; font: inherit; font-weight: 600; cursor: pointer; }}
</style></head><body>
<div class="box">
  <h1>rollout trace visualizer</h1>
  <div class="hint"> Enter  JSONL trace  file path （common.dump  stored format ）</div>
  <input id="p" placeholder="/mnt/rl-train/wenhaoli/gdrive/project/hf-backend/trace/bc200.jsonl"
         onkeydown="if(event.key==='Enter')go()">
  <button onclick="go()"> Open </button>
</div>
<script>
function go(){{ const p=document.getElementById('p').value.trim();
  if(p) location.href='/view?path='+encodeURIComponent(p); }}
</script></body></html>"""

VIEW_HTML = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>trace visualizer</title><style>{_STYLE}
  body {{ display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
  header {{ padding: 10px 16px; border-bottom: 1px solid #21262d; display: flex;
            align-items: center; gap: 14px; flex: 0 0 auto; }}
  header input {{ flex: 1; padding: 6px 10px; background: #161b22; color: #e6edf3;
                  border: 1px solid #30363d; border-radius: 6px; font: inherit; }}
  header .sum {{ color: #8b949e; white-space: nowrap; }}
  header .sum b {{ color: #58a6ff; }}
  #main {{ display: flex; flex: 1 1 auto; min-height: 0; }}
  #list {{ width: 240px; flex: 0 0 auto; overflow-y: auto; border-right: 1px solid #21262d; }}
  #list .item {{ padding: 8px 12px; border-bottom: 1px solid #161b22; cursor: pointer;
                 display: flex; justify-content: space-between; gap: 8px; }}
  #list .item:hover {{ background: #161b22; }}
  #list .item.sel {{ background: #1f2733; }}
  #list .item .idx {{ color: #8b949e; }}
  #detail {{ flex: 1 1 auto; overflow-y: auto; padding: 18px 26px; }}
  #detail .meta {{ margin-bottom: 14px; display: flex; gap: 10px; align-items: center; }}
  .seg-title {{ color: #8b949e; text-transform: uppercase; font-size: 11px; letter-spacing: .05em;
                margin: 20px 0 8px; border-bottom: 1px solid #21262d; padding-bottom: 4px; }}
  details.prompt pre {{ white-space: pre-wrap; word-break: break-word; color: #adbac7; }}
  .plain {{ white-space: pre-wrap; word-break: break-word; }}


  .para {{ border: 1px solid #21262d; border-left: 3px solid #30363d; border-radius: 8px;
           margin: 12px 0; background: #0f141a; overflow: hidden; }}
  .para > summary {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
                     padding: 7px 12px; background: #12181f; cursor: pointer;
                     font-size: 11px; color: #6e7681; list-style: none; }}
  .para > summary::-webkit-details-marker {{ display: none; }}
  .para > summary::before {{ content: '\\25B8'; color: #6e7681; }}
  .para[open] > summary::before {{ content: '\\25BE'; }}
  .para > summary:hover {{ background: #161d26; }}
  .para .kind {{ font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }}
  .para .tag {{ background: #161b22; border: 1px solid #21262d; border-radius: 4px;
                padding: 0 6px; white-space: nowrap; }}
  .para .tag.pid {{ color: #79c0ff; }}
  .para .tag.uid {{ color: #d2a8ff; }}
  .para .tag.fold {{ background: #3a3016; border-color: #4a3d1a; color: #d29922; }}
  .para .prev {{ color: #4a525c; flex: 1 1 140px; min-width: 0; overflow: hidden;
                 white-space: nowrap; text-overflow: ellipsis; }}
  .pbody {{ padding: 8px 14px 10px; }}
  .k-prompt {{ border-left-color: #58a6ff; }}    .k-prompt .kind {{ color: #58a6ff; }}
  .k-assistant {{ border-left-color: #adbac7; }} .k-assistant .kind {{ color: #adbac7; }}
  .k-obs {{ border-left-color: #3fb950; }}       .k-obs .kind {{ color: #3fb950; }}
  .k-probe {{ border-left-color: #a371f7; }}     .k-probe .kind {{ color: #a371f7; }}


  .sub {{ margin: 8px 0; border-left: 2px solid #21262d; padding: 1px 0 1px 12px; }}
  .sub > .sh {{ font-size: 11px; color: #6e7681; letter-spacing: .05em;
                text-transform: uppercase; margin-bottom: 3px; }}
  .sub > pre {{ white-space: pre-wrap; word-break: break-word; margin: 0;
                max-height: 340px; overflow: auto; color: #8b949e; }}
  .sub.think > pre {{ color: #8b949e; }}
  .sub.call {{ border-left-color: #f0883e; }}
  .sub.call > .sh {{ color: #f0883e; font-weight: 600; text-transform: none;
                     font-size: 13px; letter-spacing: 0; }}
  .sub.call .param {{ margin-top: 3px; white-space: pre-wrap; word-break: break-word; }}
  .sub.call .pk {{ color: #79c0ff; }}
  .sub.call .pv {{ color: #adbac7; }}
  .sub.call details.raw summary {{ color: #6e7681; cursor: pointer; font-size: 11px; }}
  .sub.call details.raw summary:hover {{ color: #f0883e; }}
  .sub.call details.raw pre {{ white-space: pre-wrap; word-break: break-word; color: #adbac7;
                               background: #0d1117; border: 1px solid #21262d; border-radius: 4px;
                               padding: 6px 8px; margin: 5px 0 2px; max-height: 300px; overflow: auto; }}
  .sub.resp {{ border-left-color: #2b6e3b; }}
  .sub.pout {{ border-left-color: #6f42a8; }}
  .sub.pout > pre {{ color: #c9b6ef; }}
  .sub.sum {{ border-left-color: #d29922; }}
  .sub.sum > .sh {{ color: #d29922; }}
  .sub.sum > pre {{ color: #cdb995; }}
  .sub.text > pre {{ color: #adbac7; }}
  .sub .answer {{ background: none; border: 0; padding: 0; }}
  .answer {{ background: #0f141a; border: 1px solid #21262d; border-radius: 8px; padding: 4px 18px; }}
  .answer table {{ border-collapse: collapse; margin: 10px 0; }}
  .answer th, .answer td {{ border: 1px solid #30363d; padding: 4px 10px; }}
  .answer th {{ background: #161b22; }}
  .answer code {{ background: #161b22; padding: 1px 5px; border-radius: 4px; }}
  .answer h3, .answer h4 {{ margin: 12px 0 6px; }}
  #err {{ padding: 20px; color: #f85149; }}
  .hidden {{ display: none !important; }}
  table.grid {{ border-collapse: collapse; font-size: 12px; margin: 8px 0; }}
  table.grid th, table.grid td {{ border: 1px solid #21262d; padding: 3px 10px; text-align: right; }}
  table.grid th {{ background: #161b22; color: #8b949e; font-weight: 600;
                   position: sticky; top: -18px; }}
  table.grid td.l, table.grid th.l {{ text-align: left; }}
  table.grid tr:hover td {{ background: #12181f; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }}
  .card {{ background: #10161d; border: 1px solid #21262d; border-radius: 8px;
           padding: 8px 14px; min-width: 128px; }}
  .card .k {{ font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: .04em; }}
  .card .v {{ font-size: 17px; color: #e6edf3; }}
  .card .v small {{ font-size: 12px; color: #6e7681; }}
  .bar {{ width: 150px; height: 8px; background: #0d1117; border: 1px solid #21262d;
          border-radius: 3px; overflow: hidden; }}
  .bar span {{ display: block; height: 100%; }}
  .bar .f {{ background: #30363d; }}
  .bar .a {{ background: #3fb950; }}
  .note {{ color: #8b949e; margin: 6px 0 2px; }}
  .jbtn {{ background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px;
           padding: 2px 12px; font: inherit; cursor: pointer; }}
  .jbtn:hover {{ background: #2d333b; }}
  .jbtn.active {{ background: #1f6feb; color: #fff; border-color: #1f6feb; }}
</style></head><body>
<header>
  <span>trace</span>
  <input id="path" onkeydown="if(event.key==='Enter')reload()">
  <span class="sum" id="sum"></span>
</header>
<div id="main">
  <div id="list"></div>
  <div id="detail"></div>
</div>
<script>
const qs = new URLSearchParams(location.search);
let RECS = [];

function esc(s){{ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

function rewardClass(r){{ if(r==null) return 'rn'; if(r>=1) return 'r1'; if(r<=0) return 'r0'; return 'rp'; }}
function rewardText(r){{ return r==null ? '—' : (Number.isInteger(r)? r : r.toFixed(2)); }}
function rewardOf(rec){{ return (rec.reward!=null) ? rec.reward : (rec.judge ? rec.judge.score : null); }}

function parseTurns(text){{
  const turns=[];
  for(const p of text.split('<|im_start|>')){{
    if(!p.trim()) continue;
    let body=p; const e=body.indexOf('<|im_end|>'); if(e!==-1) body=body.slice(0,e);
    const nl=body.indexOf('\\n');
    const role=(nl===-1?body:body.slice(0,nl)).trim();
    const content=(nl===-1?'':body.slice(nl+1));
    turns.push({{role, content}});
  }}
  return turns;
}}

const sub=(cls,head,body)=>`<div class="sub ${{cls}}">${{head?`<div class="sh">${{head}}</div>`:''}}${{body}}</div>`;




function parseCall(tc){{
  const fm=tc.match(/<function=([^>]+)>/);
  if(fm){{
    const ps=[], re=/<parameter=([^>]+)>([\\s\\S]*?)<\\/parameter>/g; let m;
    while((m=re.exec(tc))) ps.push([m[1].trim(), m[2].trim()]);
    return {{fn: fm[1].trim(), ps}};
  }}
  try{{
    const o=JSON.parse(tc.trim()), a=o.arguments;
    return {{fn: ''+(o.name||'?'),
             ps: (a&&typeof a==='object')? Object.entries(a).map(([k,v])=>[k,''+v]) : []}};
  }}catch(e){{ return {{fn:'?', ps:[]}}; }}
}}

function renderToolCall(tc){{
  const c=parseCall(tc);
  const ph=c.ps.map(([k,v])=>
    `<div class="param"><span class="pk">${{esc(k)}}</span> = <span class="pv">${{esc(v)}}</span></div>`).join('');
  const raw=`<tool_call>${{tc}}</tool_call>`;
  return sub('call', `&rarr; ${{esc(c.fn)}}()`,
    ph + `<details class="raw"><summary>raw</summary><pre>${{esc(raw)}}</pre></details>`);
}}


function assistantPreview(c){{
  const m=c.match(/<tool_call>([\\s\\S]*?)<\\/tool_call>/);
  if(!m) return c.replace(/<think>[\\s\\S]*?<\\/think>/g,' ');
  const p=parseCall(m[1]);
  return `\\u2192 ${{p.fn}}(${{p.ps.map(([k,v])=>k+'='+v).join(', ')}})`;
}}



function renderAssistant(c){{
  let html='', last=0, m;

  const re=/<think>([\\s\\S]*?)<\\/think>|<tool_call>([\\s\\S]*?)<\\/tool_call>|<think>([\\s\\S]*)$/g;
  while((m=re.exec(c))){{
    const plain=c.slice(last,m.index).trim();
    if(plain) html+=sub('ans','text',`<div class="answer">${{renderMarkdown(plain)}}</div>`);
    if(m[1]!=null) html+=sub('think','think',`<pre>${{esc(m[1].trim())}}</pre>`);
    else if(m[2]!=null) html+=renderToolCall(m[2]);
    else html+=sub('think','think ( unclosed )',`<pre>${{esc(m[3].trim())}}</pre>`);
    last=re.lastIndex;
  }}
  const tail=c.slice(last).trim();
  if(tail) html+=sub('ans','text',`<div class="answer">${{renderMarkdown(tail)}}</div>`);
  return html;
}}

function obsBody(c){{
  const m=c.match(/<tool_response>([\\s\\S]*?)<\\/tool_response>/);
  const body=m? m[1].trim() : c.trim();
  let pretty=body; try{{ pretty=JSON.stringify(JSON.parse(body),null,2); }}catch(e){{}}
  return sub('resp','tool response',`<pre>${{esc(pretty)}}</pre>`);
}}

function inline(s){{ return esc(s)
  .replace(/\\*\\*([^*]+)\\*\\*/g,'<b>$1</b>')
  .replace(/`([^`]+)`/g,'<code>$1</code>')
  .replace(/\\*([^*]+)\\*/g,'<i>$1</i>'); }}

function renderMarkdown(md){{
  const L=md.split('\\n'); let h='', i=0;
  const sep=s=>/^[\\s|:-]+$/.test(s)&&s.includes('-')&&s.includes('|');
  while(i<L.length){{
    let ln=L[i];
    if(/\\|/.test(ln) && i+1<L.length && sep(L[i+1])){{
      const head=ln.split('|').slice(1,-1).map(x=>x.trim()); i+=2; const rows=[];
      while(i<L.length && /\\|/.test(L[i]) && L[i].trim()){{ rows.push(L[i].split('|').slice(1,-1).map(x=>x.trim())); i++; }}
      h+='<table><thead><tr>'+head.map(x=>`<th>${{inline(x)}}</th>`).join('')+'</tr></thead><tbody>'
        +rows.map(r=>'<tr>'+r.map(c=>`<td>${{inline(c)}}</td>`).join('')+'</tr>').join('')+'</tbody></table>'; continue;
    }}
    const hd=ln.match(/^(#{{1,6}})\\s+(.*)$/);
    if(hd){{ const n=Math.min(hd[1].length+2,6); h+=`<h${{n}}>${{inline(hd[2])}}</h${{n}}>`; i++; continue; }}
    if(/^\\s*[-*]\\s+/.test(ln)){{ const it=[]; while(i<L.length && /^\\s*[-*]\\s+/.test(L[i])){{ it.push(L[i].replace(/^\\s*[-*]\\s+/,'')); i++; }}
      h+='<ul>'+it.map(x=>`<li>${{inline(x)}}</li>`).join('')+'</ul>'; continue; }}
    if(!ln.trim()){{ i++; continue; }}
    const para=[ln]; i++;
    while(i<L.length && L[i].trim() && !/^\\s*[-*#]|\\|/.test(L[i])){{ para.push(L[i]); i++; }}
    h+=`<p>${{inline(para.join(' '))}}</p>`;
  }}
  return h;
}}

function renderJudge(j){{
  if(!j) return `<div class="seg-title">judger</div><div class="plain" style="color:#8b949e"> no  judger（ This  trace  scoring details were not stored ）</div>`;
  let h='';
  h+=`<div class="seg-title">golden</div><div class="plain">${{(j.golden||[]).map(esc).join('\\n')||'—'}}</div>`;
  h+=`<div class="seg-title">predicted</div>`
    + (j.predicted==null ? `<div class="plain">（ empty ）</div>` : `<div class="answer">${{renderMarkdown(j.predicted)}}</div>`);
  if(!j.prompt && !j.error) h+=`<div class="plain" style="color:#3fb950">exact match —  not called  LLM judge</div>`;
  if(j.verdict!=null) h+=`<div class="seg-title">verdict</div><div class="plain">${{esc(j.verdict)}}</div>`;
  if(j.prompt) h+=`<details class="prompt"><summary class="seg-title" style="display:inline">grade prompt</summary><pre>${{esc(j.prompt)}}</pre></details>`;
  if(j.error) h+=`<div class="seg-title">error</div><div class="plain" style="color:#f85149">${{esc(j.error)}}</div>`;
  return h;
}}

const fmtN = x => x==null? '—' : x.toLocaleString();
const fmtP = x => x==null? '—' : (100*x).toFixed(1)+'%';

const oneline=t=>{{ const s=(t||'').replace(/\\s+/g,' ').trim();
  return s.length>120? s.slice(0,120)+'…' : s; }};



function paraHead(s, i, kind, prev){{
  const g=[`<span class="tag">#${{i}}</span>`, `<span class="kind">${{esc(kind)}}</span>`];
  if(s){{
    if(s.pid!=null) g.push(`<span class="tag pid">pid ${{s.pid}}</span>`);
    if(s.uid) g.push(`<span class="tag uid">&lt;id&gt;${{esc(s.uid)}}&lt;/id&gt;</span>`);
    g.push(`<span class="tag">${{fmtN(s.len)}} tok</span>`);
    if(s.sum_len!=null)
      g.push(`<span class="tag fold"> Summary  ${{fmtN(s.sum_len)}} tok · ${{(100*s.sum_len/s.len).toFixed(0)}}%</span>`);
  }}
  return g.join('')+`<span class="prev">${{esc(oneline(prev))}}</span>`;
}}


function summaryBlock(s){{
  if(!s || s.sum_len==null) return '';
  const body = s.sum_text!=null ? esc(s.sum_text)
    : `（${{s.sum_len}}  items  token， not decoded ）\\n Start with  --tokenizer < model path >  to display summary text `;
  return sub('sum', `summary · ${{s.sum_len}} tok（ replaces the full paragraph when folded  ${{fmtN(s.len)}} tok  full context ）`,
             `<pre>${{body}}</pre>`);
}}



function paraCard(s, i, t){{

  const kind = s? s.kind : t.role==='assistant'?'assistant' : t.role==='unfold'?'probe'
             : /<tool_response>/.test(t.content)?'obs' : 'prompt';
  let body, prev=t.content;
  if(kind==='probe'){{
    body=sub('pout','unfold  output  ·  Model-selected paragraphs to unfold ',`<pre>${{esc(t.content.trim())}}</pre>`);
  }}else if(kind==='obs'){{
    body=obsBody(t.content); prev=t.content.replace(/<\\/?tool_response>/g,'');
  }}else if(kind==='assistant'){{
    body=renderAssistant(t.content); prev=assistantPreview(t.content);
  }}else{{
    body=sub('text','',`<pre>${{esc(t.content.trim())}}</pre>`);
  }}
  return `<details class="para k-${{kind}}">`
    + `<summary>${{paraHead(s,i,kind,prev)}}</summary>`
    + `<div class="pbody">${{body}}${{summaryBlock(s)}}</div></details>`;
}}

function noFold(st){{
  return `<div class="note"> This  trace  disabled  context folding：0  summaries 、0  fold operations 。`
    + ` Set  agent/search/config.py  of  <code>UNFOLD_CONFIGS["enable"]</code>  and rerun to generate data 。</div>`;
}}

function renderParas(r){{
  const seg=r._seg;
  if(!seg) return `<div class="note"> This  trace  missing  paras  field （ legacy trace format ）</div>`;
  const rows=seg.map((s,i)=>`<tr>
    <td>${{i}}</td><td class="l">${{esc(s.kind)}}</td>
    <td>${{s.pid==null?'—':s.pid}}</td><td class="l">${{s.uid||'—'}}</td>
    <td>${{fmtN(s.len)}}</td><td>${{s.sum_len==null?'—':fmtN(s.sum_len)}}</td>
    <td>${{s.sum_len==null?'—':(100*s.sum_len/s.len).toFixed(0)+'%'}}</td>
    <td>${{s.n_fold||'—'}}</td><td>${{s.n_unfold||'—'}}</td></tr>`).join('');
  return `<div class="seg-title">paragraphs（${{seg.length}}  paragraph ）</div>`
    + `<div class="note">pid  is  server  stable paragraph ID on the server ，probe  paragraphs are excluded from  server view  therefore no  pid。`
    + `「 Fold / Unfold 」 replayed  swap  counts from replay  —— swap  is a pure  toggle， direction is not stored 。</div>`
    + `<table class="grid"><thead><tr><th>#</th><th class="l">kind</th><th>pid</th>`
    + `<th class="l">uid</th><th>tokens</th><th> Summary </th><th> Compression </th>`
    + `<th> Fold </th><th> Unfold </th></tr></thead><tbody>${{rows}}</tbody></table>`;
}}

function renderMetrics(r){{
  const st=r._stats;
  if(!st) return `<div class="note"> This  trace  missing  paras  field （ legacy trace format ）， cannot compute folding metrics </div>`;
  const card=(k,v)=>`<div class="card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`;
  let h='';
  if(!st.n_sum) h+=noFold(st);
  h+=`<div class="seg-title"> Context savings </div><div class="cards">`
    + card(' Mean savings per turn ', fmtP(st.mean_save))
    + card(' Aggregate savings ', fmtP(st.agg_save)+` <small>token  weighted </small>`)
    + card(' Total saved ', fmtN(st.saved_tok)+` <small>tok· turn </small>`)
    + card(' Peak context ', fmtN(st.peak_act)+` <small>/ ${{fmtN(st.peak_full)}}</small>`)
    + `</div>`;
  h+=`<div class="seg-title"> Overhead and scale </div><div class="cards">`
    + card(' Turn count ', st.turns)
    + card(' Paragraph count ', st.n_para+` <small>+${{st.n_probe}} probe</small>`)
    + card(' Probe overhead ', fmtP(st.probe_frac)+` <small>${{fmtN(st.probe_tok)}} tok</small>`)
    + card(' Summary count ', st.n_sum)
    + card(' Summary compression ratio ', fmtP(st.sum_ratio))
    + card(' Fold  /  Unfold   count ', st.op_f+' / '+st.op_u)
    + card(' Folded paragraphs at completion ', st.folded_end)
    + card('transcript', fmtN(st.ntok)+` <small>tok</small>`)
    + `</div>`;
  if(!st.turns) return h;
  const mx=Math.max(1,...st.rows.map(x=>x.full));
  const rows=st.rows.map(x=>`<tr>
    <td>${{x.t}}</td><td>${{x.npara}}</td><td>${{x.nfold}}</td>
    <td>${{fmtN(x.full)}}</td><td>${{fmtN(x.act)}}</td><td>${{fmtP(x.save)}}</td>
    <td class="l"><div class="bar" title=" active  ${{x.act}} /  full context  ${{x.full}}">
      <span class="f" style="width:${{(100*x.full/mx).toFixed(1)}}%">
        <span class="a" style="width:${{x.full?(100*x.act/x.full).toFixed(1):0}}%"></span></span></div></td>
    <td>${{x.op_f||''}}</td><td>${{x.op_u||''}}</td></tr>`).join('');
  h+=`<div class="seg-title"> Context per turn </div>`
    + `<div class="note">「 full context 」=  without folding ， for this request  server  loaded  token  count ；`
    + `「 active 」=  tokens loaded with folding 。 Gray bar length  ∝  full context ， Green portion  ∝  retained active context 。</div>`
    + `<table class="grid"><thead><tr><th> turn </th><th> Paragraph count </th><th> folding </th><th> full context </th>`
    + `<th> active </th><th> Savings </th><th class="l">　</th><th> Fold </th><th> Unfold </th></tr></thead>`
    + `<tbody>${{rows}}</tbody></table>`;
  return h;
}}

let CUR=null, SEL=-1, VIEW='traj';
function setView(v){{ VIEW=v; if(CUR) renderDetail(CUR); }}


function toggleAll(btn){{
  const ds=[...document.querySelectorAll('#detail details.para')];
  if(!ds.length) return;
  const open=ds.some(d=>!d.open);
  ds.forEach(d=>d.open=open);
  btn.textContent = open? ' Collapse all ' : ' Expand all ';
}}

function traceBody(r){{
  if(r.text!=null){{

    const turns=parseTurns(r.text);


    const seg=(r._seg && r._seg.length===turns.length) ? r._seg : null;


    let fa = seg ? seg.findIndex(s=>s.kind!=='prompt') : turns.findIndex(t=>t.role==='assistant');
    if(fa<0) fa=turns.length;
    const card=i=>paraCard(seg?seg[i]:null, i, turns[i]);
    const promptHtml=turns.slice(0,fa).map((t,i)=>card(i)).join('');
    const flow=turns.slice(fa).map((t,k)=>card(fa+k)).join('');
    return `<div class="seg-title">prompt</div>${{promptHtml||'—'}}`
      +`<div class="seg-title">trajectory</div>${{flow||'<div class="plain">（ empty ）</div>'}}`;
  }}

  const turns=parseTurns(r.thinking||'');
  const flow=turns.filter(t=>t.role==='assistant'||t.role==='user')
    .map((t,i)=>paraCard(null,i,{{role:t.role==='assistant'?'assistant':'user',content:t.content}})).join('');
  return `<details class="prompt"><summary class="seg-title" style="display:inline">prompt</summary><pre>${{esc(r.prompt||'')}}</pre></details>`
    +`<div class="seg-title">trajectory</div>${{flow||'<div class="plain">（ no  thinking）</div>'}}`
    +`<div class="seg-title">final answer</div><div class="answer">${{renderMarkdown(r.response||'')}}</div>`;
}}

function renderDetail(r){{
  const d=document.getElementById('detail');
  const j=r.judge;

  const kind = !j ? '' : (j.error ? 'llm error' : (j.prompt ? 'LLM judge' : 'exact match'));
  const st=r._stats;
  const tab=v=>`class="jbtn${{VIEW===v?' active':''}}"`;
  const head=`
    <div class="meta"><span class="badge ${{rewardClass(rewardOf(r))}}">reward ${{rewardText(rewardOf(r))}}</span>
      <span class="badge rn">${{esc(r.status||'')}}</span>
      ${{kind?`<span class="badge rn">${{kind}}</span>`:''}}
      ${{st?`<span class="badge rn">${{st.turns}}  turn  · ${{fmtN(st.ntok)}} tok</span>`:''}}
      ${{st&&st.n_sum?`<span class="badge rp"> Savings  ${{fmtP(st.agg_save)}}</span>`:''}}
      <span style="flex:1"></span>
      ${{VIEW==='traj'?`<button class="jbtn" onclick="toggleAll(this)"> Expand all </button>`:''}}
      <button ${{tab('traj')}} onclick="setView('traj')">trajectory</button>
      <button ${{tab('paras')}} onclick="setView('paras')">paras</button>
      <button ${{tab('metrics')}} onclick="setView('metrics')">metrics</button>
      <button ${{tab('judge')}} onclick="setView('judge')">judger</button></div>`;
  const body = VIEW==='judge' ? renderJudge(j)
             : VIEW==='paras' ? renderParas(r)
             : VIEW==='metrics' ? renderMetrics(r) : traceBody(r);
  d.innerHTML = head + body;
  d.scrollTop=0;
}}

function select(i){{
  SEL=i;
  document.querySelectorAll('#list .item').forEach((el,j)=>el.classList.toggle('sel',j===i));
  CUR=RECS[i]; renderDetail(CUR);
}}

function renderList(){{
  const list=document.getElementById('list');
  list.innerHTML=RECS.map((r,i)=>
    `<div class="item" onclick="select(${{i}})"><span class="idx">#${{i}}</span>`
    +`<span class="badge ${{rewardClass(rewardOf(r))}}">${{rewardText(rewardOf(r))}}</span></div>`).join('');
  const rs=RECS.map(rewardOf).filter(x=>typeof x==='number');
  const mean=rs.length? (rs.reduce((a,b)=>a+b,0)/rs.length) : null;

  let tf=0, ta=0, nt=0;
  for(const r of RECS){{ const s=r._stats; if(!s) continue;
    nt+=s.turns; for(const x of s.rows){{ tf+=x.full; ta+=x.act; }} }}
  const save = tf? (1-ta/tf) : null;
  document.getElementById('sum').innerHTML=`<b>${{RECS.length}}</b> traj · mean reward `
    +`<b>${{mean==null?'—':mean.toFixed(3)}}</b>`
    +(nt?` · ${{nt}}  turn  ·  Savings  <b>${{fmtP(save)}}</b>`:'');
}}

function reload(){{ const p=document.getElementById('path').value.trim();
  if(p) location.href='/view?path='+encodeURIComponent(p); }}

let LAST_SIG='';

async function refresh(){{
  const path=qs.get('path')||'';
  const res=await fetch('/api?path='+encodeURIComponent(path));
  const data=await res.json();
  if(data.error){{ document.getElementById('detail').innerHTML=`<div id="err"> Read failed ：${{esc(data.error)}}</div>`; return; }}
  const prev = (SEL>=0 && SEL<RECS.length) ? JSON.stringify(RECS[SEL]) : null;
  RECS=data.records; renderList();
  if(SEL>=0 && SEL<RECS.length){{
    document.querySelectorAll('#list .item').forEach((el,j)=>el.classList.toggle('sel',j===SEL));
    CUR=RECS[SEL];
    const now=JSON.stringify(RECS[SEL]);
    if(now!==prev){{ const d=document.getElementById('detail'); const top=d.scrollTop; renderDetail(CUR); d.scrollTop=top; }}
  }} else if(RECS.length){{ select(0); }}
}}

async function poll(){{
  const path=qs.get('path')||'';
  try{{
    const st=await (await fetch('/stat?path='+encodeURIComponent(path))).json();
    if(st.error) return;
    const sig=st.mtime+':'+st.size;
    if(sig!==LAST_SIG){{ LAST_SIG=sig; await refresh(); }}
  }}catch(e){{}}
}}

async function boot(){{
  const path=qs.get('path')||'';
  document.getElementById('path').value=path;
  try{{
    await refresh();
    setInterval(poll, 2000);
  }}catch(e){{ document.getElementById('detail').innerHTML=`<div id="err">${{esc(''+e)}}</div>`; }}
}}
boot();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self._send(200, HOME_HTML)
        elif u.path == "/view":
            self._send(200, VIEW_HTML)
        elif u.path == "/api":
            path = (q.get("path") or [""])[0]
            try:
                self._send(200, payload(path), "application/json; charset=utf-8")
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)}), "application/json; charset=utf-8")
        elif u.path == "/stat":
            path = (q.get("path") or [""])[0]
            try:
                st = os.stat(path)
                self._send(200, json.dumps({"mtime": st.st_mtime, "size": st.st_size}),
                           "application/json; charset=utf-8")
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)}), "application/json; charset=utf-8")
        else:
            self._send(404, "not found")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="JSONL trace  file path ； open directly when provided ")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--no-open", action="store_true", help=" Do not open the browser automatically ")
    ap.add_argument("--tokenizer", default=os.path.join(
                        os.getenv("GDRIVE_LOCAL", ""), "model/Qwen3.5-4B"),
                    help=" model /tokenizer  path ； legacy only  trace  of  token-id summary  required ")
    a = ap.parse_args()

    global _TOK_PATH
    _TOK_PATH = a.tokenizer

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    base = f"http://{a.host}:{a.port}"
    url = base + ("/view?path=" + quote(os.path.abspath(a.path)) if a.path else "/")
    print(f"serving trace visualizer on {url}")
    if not a.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
