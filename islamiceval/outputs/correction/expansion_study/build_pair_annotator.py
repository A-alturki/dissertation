#!/usr/bin/env python3
"""Build a self-contained blind pair-annotation HTML tool from pairs_blind_100.json.
Same/different/unsure schema, matching the researcher's original 40-pair labels
(pair_labels.json). Progress persists in localStorage; export -> pair_labels_human100.json
in the same {pid, label} shape as pair_labels.json / gemini_labels_100.json.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
pairs = json.load(open(os.path.join(HERE, "pairs_blind_100.json"), encoding="utf-8"))

HTML = """<!doctype html><html lang=ar dir=rtl><head><meta charset=utf-8>
<title>Hadith pair reannotation — same/different matn</title>
<style>
:root{color-scheme:light dark}
body{font-family:"Segoe UI",Tahoma,sans-serif;max-width:900px;margin:0 auto;padding:16px;
  background:#0f1115;color:#e8e8e8}
h1{font-size:18px;margin:0 0 4px}
#bar{position:sticky;top:0;background:#0f1115;padding:10px 0;border-bottom:1px solid #333;
  display:flex;align-items:center;gap:12px;flex-wrap:wrap;z-index:5}
#progress{font-variant-numeric:tabular-nums}
#pidsel{font-size:12px;opacity:.8}
button{cursor:pointer;border:1px solid #444;background:#1c1f26;color:#e8e8e8;border-radius:6px;
  padding:6px 12px;font-size:14px}
button:hover{background:#262a33}
button.active{outline:2px solid #6cf}
.card{border:1px solid #333;border-radius:10px;padding:16px;margin:16px 0;background:#161923}
.card.done{border-color:#2c5}
.lbl{font-size:11px;color:#888;margin-bottom:4px;letter-spacing:.05em}
.txt{font-size:20px;line-height:1.9;white-space:pre-wrap}
.txtA{border-bottom:1px dashed #333;padding-bottom:12px;margin-bottom:12px}
.actions{display:flex;gap:8px;margin-top:12px}
.actions button{flex:1;padding:10px;font-size:15px}
.btn-same{border-color:#2c5}
.btn-same.active{background:#173}
.btn-diff{border-color:#c33}
.btn-diff.active{background:#711}
.btn-unsure{border-color:#c93}
.btn-unsure.active{background:#750}
#pidno{font-weight:bold}
.hint{font-size:11px;color:#777;margin-top:8px}
#navbtns{margin-inline-start:auto;display:flex;gap:6px}
</style></head><body>
<h1>Hadith pair reannotation — same matn or different matn?</h1>
<div class="hint">Judge whether A and B are the SAME underlying hadith matn (allowing for wording/diacritic
variation between books/narrations) or DIFFERENT hadiths. Keys: <b>S</b>=same, <b>D</b>=different,
<b>U</b>=unsure, <b>&larr;/&rarr;</b>=prev/next. Blind — no metric scores or model labels shown.</div>
<div id="bar">
  <span id="progress">0 / __N__ labelled</span>
  <button id="bJump">Jump to next unlabelled</button>
  <button id="bExport">&#11015; Export pair_labels_human100.json</button>
  <button id="bClear">Clear all</button>
  <button id="bDiac">Diacritics: off</button>
  <div id="navbtns"></div>
</div>
<div id="cards"></div>
<script>
const PAIRS = __PAIRS__;
const LS = "pairAnnotator100_v1";
const LS_DIAC = "pairAnnotator100_showDiac";
let state = JSON.parse(localStorage.getItem(LS) || "{}");
let showDiac = localStorage.getItem(LS_DIAC) === "1";
const save = () => localStorage.setItem(LS, JSON.stringify(state));

// Arabic diacritics (tashkeel: fatha/damma/kasra/tanwin/shadda/sukun/quranic marks) + tatweel.
// Built from explicit \\uXXXX escapes (never literal combining-mark glyphs in source) so the
// range can't be silently re-ordered/mangled in transit -- see project memory on Arabic NFC
// reordering bugs. Ranges: U+0610-061A (honorific marks), U+0640 (tatweel),
// U+064B-0670 (fatha..superscript-alef), U+065F (wavy hamza below), U+06D6-06ED (Quranic signs).
// Base letters (U+0621-064A) are untouched.
const DIAC_RE = /[\\u0610-\\u061A\\u0640\\u064B-\\u0670\\u065F\\u06D6-\\u06ED]/g;
function stripDiacritics(s){ return (s||"").replace(DIAC_RE, ""); }

function progressText(){
  const n = Object.keys(state).filter(k=>state[k]).length;
  return `${n} / ${PAIRS.length} labelled`;
}

function cardHTML(p, i){
  const cur = state[p.pid] || "";
  const btn = (val, cls, label) =>
    `<button data-pid="${p.pid}" data-val="${val}" class="${cls} act ${cur===val?'active':''}">${label}</button>`;
  const dispA = showDiac ? p.A : stripDiacritics(p.A);
  const dispB = showDiac ? p.B : stripDiacritics(p.B);
  return `<div class="card ${cur?'done':''}" id="c_${p.pid}">
    <div class="lbl">#${i+1} &middot; ${p.pid} &middot; <span id="pidno">A</span></div>
    <div class="txt txtA">${escapeHTML(dispA)}</div>
    <div class="lbl">B</div>
    <div class="txt">${escapeHTML(dispB)}</div>
    <div class="actions">
      ${btn("same","btn-same","نفس المتن (same)")}
      ${btn("different","btn-diff","متن مختلف (different)")}
      ${btn("unsure","btn-unsure","غير متأكد (unsure)")}
    </div>
  </div>`;
}
function escapeHTML(s){return (s||"").replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}

function render(){
  document.getElementById("cards").innerHTML = PAIRS.map(cardHTML).join("");
  document.getElementById("progress").textContent = progressText();
  document.getElementById("bDiac").textContent = "Diacritics: " + (showDiac ? "on" : "off");
}
render();

document.getElementById("bDiac").addEventListener("click", ()=>{
  showDiac = !showDiac;
  localStorage.setItem(LS_DIAC, showDiac ? "1" : "0");
  render();
});

document.getElementById("cards").addEventListener("click", e=>{
  const b = e.target.closest("button.act");
  if(!b) return;
  const pid = b.dataset.pid, val = b.dataset.val;
  state[pid] = (state[pid] === val) ? "" : val;  // click again to unset
  save(); render();
  document.getElementById("c_"+pid).scrollIntoView({block:"center"});
});

document.getElementById("bJump").addEventListener("click", ()=>{
  const next = PAIRS.find(p=>!state[p.pid]);
  if(next) document.getElementById("c_"+next.pid).scrollIntoView({block:"start"});
  else alert("All pairs labelled!");
});
document.getElementById("bClear").addEventListener("click", ()=>{
  if(confirm("Clear all labels?")){ state={}; save(); render(); }
});
document.getElementById("bExport").addEventListener("click", ()=>{
  const out = PAIRS.filter(p=>state[p.pid]).map(p=>({pid:p.pid, label:state[p.pid]}));
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "pair_labels_human100.json"; a.click();
});

// keyboard: focus nearest card to viewport center, apply S/D/U, arrow to move
function centerCard(){
  const cards=[...document.querySelectorAll(".card")];
  const mid=window.innerHeight/2;
  let best=null,bd=Infinity;
  for(const c of cards){const r=c.getBoundingClientRect();const d=Math.abs(r.top+r.height/2-mid);
    if(d<bd){bd=d;best=c;}}
  return best;
}
document.addEventListener("keydown", e=>{
  if(["INPUT","TEXTAREA"].includes(document.activeElement.tagName)) return;
  const card = centerCard(); if(!card) return;
  const pid = card.id.slice(2);
  const map = {s:"same", d:"different", u:"unsure"};
  const k = e.key.toLowerCase();
  if(map[k]){ state[pid] = (state[pid]===map[k])?"":map[k]; save(); render();
    document.getElementById("c_"+pid).scrollIntoView({block:"center"}); e.preventDefault(); }
  else if(e.key==="ArrowDown"||e.key==="ArrowRight"){
    const idx = PAIRS.findIndex(p=>p.pid===pid);
    if(idx>=0 && idx<PAIRS.length-1) document.getElementById("c_"+PAIRS[idx+1].pid).scrollIntoView({block:"center"});
  } else if(e.key==="ArrowUp"||e.key==="ArrowLeft"){
    const idx = PAIRS.findIndex(p=>p.pid===pid);
    if(idx>0) document.getElementById("c_"+PAIRS[idx-1].pid).scrollIntoView({block:"center"});
  }
});
</script>
</body></html>
"""

html = HTML.replace("__N__", str(len(pairs))).replace("__PAIRS__", json.dumps(pairs, ensure_ascii=False))
out_path = os.path.join(HERE, "pair_annotator_100.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path, f"({len(pairs)} pairs)")
