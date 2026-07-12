#!/usr/bin/env python3
"""
build_gold_tool.py — bake the manual gold-builder HTML.

Injects gold_pool_100.json (candidate pool + display block) + a slim Quran corpus
(verse search) into a self-contained RTL tool styled like correction_results_viewer.html:
each span shows the two model answers (GPT / Gemini) with every annotator's verdict +
manual corrections, and UNDER them the gold-set builder (candidate pool with counts).
The curator confirms/merges/drops, sets خطأ, adds missed, exports gold_100_final.json.

  python build_gold_tool.py   ->  gold_builder.html
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
POOL = json.load(open(os.path.join(HERE, "..", "gold_pool_100.json"), encoding="utf-8"))
qraw = json.load(open(os.path.join(ROOT, "data", "corpora", "quranic_verses.json"), encoding="utf-8"))
QURAN = [[v["surah_name"], v["ayah_id"], v["ayah_text"]] for v in qraw]

TEMPLATE = r"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>بناء المجموعة الذهبية — التصحيح</title>
<style>
:root{--g:#15803d;--r:#b91c1c;--n:#6b7280;--acc:#2563eb;--maj:#7c3aed;--gold:#a16207}
*{box-sizing:border-box}
body{font-family:'Segoe UI',Tahoma,Arial,sans-serif;background:#f1f5f9;margin:0;color:#0f172a;line-height:1.7}
header{position:sticky;top:0;background:#0f172a;color:#fff;padding:12px 18px;z-index:5;box-shadow:0 2px 8px #0003}
header h1{margin:0 0 8px;font-size:1.05rem}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:.85rem}
.bar select{padding:5px 8px;border-radius:6px;border:1px solid #475569;background:#1e293b;color:#fff}
.btn{background:var(--acc);color:#fff;border:0;padding:6px 12px;border-radius:7px;cursor:pointer;font-size:.85rem}
.btn.alt{background:#475569}.btn.gold{background:var(--gold)}
.pill{background:#1e293b;padding:3px 9px;border-radius:999px}
.wrap{max-width:1040px;margin:14px auto;padding:0 14px}
.card{background:#fff;border-radius:12px;padding:16px 18px;margin:0 0 16px;box-shadow:0 1px 4px #0001;border-right:5px solid #cbd5e1;scroll-margin-top:120px}
.card.reviewed{border-right-color:var(--g)}
.meta{display:flex;flex-wrap:wrap;gap:7px;font-size:.74rem;margin-bottom:8px;align-items:center}
.tag{background:#e2e8f0;border-radius:6px;padding:2px 8px;color:#334155}
.tag.q{background:#dbeafe}.tag.h{background:#fef3c7}
.lbl{color:#64748b;font-size:.78rem;margin:10px 0 3px;font-weight:700}
.q{font-weight:600}
.ans{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px;white-space:pre-wrap;font-size:.9rem;max-height:230px;overflow:auto}
mark{background:#fde68a;padding:0 2px;border-radius:3px}
.slots{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
@media(max-width:680px){.slots{grid-template-columns:1fr}}
.slot{border:1px solid #e2e8f0;border-radius:8px;padding:10px}
.slot.first{background:#eff6ff}.slot.second{background:#fffbeb}
.corr{font-size:.86rem;margin-top:4px}.src{color:var(--acc);font-size:.78rem}
.chip{display:inline-block;font-size:.72rem;font-weight:700;color:#fff;border-radius:6px;padding:2px 7px;margin:3px 3px 0 0}
.chip.correct{background:var(--g)}.chip.wrong{background:var(--r)}.chip.none{background:var(--n)}
.chip.who{background:#e2e8f0;color:#334155;font-weight:600}
.manual{background:#ecfdf5;border:1px solid #6ee7b7;border-radius:6px;padding:6px;margin-top:5px;font-size:.83rem;white-space:pre-wrap}
/* ---- gold builder section ---- */
.gold{margin-top:14px;border:2px solid #fde68a;background:#fffdf5;border-radius:10px;padding:12px}
.gold h3{margin:0 0 8px;font-size:.92rem;color:#92400e;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.cand{display:flex;gap:9px;align-items:flex-start;padding:8px 10px;border:1px solid #e2e8f0;border-radius:8px;margin:6px 0;background:#fff}
.cand.ung{border-color:#f59e0b;background:#fff7ed}
.cand.khata{border-color:#c4b5fd;background:#f5f3ff}
.cand.off{opacity:.45}
.cand.merging{box-shadow:0 0 0 2px #f59e0b}
.cand .body{flex:1;min-width:0}
.cand .txt{outline:none;font-size:.88rem}
.cand .txt[contenteditable]:focus{background:#f8fafc;box-shadow:0 0 0 1px var(--acc);border-radius:4px}
.gmeta{font-size:.72rem;color:#64748b;margin-top:5px;display:flex;gap:7px;flex-wrap:wrap;align-items:center}
.cnt{font-weight:800;color:#fff;background:#334155;border-radius:6px;padding:1px 8px;font-size:.78rem}
.badge.maj{background:var(--maj);color:#fff;border-radius:999px;padding:1px 8px;font-size:.7rem;font-weight:700}
.ra{display:flex;gap:4px;flex-shrink:0}
.mini{font-size:.72rem;padding:3px 8px;border:1px solid #cbd5e1;background:#fff;border-radius:6px;cursor:pointer}
.mini:hover{border-color:var(--acc)}.mini.p{background:var(--acc);color:#fff;border-color:var(--acc)}
.add{margin-top:8px;padding:9px;border:1px dashed #cbd5e1;border-radius:8px}
.add input{width:100%;padding:6px 9px;border:1px solid #cbd5e1;border-radius:6px;font-family:inherit;font-size:.85rem;margin-bottom:6px}
.sugg{max-height:170px;overflow:auto;border:1px solid #e2e8f0;border-radius:6px}
.sugg div{padding:5px 8px;cursor:pointer;border-bottom:1px solid #eef2f6;font-size:.82rem}
.sugg div:hover{background:#eff6ff}
input[type=checkbox]{width:17px;height:17px;accent-color:var(--g);cursor:pointer;margin-top:2px}
.hidden{display:none}
</style></head><body>
<header>
  <h1>بناء المجموعة الذهبية للتصحيح — 100 نطاق</h1>
  <div class="bar">
    <span class="pill" id="prog"></span>
    <select id="filter">
      <option value="all">كل البطاقات</option>
      <option value="unrev">غير المراجَعة</option>
      <option value="ung">تحوي مرشحاً غير مُوثَّق</option>
      <option value="quran">قرآن</option><option value="hadith">حديث</option>
    </select>
    <button class="btn alt" id="bNext">التالي غير مراجَع ↩</button>
    <button class="btn gold" id="bExport">⬇ تصدير gold_100_final.json</button>
    <button class="btn alt" id="bClear">مسح</button>
  </div>
</header>
<div class="wrap" id="root"></div>
<script>
const POOL=__POOL__, QURAN=__QURAN__, LS="goldBuilder_v3", ANN=POOL.annotators;
const esc=s=>(s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function nrm(s){return (s||"").normalize("NFC").replace(/[ؐ-ًؚ-ٰٟۖ-ۭ]/g,"")
  .replace(/[آأإٱ]/g,"ا").replace(/ى/g,"ي").replace(/ة/g,"ه").replace(/\s+/g," ").trim();}
const QN=QURAN.map(q=>nrm(q[2]));

let state=JSON.parse(localStorage.getItem(LS)||"{}");
const save=()=>localStorage.setItem(LS,JSON.stringify(state));
function initSpan(sp){
  if(state[sp.uid]) return state[sp.uid];
  state[sp.uid]={candidates:sp.candidates.map(c=>({text:c.text,sources:c.sources.slice(),grounded:c.grounded,
      provenance:c.provenance.slice(),backers:c.backers.slice(),included:c.grounded})),
    khata:{count:sp.khata.count,backers:sp.khata.backers.slice(),included:sp.khata.count>=1},
    added:[],reviewed:false};
  return state[sp.uid];
}
let mergeSrc=null;

function render(){
  const f=document.getElementById("filter").value, root=document.getElementById("root"); root.innerHTML="";
  let rev=0;
  POOL.spans.forEach(sp=>{const st=initSpan(sp); if(st.reviewed)rev++;
    if(f==="unrev"&&st.reviewed)return; if(f==="quran"&&sp.ref!=="quran")return;
    if(f==="hadith"&&sp.ref!=="hadith")return; if(f==="ung"&&!st.candidates.some(c=>!c.grounded))return;
    root.appendChild(cardEl(sp,st));});
  document.getElementById("prog").textContent=`مُراجَع ${rev}/${POOL.spans.length}`; save();
}
function ansHTML(sp){const a=sp.answer||"",d=sp.display||{};
  if(d.s==null||d.e==null||d.e<=d.s) return esc(a);
  return esc(a.slice(0,d.s))+"<mark>"+esc(a.slice(d.s,d.e))+"</mark>"+esc(a.slice(d.e));}
function slotHTML(sp,which){
  const d=(sp.display||{})[which]||{annotators:{}};
  const label=which==="first"?"إجابة GPT‑5.4":"إجابة Gemini";
  const body=d.model_text
    ? d.model_text.split("|||").map(t=>`<div class="corr">• ${esc(t.trim())}</div>`).join("")
    : `<div class="corr" style="color:#94a3b8">النموذج امتنع (خطأ)</div>`;
  const juds=Object.entries(d.annotators||{}).map(([who,o])=>{
    const cls=o.judgment==="correct"?"correct":o.judgment==="wrong"?"wrong":"none";
    return `<span class="chip ${cls}">${esc(who)}: ${o.judgment==="correct"?"صحيح":o.judgment==="wrong"?"خطأ":"—"}</span>`;}).join("");
  const man=Object.entries(d.annotators||{}).filter(([_,o])=>o.manual&&o.manual.length)
    .map(([who,o])=>`<div class="manual"><b>تصحيح ${esc(who)}:</b><br>${o.manual.map(m=>m.split("|||").map(x=>"• "+esc(x.trim())).join("<br>")).join("<br>")}</div>`).join("");
  return `<div class="slot ${which}"><b style="font-size:.82rem">${label}</b>
    <span style="font-size:.7rem;color:#64748b">(حكم النموذج: ${esc(d.model_verdict||"—")})</span>
    ${body}<div class="lbl" style="margin:8px 0 2px">أحكام المُقيّمين</div>${juds||'<span class="chip none">—</span>'}${man}</div>`;
}
function cardEl(sp,st){
  const d=document.createElement("div"); d.className="card"+(st.reviewed?" reviewed":""); d.id="c_"+sp.uid;
  d.innerHTML=`<div class="meta"><span class="tag ${sp.ref==='quran'?'q':'h'}">${sp.ref}</span>
      <span class="tag">${sp.id}</span><span class="tag">${esc(sp.model||"")}</span>
      <label style="margin-inline-start:auto;cursor:pointer;font-size:.8rem"><input type=checkbox ${st.reviewed?"checked":""}
        data-act="rev" data-id="${sp.uid}"> تمّت المراجعة</label></div>
    <div class="lbl">النص المُقتبَس (خطأ)</div><div class="ans" style="max-height:none;color:#b91c1c">${esc(sp.span)}</div>
    ${sp.question?`<div class="lbl">السؤال</div><div class="q">${esc(sp.question)}</div>`:""}
    ${sp.answer?`<div class="lbl">الإجابة (النطاق مظلَّل)</div><div class="ans">${ansHTML(sp)}</div>`:""}
    <div class="slots">${slotHTML(sp,"first")}${slotHTML(sp,"second")}</div>
    <div class="gold" id="g_${sp.uid}"></div>`;
  buildGold(d.querySelector("#g_"+sp.uid),sp,st);
  return d;
}
function buildGold(box,sp,st){
  box.innerHTML=`<h3>🏆 المجموعة الذهبية <span style="font-size:.72rem;color:#64748b;font-weight:400">— اختر كل التصحيحات المقبولة (المُوثَّقة مُحدَّدة مسبقاً؛ راجِع غير المُوثَّقة)</span></h3>`;
  st.candidates.forEach((c,i)=>box.appendChild(candEl(sp,st,c,i)));
  box.appendChild(khataEl(sp,st));
  st.added.forEach((c,i)=>box.appendChild(addedEl(sp,i,c)));
  box.appendChild(addBox(sp));
}
function candEl(sp,st,c,i){
  const maj=c.backers.length>=2, e=document.createElement("div");
  e.className="cand"+(c.grounded?"":" ung")+(c.included?"":" off")+((mergeSrc&&mergeSrc.id===sp.uid&&mergeSrc.idx===i)?" merging":"");
  e.innerHTML=`<input type=checkbox ${c.included?"checked":""} data-act="tog" data-id="${sp.uid}" data-i="${i}">
    <div class="body"><div class="txt" contenteditable data-act="edit" data-id="${sp.uid}" data-i="${i}">${esc(c.text)}</div>
      <div class="gmeta"><span class="cnt">${c.backers.length}</span>${maj?'<span class="badge maj">أغلبية</span>':''}
        ${c.grounded?`<span class="src">${(c.sources||[]).join(" · ")||"مُوثَّق"}</span>`:'<span style="color:#b45309;font-weight:700">غير مُوثَّق — راجِع</span>'}
        ${c.provenance.map(p=>`<span class="chip who">${p}</span>`).join("")}
        ${c.backers.map(b=>`<span class="chip who">${esc(b)}</span>`).join("")}</div></div>
    <div class="ra"><button class="mini" data-act="merge" data-id="${sp.uid}" data-i="${i}">دمج</button>
      <button class="mini" data-act="del" data-id="${sp.uid}" data-i="${i}">حذف</button></div>`;
  return e;
}
function khataEl(sp,st){const maj=st.khata.count>=2,e=document.createElement("div");
  e.className="cand khata"+(st.khata.included?"":" off");
  e.innerHTML=`<input type=checkbox ${st.khata.included?"checked":""} data-act="togk" data-id="${sp.uid}">
    <div class="body"><div class="txt"><b>خطأ</b> — لا تصحيح ممكن</div>
      <div class="gmeta"><span class="cnt">${st.khata.count}</span>${maj?'<span class="badge maj">أغلبية</span>':''}
        ${st.khata.backers.map(b=>`<span class="chip who">${esc(b)}</span>`).join("")}
        <button class="mini" data-act="bumpk" data-id="${sp.uid}" data-dz="1">+</button>
        <button class="mini" data-act="bumpk" data-id="${sp.uid}" data-dz="-1">−</button></div></div>`;
  return e;
}
function addedEl(sp,i,c){
  const e=document.createElement("div"); e.className="cand"+(c.is_khata?" khata":"")+(c.included===false?" off":"");
  const cb=document.createElement("input"); cb.type="checkbox"; cb.checked=c.included!==false;
  cb.onchange=()=>{c.included=cb.checked; save(); refresh(sp.uid);};
  const body=document.createElement("div"); body.className="body";
  body.innerHTML=`<div class="txt">${esc(c.text)}</div><div class="gmeta"><span class="chip who">مُضاف</span>`+
    `${c.is_khata?'<span class="badge maj">خطأ</span>':''}${c.sources&&c.sources.length?`<span class="src">${esc(c.sources.join(" · "))}</span>`:''}</div>`;
  const ra=document.createElement("div"); ra.className="ra";
  const del=document.createElement("button"); del.className="mini"; del.textContent="حذف";
  del.onclick=()=>{state[sp.uid].added.splice(i,1); save(); refresh(sp.uid);};
  ra.appendChild(del); e.append(cb,body,ra); return e;
}
// element-based add box: direct listeners, no getElementById / no inline onclick
function addBox(sp){
  const e=document.createElement("div"); e.className="add"; const qs=sp.ref==="quran";
  const inp=document.createElement("input"); inp.type="text";
  inp.placeholder=qs?"ابحث عن آية أو اكتب النص…":"الصق نص الحديث/التصحيح…";
  const sugg=document.createElement("div"); sugg.className="sugg";
  if(qs) inp.addEventListener("input",()=>quranSearchEl(inp,sugg));
  const bs=document.createElement("div"); bs.style.cssText="display:flex;gap:6px;margin-top:6px";
  const b1=document.createElement("button"); b1.type="button"; b1.className="mini p"; b1.textContent="+ إضافة كتصحيح";
  b1.addEventListener("click",()=>addCandEl(sp,inp,false));
  const b2=document.createElement("button"); b2.type="button"; b2.className="mini"; b2.textContent="+ إضافة كخطأ";
  b2.addEventListener("click",()=>addCandEl(sp,inp,true));
  bs.append(b1,b2); e.append(inp,sugg,bs); return e;
}
function addCandEl(sp,inp,isK){
  const t=(inp.value||"").trim();
  if(!t){ inp.style.boxShadow="0 0 0 2px #ef4444"; setTimeout(()=>{inp.style.boxShadow="";},800); return; }
  (state[sp.uid].added||(state[sp.uid].added=[])).push({text:t,is_khata:isK,included:true,sources:inp.dataset.src?[inp.dataset.src]:[]});
  save(); refresh(sp.uid);
}
function quranSearchEl(inp,box){
  const q=nrm(inp.value); if(q.length<4){box.innerHTML="";return;}
  const hits=[]; for(let i=0;i<QN.length&&hits.length<12;i++){if(QN[i].includes(q))hits.push(i);}
  box.innerHTML="";
  hits.forEach(i=>{const d=document.createElement("div");
    d.innerHTML=`${esc(QURAN[i][2])} <span class="src">[سورة ${QURAN[i][0]} ${QURAN[i][1]}]</span>`;
    d.addEventListener("click",()=>{inp.value=QURAN[i][2]; inp.dataset.src=`سورة ${QURAN[i][0]} ${QURAN[i][1]}`; box.innerHTML="";});
    box.appendChild(d);});
}
// actions
function refresh(id){const sp=POOL.spans.find(s=>s.uid===id);
  const old=document.getElementById("c_"+id);
  if(old){ old.replaceWith(cardEl(sp,state[id])); }
  else { render(); return; }                 // card not in DOM -> full re-render (never silently no-op)
  const p=document.getElementById("prog");
  if(p) p.textContent=`مُراجَع ${POOL.spans.filter(s=>state[s.uid]&&state[s.uid].reviewed).length}/${POOL.spans.length}`;
  save();}
function tog(id,i,v){state[id].candidates[i].included=v; save(); refresh(id);}
function togKhata(id,v){state[id].khata.included=v; save(); refresh(id);}
function bumpKhata(id,dz){const k=state[id].khata; k.count=Math.max(0,k.count+dz); if(k.count>=1)k.included=true; save(); refresh(id);}
function edit(id,i,t){state[id].candidates[i].text=t.trim(); save();}
function del(id,i){state[id].candidates.splice(i,1); save(); refresh(id);}
function delAdded(id,i){state[id].added.splice(i,1); save(); refresh(id);}
function toggleRev(id,v){state[id].reviewed=v; save(); render();}
function startMerge(id,i){
  if(mergeSrc&&mergeSrc.id===id&&mergeSrc.idx!==i){
    const a=state[id].candidates[mergeSrc.idx], b=state[id].candidates[i];
    b.backers=[...new Set([...b.backers,...a.backers])]; b.provenance=[...new Set([...b.provenance,...a.provenance])];
    b.sources=[...new Set([...b.sources,...a.sources])]; b.grounded=b.grounded||a.grounded;
    state[id].candidates.splice(mergeSrc.idx,1); mergeSrc=null; save(); refresh(id);
  } else {mergeSrc={id,idx:i}; refresh(id);}
}
function jumpNext(){const nx=POOL.spans.find(s=>!(state[s.uid]&&state[s.uid].reviewed));
  if(nx)document.getElementById("c_"+nx.uid)?.scrollIntoView({behavior:"smooth"}); else alert("كل البطاقات مُراجَعة ✔");}
function exportGold(){
  const out=POOL.spans.map(sp=>{const st=initSpan(sp), gold=[];
    st.candidates.forEach(c=>{if(c.included)gold.push({text:c.text,sources:c.sources,count:c.backers.length,backers:c.backers,is_khata:false,provenance:c.provenance,grounded:c.grounded});});
    if(st.khata.included)gold.push({text:"خطأ",sources:[],count:st.khata.count,backers:st.khata.backers,is_khata:true,provenance:["annotator"]});
    st.added.forEach(a=>{if(a.included!==false)gold.push({text:a.text,sources:a.sources,count:0,backers:[],is_khata:a.is_khata,provenance:["curator"]});});
    return {uid:sp.uid,id:sp.id,ref:sp.ref,model:sp.model,span:sp.span,gold,reviewed:st.reviewed};});
  const blob=new Blob([JSON.stringify({n:out.length,annotators:ANN,generated:new Date().toISOString(),spans:out},null,2)],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="gold_100_final.json"; a.click();}
// ---- event delegation: one set of listeners on #root, no inline handlers ----
const ROOT=document.getElementById("root");
ROOT.addEventListener("click",ev=>{const el=ev.target.closest("[data-act]"); if(!el)return;
  const a=el.dataset.act,id=el.dataset.id,i=+el.dataset.i;
  if(a==="merge")startMerge(id,i);
  else if(a==="del")del(id,i);
  else if(a==="bumpk")bumpKhata(id,+el.dataset.dz);});
ROOT.addEventListener("change",ev=>{const el=ev.target.closest("[data-act]"); if(!el)return;
  const a=el.dataset.act,id=el.dataset.id,i=+el.dataset.i;
  if(a==="tog")tog(id,i,el.checked);
  else if(a==="togk")togKhata(id,el.checked);
  else if(a==="rev")toggleRev(id,el.checked);});
ROOT.addEventListener("focusout",ev=>{const el=ev.target.closest('[data-act="edit"]'); if(!el)return;
  edit(el.dataset.id,+el.dataset.i,el.innerText);});
document.getElementById("filter").addEventListener("change",render);
document.getElementById("bNext").addEventListener("click",jumpNext);
document.getElementById("bExport").addEventListener("click",exportGold);
document.getElementById("bClear").addEventListener("click",()=>{if(confirm("مسح كل التعديلات؟")){localStorage.removeItem(LS);location.reload();}});
render();
</script></body></html>"""

html = TEMPLATE.replace("__POOL__", json.dumps(POOL, ensure_ascii=False))
html = html.replace("__QURAN__", json.dumps(QURAN, ensure_ascii=False))
outp = os.path.join(HERE, "gold_builder.html")
open(outp, "w", encoding="utf-8").write(html)
print(f"wrote {os.path.relpath(outp, HERE)}  ({len(html)//1024} KB) — {POOL['n']} spans, {len(QURAN)} ayahs baked")
