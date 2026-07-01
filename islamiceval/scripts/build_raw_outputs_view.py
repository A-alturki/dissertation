#!/usr/bin/env python3
"""
build_raw_outputs_view.py
=========================
Researcher-facing viewer of the RAW one-shot model outputs (gpt-5.4 + gemini) for
every span in the correction set — verdict, rationale, and the raw candidates
(text / source / matched_run / confidence), BEFORE grounding/snapping.

Joins correction_annotation.json (question/answer/span/offsets) with the raw
predictions in pipeline_oneshot.json. Emits a self-contained HTML (data embedded).
Unlike the annotator tool, this one DOES show the model names — it's for you, to
sanity-check what the models actually said.

Usage:
    python build_raw_outputs_view.py
    -> ../tools/raw_model_outputs.html
"""
import os, sys, json
HERE=os.path.dirname(os.path.abspath(__file__))
ANNOT=os.path.join(HERE,"..","outputs","correction","correction_annotation.json")
# all correction batches whose spans can appear in the final set: original 100,
# the extension (topup1), and the batch2 (minor/no_match JSONL) top-up
ONESHOTS=[os.path.join(HERE,"..","outputs","correction","pipeline_oneshot.json"),
          os.path.join(HERE,"..","outputs","correction","pipeline_oneshot_topup1.json"),
          os.path.join(HERE,"..","outputs","correction","pipeline_oneshot_batch2.json")]
OUT=os.path.join(HERE,"..","tools","raw_model_outputs.html")
GPT,GEM="gpt-5.4","gemini-3.1-pro-preview"
try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
except Exception: pass

def raw(block,model):
    p=((block or {}).get(model) or {}).get("prediction") or {}
    return {"verdict":p.get("verdict","?"),"rationale":p.get("rationale",""),
            "candidates":[{"text":c.get("text",""),"source":c.get("source",""),
                           "matched_run":c.get("matched_run",""),"confidence":c.get("confidence","")}
                          for c in (p.get("candidates") or [])]}

def main():
    ann=json.load(open(ANNOT,encoding="utf-8"))["items"]
    one={}
    for path in ONESHOTS:
        if not os.path.exists(path):
            print(f"  WARNING: oneshot not found, skipping: {path}"); continue
        for s in json.load(open(path,encoding="utf-8"))["per_sample"]:
            one[(s["id"],s["answer_model"],s["span"])]=s
    recs=[]
    for it in ann:
        block=(one.get((it["id"],it["model"],it["span"])) or {}).get("oneshot",{})
        recs.append({"id":it["id"],"ref":it["ref"],"model":it["model"],
                     "question":it["question"],"answer":it["answer"],"span":it["span"],
                     "span_start":it["span_start"],"span_end":it["span_end"],
                     "gpt":raw(block,GPT),"gemini":raw(block,GEM)})
    data={"first_model":GPT,"second_model":GEM,"n":len(recs),"items":recs}
    html=TEMPLATE.replace("/*__DATA__*/", json.dumps(data,ensure_ascii=False).replace("<","\\u003c"))
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    open(OUT,"w",encoding="utf-8").write(html)
    print(f"wrote {len(recs)} spans -> {OUT}")

TEMPLATE=r"""<!DOCTYPE html>
<html lang="ar" dir="rtl"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>المخرجات الخام للنماذج</title>
<style>
 :root{--bg:#0f172a;--card:#1e293b;--ink:#e2e8f0;--muted:#94a3b8;--line:#334155;
   --accent:#38bdf8;--green:#34d399;--red:#f87171;--mark:#fbbf24;--mark-ink:#422006}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI","Noto Naskh Arabic",Tahoma,sans-serif;line-height:1.9}
 header{background:#0b1220;padding:18px 20px;text-align:center;border-bottom:1px solid var(--line)}
 header h1{margin:0;font-size:1.4rem;color:#f1f5f9}
 header p{margin:4px 0 0;color:var(--muted);font-size:.85rem}
 .bar{position:sticky;top:0;background:#0b1220;border-bottom:1px solid var(--line);padding:10px 18px;
   display:flex;gap:8px;flex-wrap:wrap;align-items:center;z-index:5}
 .bar input{flex:1;min-width:180px;padding:9px 13px;border:1px solid var(--line);border-radius:9px;background:#0f172a;color:var(--ink);font-family:inherit;font-size:1rem}
 .chip{padding:7px 13px;border:1px solid var(--line);border-radius:18px;background:#0f172a;color:var(--muted);cursor:pointer;font-size:.85rem;font-family:inherit}
 .chip.on{background:var(--accent);color:#04222e;border-color:var(--accent);font-weight:700}
 .count{margin-inline-start:auto;color:var(--muted);font-size:.85rem}
 .wrap{max-width:1000px;margin:20px auto;padding:0 14px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:16px;overflow:hidden}
 .chead{display:flex;align-items:center;gap:9px;padding:11px 16px;background:#172033;border-bottom:1px solid var(--line)}
 .num{background:var(--accent);color:#04222e;width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-weight:700;font-size:.85rem}
 .badge{padding:3px 11px;border-radius:13px;font-size:.8rem;font-weight:700}
 .b-h{background:#0c3b32;color:var(--green)} .b-q{background:#0c2b4a;color:#7cc4fb}
 .mtag{color:var(--muted);font-size:.82rem}
 .cbody{padding:8px 16px 14px}
 .lab{font-size:.72rem;color:var(--accent);font-weight:700;margin:9px 0 3px}
 .q{font-size:1rem}
 .ans{font-size:.98rem;line-height:1.9;max-height:160px;overflow:auto;background:#0f172a;border:1px solid var(--line);border-radius:9px;padding:10px 12px;white-space:pre-wrap}
 .ans mark{background:var(--mark);color:var(--mark-ink);padding:0 2px;border-radius:3px;font-weight:600}
 .span{font-size:1.05rem;font-weight:600;background:#0f172a;border-right:3px solid var(--accent);padding:7px 11px;border-radius:6px}
 .cols{display:flex;gap:12px;margin-top:8px}
 .col{flex:1;min-width:0;background:#0f172a;border:1px solid var(--line);border-radius:10px;padding:11px}
 .chd{display:flex;align-items:center;gap:8px;margin-bottom:6px}
 .mname{font-weight:700;font-size:.9rem}
 .v-ok{padding:2px 9px;border-radius:11px;font-size:.74rem;font-weight:700;background:#0c3b32;color:var(--green)}
 .v-no{padding:2px 9px;border-radius:11px;font-size:.74rem;font-weight:700;background:#3b1414;color:var(--red)}
 .rat{font-size:.9rem;color:var(--muted);font-style:italic;border-inline-start:2px solid var(--line);padding-inline-start:9px;margin:5px 0}
 .cand{padding:7px 0;border-top:1px dashed var(--line)}
 .ctext{font-size:1rem} .cmeta{font-size:.76rem;color:var(--accent);margin-top:2px}
 .crun{font-size:.78rem;color:var(--muted)}
 .none{color:#64748b;font-style:italic;font-size:.9rem}
 @media(max-width:720px){.cols{flex-direction:column}}
</style></head>
<body>
<header><h1>المخرجات الخام للنماذج</h1><p id="sub"></p></header>
<div class="bar">
 <input id="q" type="text" placeholder="🔍 بحث (سؤال / استشهاد / تعليل / معرّف)...">
 <button class="chip on" data-f="all">الكل</button>
 <button class="chip" data-f="quran">آية</button>
 <button class="chip" data-f="hadith">حديث</button>
 <button class="chip" data-f="bothok">كلاهما صحّح</button>
 <button class="chip" data-f="bothno">كلاهما خطأ</button>
 <button class="chip" data-f="disagree">اختلفا</button>
 <span class="count" id="count"></span>
</div>
<div class="wrap" id="list"></div>
<script id="d" type="application/json">/*__DATA__*/</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("d").textContent);
const esc=s=>(s==null?"":String(s)).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
let filt="all",qy="";
document.getElementById("sub").textContent=`${D.n} استشهاد · المصحّح الأول = ${D.first_model} · المصحّح الثاني = ${D.second_model}`;
function hl(it){const a=it.answer||"";let s=it.span_start,e=it.span_end;
 if(!(s>=0&&e>s)){const i=a.indexOf(it.span||"");if(i>=0){s=i;e=i+it.span.length;}}
 return s>=0&&e>s? esc(a.slice(0,s))+"<mark>"+esc(a.slice(s,e))+"</mark>"+esc(a.slice(e)) : esc(a);}
function col(name,r){
 const ok=r.verdict==="ok";
 const v=ok?'<span class="v-ok">صحّح</span>':'<span class="v-no">خطأ</span>';
 let cands=r.candidates&&r.candidates.length? r.candidates.map(c=>
   '<div class="cand"><div class="ctext">'+esc(c.text)+'</div>'+
   '<div class="cmeta">'+esc(c.source)+' · ثقة: '+esc(c.confidence)+'</div>'+
   (c.matched_run?'<div class="crun">المطابق: '+esc(c.matched_run)+'</div>':'')+'</div>').join("")
   : '<div class="none">لا مرشّحات</div>';
 return '<div class="col"><div class="chd"><span class="mname">'+esc(name)+'</span>'+v+'</div>'+
   (r.rationale?'<div class="rat">'+esc(r.rationale)+'</div>':'')+cands+'</div>';
}
function card(it,i){
 const isH=it.ref==="hadith";
 return '<div class="card"><div class="chead"><span class="num">'+(i+1)+'</span>'+
  '<span class="badge '+(isH?"b-h":"b-q")+'">'+(isH?"حديث":"آية")+'</span>'+
  '<span class="mtag">'+esc(it.model)+' · '+esc(it.id)+'</span></div>'+
  '<div class="cbody">'+
   '<div class="lab">السؤال</div><div class="q">'+esc(it.question)+'</div>'+
   '<div class="lab">الإجابة (الاستشهاد مظلَّل)</div><div class="ans">'+hl(it)+'</div>'+
   '<div class="lab">الاستشهاد</div><div class="span">'+esc(it.span)+'</div>'+
   '<div class="lab">المخرجات الخام</div>'+
   '<div class="cols">'+col(D.first_model,it.gpt)+col(D.second_model,it.gemini)+'</div>'+
  '</div></div>';
}
function show(){
 const L=document.getElementById("list");
 const items=D.items.filter(it=>{
  if(filt==="quran"&&it.ref!=="quran")return false;
  if(filt==="hadith"&&it.ref!=="hadith")return false;
  const a=it.gpt.verdict==="ok",b=it.gemini.verdict==="ok";
  if(filt==="bothok"&&!(a&&b))return false;
  if(filt==="bothno"&&!(!a&&!b))return false;
  if(filt==="disagree"&&a===b)return false;
  if(qy){const h=(it.question+" "+it.span+" "+it.id+" "+it.gpt.rationale+" "+it.gemini.rationale).toLowerCase();if(!h.includes(qy))return false;}
  return true;});
 L.innerHTML=items.map((it)=>card(it,D.items.indexOf(it))).join("");
 document.getElementById("count").textContent=items.length+" / "+D.n;
}
document.getElementById("q").addEventListener("input",e=>{qy=e.target.value.trim().toLowerCase();show();});
document.querySelectorAll(".chip").forEach(c=>c.addEventListener("click",()=>{
 document.querySelectorAll(".chip").forEach(x=>x.classList.remove("on"));c.classList.add("on");filt=c.dataset.f;show();}));
show();
</script></body></html>"""

if __name__=="__main__":
    main()
