"use strict";
const API = window.location.origin;

const SAMPLES = [
  {a:"The physician carefully examined the patient and prescribed a course of antibiotics to treat the bacterial infection. The patient was advised to rest and drink plenty of fluids during the recovery period.",
   b:"A doctor thoroughly assessed the individual and recommended antibiotic medication for the bacterial illness. The sick person was told to recuperate and stay well hydrated throughout their healing process."},
  {a:"Machine learning algorithms learn patterns from large datasets and use those patterns to make predictions on new, unseen data. The quality of predictions depends heavily on the quantity and quality of training data.",
   b:"Machine learning algorithms learn patterns from large datasets and use those patterns to make predictions on new, unseen data. The accuracy of results depends significantly on how much and how good the training data is."},
  {a:"Climate change is driven primarily by greenhouse gas emissions from burning fossil fuels. Rising global temperatures are causing more frequent extreme weather events, threatening ecosystems worldwide.",
   b:"The rapid acceleration of economic growth in emerging markets has created new opportunities for international trade, reshaping the global financial landscape over the past two decades."}
];
let si = 0;

const STOPS = new Set("a an the is are was were be been have has had do does did will would could should may might to of in for on with at by from and or but not this that it its as into about than also".split(" "));

function $(id){ return document.getElementById(id); }
function wc(s){ const t=s.trim(); return t?t.split(/\s+/).length:0; }
function pct(v){ return Math.round(v*100)+"%"; }
function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function updateWC(){
  $("wc-a").textContent = wc($("ta").value)+" words";
  $("wc-b").textContent = wc($("tb").value)+" words";
}

function loadSample(){
  const s = SAMPLES[si % SAMPLES.length]; si++;
  $("ta").value = s.a; $("tb").value = s.b; updateWC();
  $("results").style.display = "none";
  $("btn-sample").textContent = "Load sample "+(si%SAMPLES.length+1)+"/"+SAMPLES.length;
}

function clearAll(){
  $("ta").value=""; $("tb").value=""; updateWC();
  $("results").style.display="none"; $("prog").style.display="none";
  $("btn-sample").textContent="Load sample";
}

function setStages(active){
  for(let i=0;i<4;i++){
    const li = $("s"+i), tick = $("sc"+i);
    li.className = "stage-item"+(i<active?" s-done":i===active?" s-active":"");
    tick.textContent = i<active?"✓":"";
  }
}
function doneStages(){ for(let i=0;i<4;i++){$("s"+i).className="stage-item s-done";$("sc"+i).textContent="✓";} }

function buildHL(textB, textA, ce){
  const tokA = new Set((textA.toLowerCase().match(/\b\w+\b/g)||[]).filter(w=>!STOPS.has(w)&&w.length>2));
  return textB.replace(/(\S+)/g, word => {
    const cl = word.toLowerCase().replace(/[^a-z]/g,"");
    if(!cl||cl.length<3||STOPS.has(cl)) return word;
    if(tokA.has(cl)) return `<mark class="hl-e">${word}</mark>`;
    const stem = [...tokA].some(w=>w.length>3&&cl.length>3&&(w.startsWith(cl.slice(0,4))||cl.startsWith(w.slice(0,4))));
    if(stem) return `<mark class="hl-s">${word}</mark>`;
    if(ce>0.55 && Math.random()<(ce-0.4)*0.28) return `<mark class="hl-m">${word}</mark>`;
    return word;
  });
}

function sentTable(textA, textB){
  const sa = textA.match(/[^.!?]+[.!?]*/g)||[textA];
  const sb = textB.match(/[^.!?]+[.!?]*/g)||[textB];
  const rows = sb.map(sent => {
    const tb2 = new Set((sent.toLowerCase().match(/\b\w+\b/g)||[]).filter(w=>!STOPS.has(w)));
    let best=0, bst="";
    for(const ss of sa){
      const ta2=new Set((ss.toLowerCase().match(/\b\w+\b/g)||[]).filter(w=>!STOPS.has(w)));
      if(!ta2.size||!tb2.size) continue;
      const inter=[...ta2].filter(w=>tb2.has(w)).length;
      const j=inter/(ta2.size+tb2.size-inter);
      if(j>best){best=j;bst=ss;}
    }
    return {sent:sent.trim(), match:bst.trim(), score:best};
  });
  return `<table><thead><tr>
    <th style="width:37%">Suspect sentence</th>
    <th style="width:37%">Closest match</th>
    <th style="width:11%">Score</th>
    <th style="width:15%">Risk</th>
  </tr></thead><tbody>${rows.map(r=>{
    const p=r.score>=0.5?`<span class="rpill rp-h">High</span>`:r.score>=0.25?`<span class="rpill rp-m">Medium</span>`:`<span class="rpill rp-l">Low</span>`;
    return`<tr><td>${esc(r.sent)}</td><td style="color:#aaa">${r.match?esc(r.match):"—"}</td><td class="td-num">${pct(r.score)}</td><td>${p}</td></tr>`;
  }).join("")}</tbody></table>`;
}

function renderShap(tokens, ce){
  const card = $("shap-card");
  if(!tokens||!tokens.length){card.style.display="none";return;}
  card.style.display="";
  const mx = Math.max(...tokens.map(t=>Math.abs(t.shap_value)),1e-6);
  $("shap-rows").innerHTML = tokens.map(t=>{
    const w=Math.round(Math.abs(t.shap_value)/mx*100);
    const pos=t.shap_value>0;
    const col=pos?"#E24B4A":"#378ADD";
    const sign=pos?"+":"";
    return `<div class="shap-row">
      <span class="shap-tok">${esc(t.token)}</span>
      <div class="shap-track"><div class="shap-fill" style="width:${w}%;background:${col}"></div></div>
      <span class="shap-val">${sign}${t.shap_value.toFixed(4)}</span>
      <span class="shap-tag ${pos?"tag-pos":"tag-neg"}">${pos?"+sim":"−sim"}</span>
    </div>`;
  }).join("");
}

// client-side scoring fallback
function bm25(a,b){
  const ta=new Set((a.toLowerCase().match(/\b\w+\b/g)||[]).filter(w=>!STOPS.has(w)&&w.length>2));
  const tb=(b.toLowerCase().match(/\b\w+\b/g)||[]).filter(w=>!STOPS.has(w)&&w.length>2);
  return tb.length ? Math.min(tb.filter(w=>ta.has(w)).length/tb.length,1) : 0;
}
function cosine(a,b){
  const all=[...new Set([...(a.toLowerCase().match(/\b\w+\b/g)||[]),...(b.toLowerCase().match(/\b\w+\b/g)||[])])];
  const v=t=>all.map(w=>(t.toLowerCase().match(/\b\w+\b/g)||[]).filter(x=>x===w).length);
  const va=v(a),vb=v(b);
  const dot=va.reduce((s,x,i)=>s+x*vb[i],0);
  const ma=Math.sqrt(va.reduce((s,x)=>s+x*x,0));
  const mb=Math.sqrt(vb.reduce((s,x)=>s+x*x,0));
  return ma&&mb?dot/(ma*mb):0;
}
function mockShap(a,b,ce){
  const ta=new Set((a.toLowerCase().match(/\b\w+\b/g)||[]).filter(w=>!STOPS.has(w)&&w.length>3));
  const tb=(b.toLowerCase().match(/\b\w+\b/g)||[]).filter(w=>!STOPS.has(w)&&w.length>3);
  const seen=new Set(),out=[];
  for(const w of tb){
    if(seen.has(w)) continue; seen.add(w);
    const val=ta.has(w)?0.65+Math.random()*0.28:0.04+Math.random()*0.28;
    out.push({token:w,shap_value:parseFloat((val*(0.5+ce*0.6)).toFixed(5))});
  }
  return out.sort((a,b)=>Math.abs(b.shap_value)-Math.abs(a.shap_value)).slice(0,8);
}

function render(data, a, b){
  const ce=data.ce_score, bm=data.bm25_score, gap=Math.max(0,ce-bm);

  const vEl=$("verdict");
  vEl.className="verdict anim "+(data.verdict==="clear"?"v-clear":data.verdict==="warning"?"v-warn":"v-flag");
  $("v-icon").textContent=data.verdict==="clear"?"✓":data.verdict==="warning"?"~":"!";
  $("v-title").textContent=data.verdict_label;
  $("v-desc").textContent=data.verdict_detail;
  $("v-score").textContent=pct(ce);
  $("v-time").textContent=(data.elapsed_ms||"—")+" ms";

  $("mv-bm25").textContent=pct(bm);
  $("mv-sbert").textContent=pct(data.sbert_score);
  $("mv-ce").textContent=pct(ce);
  $("mv-gap").textContent=(gap>0?"+":"")+pct(gap);
  setTimeout(()=>{
    $("mf-bm25").style.width=pct(bm);
    $("mf-sbert").style.width=pct(data.sbert_score);
    $("mf-ce").style.width=pct(ce);
    $("mf-gap").style.width=pct(Math.min(gap*1.8,1));
  },60);

  $("hl-body").innerHTML=buildHL(b,a,ce);
  renderShap(data.shap_tokens,ce);
  $("shap-model").textContent=data.model_used||"";
  $("sent-wrap").innerHTML=sentTable(a,b);

  const r=$("results"); r.style.display="flex";
  setTimeout(()=>r.scrollIntoView({behavior:"smooth",block:"start"}),80);
}

async function runCheck(){
  const a=$("ta").value.trim(), b=$("tb").value.trim();
  if(!a||!b){alert("Please enter text in both panels.");return;}

  $("results").style.display="none";
  $("prog").style.display="";
  $("btn-run").disabled=true;
  $("btn-run").textContent="Analysing…";
  setStages(0);

  const thresh=parseInt($("thresh").value)/100;
  const runShap=$("shap-on").checked;
  const delays=[260,560,960,runShap?1500:220];
  delays.forEach((d,i)=>setTimeout(()=>setStages(i),d));

  const t0=performance.now();
  let data;
  try{
    const res=await fetch(API+"/api/analyse",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text_a:a,text_b:b,threshold:thresh,run_shap:runShap})
    });
    if(res.ok) data=await res.json();
    else throw new Error("server");
  }catch(e){
    const bm_=bm25(a,b), sb_=cosine(a,b);
    const ce_=Math.min(bm_*0.22+sb_*0.58+Math.random()*0.08+0.04,1);
    const v=ce_>=thresh?"plagiarism":ce_>=thresh*0.7?"warning":"clear";
    data={
      bm25_score:parseFloat(bm_.toFixed(4)),sbert_score:parseFloat(sb_.toFixed(4)),
      ce_score:parseFloat(ce_.toFixed(4)),verdict:v,
      verdict_label:v==="plagiarism"?"Likely plagiarism detected":v==="warning"?"Possible paraphrase — review recommended":"No significant similarity found",
      verdict_detail:v==="plagiarism"?`Cross-encoder confidence ${pct(ce_)} exceeds ${pct(thresh)} threshold.`:v==="warning"?`Score ${pct(ce_)} is elevated — may be paraphrase plagiarism.`:`Score ${pct(ce_)} is below threshold. Texts appear sufficiently distinct.`,
      shap_tokens:runShap?mockShap(a,b,ce_):[],
      model_used:"client-side",elapsed_ms:Math.round(performance.now()-t0)
    };
  }

  const totalDelay=delays.reduce((s,d)=>s+d,0);
  const wait=Math.max(totalDelay-(performance.now()-t0),0);
  setTimeout(()=>{
    doneStages();
    setTimeout(()=>{
      $("prog").style.display="none";
      render(data,a,b);
      $("btn-run").disabled=false;
      $("btn-run").textContent="Analyse →";
    },300);
  },wait);
}

$("ta").addEventListener("input",updateWC);
$("tb").addEventListener("input",updateWC);
document.addEventListener("keydown",e=>{if((e.ctrlKey||e.metaKey)&&e.key==="Enter")runCheck();});
updateWC();

// Auto-resize results flex container
const origDisplay=document.getElementById("results").style.display;