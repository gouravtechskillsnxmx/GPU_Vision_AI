import os
import uuid
import json
import time
import threading
import queue
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Optional deps (install only what you use)
import cv2
import numpy as np

# PaddleOCR (CPU/GPU depending on your env)
from paddleocr import PaddleOCR

# ---------------------------
# Config (Mumbai)
# ---------------------------
APP_NAME = os.getenv("APP_NAME", "LIC Mumbai - One File")
API_KEYS = set([k.strip() for k in os.getenv("API_KEYS", "agent1_key").split(",") if k.strip()])

# Usage limits (Mumbai-friendly)
MONTHLY_DOC_LIMIT = int(os.getenv("MONTHLY_DOC_LIMIT", "1000"))

# Storage
LOCAL_DIR = os.getenv("LOCAL_STORAGE_DIR", "./uploads")
os.makedirs(LOCAL_DIR, exist_ok=True)

# OCR
OCR_LANG = os.getenv("OCR_LANG", "en")
USE_GPU = os.getenv("USE_GPU", "false").lower() in ("1", "true", "yes")

# Note: Render has no GPU → keep USE_GPU=false on Render
ocr_engine = PaddleOCR(use_angle_cls=True, lang=OCR_LANG, use_gpu=USE_GPU)

# ---------------------------
# Simple in-memory DB (MVP)
# Replace with Postgres later if needed
# ---------------------------
JOBS: Dict[int, Dict[str, Any]] = {}
USAGE: Dict[str, Dict[str, int]] = {}  # tenant -> {yyyymm: docs}
JOB_ID = 0
LOCK = threading.Lock()

# ---------------------------
# In-process job queue
# ---------------------------
job_q: "queue.Queue[int]" = queue.Queue()

def yyyymm() -> str:
    return datetime.utcnow().strftime("%Y%m")

def require_tenant(x_api_key: str) -> str:
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

def check_doc_limit(tenant: str):
    ym = yyyymm()
    t = USAGE.setdefault(tenant, {})
    t.setdefault(ym, 0)
    if t[ym] + 1 > MONTHLY_DOC_LIMIT:
        raise HTTPException(status_code=402, detail=f"Monthly document limit exceeded ({MONTHLY_DOC_LIMIT})")
    t[ym] += 1

def save_upload(data: bytes, filename: str) -> str:
    safe = f"{uuid.uuid4().hex}_{os.path.basename(filename)}"
    path = os.path.join(LOCAL_DIR, safe)
    with open(path, "wb") as f:
        f.write(data)
    return path

def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to read image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def worker_loop():
    while True:
        job_id = job_q.get()
        try:
            job = JOBS.get(job_id)
            if not job:
                continue
            job["status"] = "running"
            job["updated_at"] = datetime.utcnow().isoformat()

            if job["job_type"] == "ocr":
                img = load_image_rgb(job["input_uri"])
                out = ocr_engine.ocr(img, cls=True)
                job["result"] = {"ocr": out}

            elif job["job_type"] == "face_verify":
                # Placeholder: wire your real insightface pipeline here.
                # Keep output stable: return a JSON with score/verified/boxes.
                job["result"] = {
                    "verified": False,
                    "match_score": None,
                    "note": "face_verify TODO: integrate insightface model + alignment + similarity threshold"
                }

            else:
                raise RuntimeError("Unknown job_type")

            job["status"] = "done"
            job["updated_at"] = datetime.utcnow().isoformat()
        except Exception as e:
            job = JOBS.get(job_id)
            if job:
                job["status"] = "failed"
                job["error"] = str(e)
                job["updated_at"] = datetime.utcnow().isoformat()
        finally:
            job_q.task_done()

threading.Thread(target=worker_loop, daemon=True).start()

# ---------------------------
# FastAPI + Dashboard
# ---------------------------
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

DASH_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LIC Office Dashboard</title>
  <style>
    :root{
      --bg:#f5f7fb; --card:#fff; --text:rgba(0,0,0,.88); --muted:rgba(0,0,0,.55);
      --shadow:0 12px 30px rgba(16,24,40,.08); --r:16px; --b:1px solid rgba(0,0,0,.06);
      --pri:#1677ff;
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;color:var(--text)}
    .wrap{max-width:1150px;margin:0 auto;padding:18px}
    .top{display:flex;justify-content:space-between;align-items:center;gap:12px}
    .brand{background:var(--card);border:var(--b);border-radius:var(--r);box-shadow:var(--shadow);padding:14px 16px}
    .brand .t{font-weight:750;letter-spacing:-.2px}
    .brand .s{color:var(--muted);font-size:12px;margin-top:4px}
    .nav{display:flex;gap:10px;flex-wrap:wrap}
    .btn{border:var(--b);background:var(--card);border-radius:14px;padding:10px 12px;cursor:pointer;transition:.15s;box-shadow:0 6px 16px rgba(0,0,0,.04)}
    .btn:hover{transform:translateY(-1px)}
    .btn.primary{background:var(--pri);color:#fff;border:1px solid rgba(0,0,0,.08)}
    .grid{display:grid;grid-template-columns:260px 1fr;gap:14px;margin-top:14px}
    @media(max-width:900px){.grid{grid-template-columns:1fr}}
    .side,.main{background:var(--card);border:var(--b);border-radius:var(--r);box-shadow:var(--shadow)}
    .side{padding:14px}
    .main{padding:14px}
    .h{font-size:18px;font-weight:800;margin:0 0 10px 0}
    .muted{color:var(--muted)}
    .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
    @media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}}
    @media(max-width:540px){.kpis{grid-template-columns:1fr}}
    .kpi{border:var(--b);border-radius:14px;padding:12px}
    .kpi .n{font-size:22px;font-weight:800;margin-top:6px}
    .card{border:var(--b);border-radius:14px;padding:12px}
    input,select{width:100%;padding:11px 12px;border:var(--b);border-radius:12px;outline:none}
    label{display:block;font-size:12px;color:var(--muted);margin:10px 0 6px}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    @media(max-width:700px){.row{grid-template-columns:1fr}}
    .table{width:100%;border-collapse:separate;border-spacing:0 8px}
    .tr{background:rgba(0,0,0,.02);border:var(--b);border-radius:12px}
    .tr td{padding:10px 10px}
    .pill{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;font-size:12px;border:var(--b);background:#fff}
    .pill.done{border-color:rgba(34,197,94,.25);background:rgba(34,197,94,.08)}
    .pill.fail{border-color:rgba(239,68,68,.25);background:rgba(239,68,68,.08)}
    .pill.run{border-color:rgba(59,130,246,.25);background:rgba(59,130,246,.08)}
    pre{white-space:pre-wrap;background:rgba(0,0,0,.03);padding:12px;border-radius:14px;border:var(--b);overflow:auto}
    .fade{animation:fade .18s ease}
    @keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
    .toast{position:fixed;right:16px;bottom:16px;background:#111;color:#fff;padding:10px 12px;border-radius:12px;opacity:0;transform:translateY(6px);transition:.2s}
    .toast.show{opacity:1;transform:translateY(0)}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="brand">
        <div class="t">LIC Office Suite</div>
        <div class="s">Mumbai plan • One-file app + dashboard</div>
      </div>
      <div class="nav">
        <button class="btn" onclick="go('overview')">Overview</button>
        <button class="btn" onclick="go('upload')">Upload & Run</button>
        <button class="btn" onclick="go('jobs')">Jobs</button>
        <button class="btn" onclick="go('settings')">Settings</button>
        <button class="btn primary" onclick="setKey()">Set API Key</button>
      </div>
    </div>

    <div class="grid">
      <div class="side">
        <div class="h">Quick actions</div>
        <div class="card">
          <div class="muted" style="font-size:12px">API Key</div>
          <div id="keyLabel" style="margin-top:6px;font-weight:700">Not set</div>
          <div class="muted" style="margin-top:8px;font-size:12px">Tip: set once, saved in browser.</div>
        </div>

        <div style="height:10px"></div>
        <div class="card">
          <div class="muted" style="font-size:12px">Job type</div>
          <select id="jobType" style="margin-top:8px">
            <option value="ocr">OCR (extract fields)</option>
            <option value="face_verify">Face Verify (KYC)</option>
          </select>
          <label>Pick file</label>
          <input id="file" type="file" />
          <button class="btn primary" style="width:100%;margin-top:10px" onclick="run()">Run</button>
        </div>
      </div>

      <div class="main fade" id="main">
        <!-- pages render here -->
      </div>
    </div>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    const $ = (id)=>document.getElementById(id)
    let state = { page:'overview', jobs:[], selected:null }

    function toast(msg){
      const t=$('toast'); t.textContent=msg; t.classList.add('show');
      setTimeout(()=>t.classList.remove('show'), 2200)
    }
    function apiKey(){ return localStorage.getItem('lic_api_key') || '' }
    function setKey(){
      const k = prompt("Paste API key (from your admin):", apiKey()||"");
      if(k!==null){ localStorage.setItem('lic_api_key', k.trim()); syncKey(); toast("Saved API key"); refresh(); }
    }
    function syncKey(){
      const k = apiKey();
      $('keyLabel').textContent = k ? "Set ✅" : "Not set";
    }

    async function request(url, opts={}){
      const headers = opts.headers || {};
      headers['X-API-Key']=apiKey();
      opts.headers=headers;
      const res = await fetch(url, opts);
      if(!res.ok){
        const txt = await res.text().catch(()=> '');
        throw new Error(txt || ("HTTP "+res.status));
      }
      const ct = res.headers.get('content-type')||'';
      if(ct.includes('application/json')) return res.json();
      return res.text();
    }

    async function loadJobs(){
      const data = await request('/v1/jobs?limit=50&offset=0');
      state.jobs = data.items || [];
      return state.jobs;
    }

    function renderOverview(){
      const total = state.jobs.length;
      const done = state.jobs.filter(j=>j.status==='done').length;
      const failed = state.jobs.filter(j=>j.status==='failed').length;
      const running = state.jobs.filter(j=>j.status==='queued'||j.status==='running').length;

      $('main').innerHTML = `
        <div class="pageTitle">
          <h2 class="h">Overview</h2>
          <div class="muted">Upload KYC/policy docs → extract fields → track results.</div>
        </div>
        <div class="kpis" style="margin-top:12px">
          <div class="kpi"><div class="muted">Jobs (last 50)</div><div class="n">${total}</div></div>
          <div class="kpi"><div class="muted">Done</div><div class="n">${done}</div></div>
          <div class="kpi"><div class="muted">Queued / Running</div><div class="n">${running}</div></div>
          <div class="kpi"><div class="muted">Failed</div><div class="n">${failed}</div></div>
        </div>
        <div style="height:12px"></div>
        <div class="card">
          <div style="font-weight:800;margin-bottom:8px">Recent jobs</div>
          ${state.jobs.slice(0,8).map(j=>`
            <div class="card" style="margin-bottom:10px;background:rgba(0,0,0,.015)">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
                <div>
                  <div style="font-weight:750">Job #${j.id} <span class="muted">(${j.job_type})</span></div>
                  <div class="muted" style="font-size:12px">${j.created_at||''}</div>
                </div>
                <div style="display:flex;gap:8px;align-items:center">
                  ${pill(j.status)}
                  <button class="btn" onclick="openJob(${j.id})">View</button>
                </div>
              </div>
            </div>
          `).join('') || `<div class="muted">No jobs yet. Upload to start.</div>`}
        </div>
      `;
    }

    function pill(status){
      if(status==='done') return `<span class="pill done">● done</span>`;
      if(status==='failed') return `<span class="pill fail">● failed</span>`;
      return `<span class="pill run">● ${status}</span>`;
    }

    async function openJob(id){
      go('job');
      const j = await request('/v1/jobs/'+id);
      state.selected = j;
      renderJob();
    }

    function renderJobs(){
      $('main').innerHTML = `
        <div>
          <h2 class="h">Jobs</h2>
          <div class="muted">Click any job to view output.</div>
          <div style="height:10px"></div>
          <div class="card">
            <table class="table">
              <tbody>
                ${state.jobs.map(j=>`
                  <tr class="tr" style="cursor:pointer" onclick="openJob(${j.id})">
                    <td style="width:90px;font-weight:800">#${j.id}</td>
                    <td style="width:150px">${j.job_type}</td>
                    <td style="width:160px">${pill(j.status)}</td>
                    <td class="muted" style="font-size:12px">${j.input_uri}</td>
                  </tr>
                `).join('') || `<tr><td class="muted">No jobs</td></tr>`}
              </tbody>
            </table>
          </div>
        </div>
      `;
    }

    function renderUpload(){
      $('main').innerHTML = `
        <div>
          <h2 class="h">Upload & Run</h2>
          <div class="muted">Use the panel on the left to run OCR or Face Verify.</div>
          <div style="height:12px"></div>
          <div class="card">
            <div style="font-weight:800;margin-bottom:6px">How it works</div>
            <div class="muted">
              1) Choose job type → 2) Pick file → 3) Run → 4) Results appear in Jobs.
            </div>
          </div>
        </div>
      `;
    }

    function renderSettings(){
      $('main').innerHTML = `
        <div>
          <h2 class="h">Settings</h2>
          <div class="muted">MVP settings are minimal. Next: add office branding, staff users, roles.</div>
          <div style="height:12px"></div>
          <div class="card">
            <div style="font-weight:800">API Key</div>
            <div class="muted" style="margin-top:6px">Used to identify your tenant/office.</div>
            <button class="btn primary" style="margin-top:10px" onclick="setKey()">Set / Change key</button>
          </div>
        </div>
      `;
    }

    function renderJob(){
      const j = state.selected;
      if(!j){ $('main').innerHTML = `<div class="muted">No job selected</div>`; return; }
      $('main').innerHTML = `
        <div>
          <h2 class="h">Job #${j.id}</h2>
          <div class="muted">Type: ${j.job_type} • Status: ${j.status}</div>
          <div style="height:10px"></div>
          <div class="card">
            <div style="display:flex;gap:10px;align-items:center;justify-content:space-between">
              <div>${pill(j.status)}</div>
              <div style="display:flex;gap:8px">
                <button class="btn" onclick="go('jobs')">Back</button>
                <button class="btn primary" onclick="openJob(${j.id})">Refresh</button>
              </div>
            </div>
            <div class="muted" style="margin-top:10px;font-size:12px">Input: ${j.input_uri}</div>
            ${j.error ? `<div class="card" style="margin-top:10px;border-color:rgba(239,68,68,.25)"><pre>${j.error}</pre></div>`:''}
            <div style="margin-top:10px;font-weight:800">Result JSON</div>
            <pre>${JSON.stringify(j.result || {}, null, 2)}</pre>
          </div>
        </div>
      `;
    }

    function go(page){
      state.page=page;
      if(page==='overview') renderOverview();
      if(page==='upload') renderUpload();
      if(page==='jobs') renderJobs();
      if(page==='settings') renderSettings();
      if(page==='job') renderJob();
      $('main').classList.remove('fade'); void $('main').offsetWidth; $('main').classList.add('fade');
    }

    async function run(){
      const f = $('file').files[0];
      if(!f) return toast("Choose a file first");
      const type = $('jobType').value;
      toast("Uploading…");
      const fd = new FormData();
      fd.append('file', f);
      const created = await request(`/v1/jobs/create?job_type=${type}`, { method:'POST', body: fd });
      toast("Job queued: #"+created.id);

      // poll until done/failed
      for(let i=0;i<60;i++){
        await new Promise(r=>setTimeout(r, 1000));
        const j = await request('/v1/jobs/'+created.id);
        if(j.status==='done'){ toast("Done ✅"); await refresh(); openJob(created.id); return; }
        if(j.status==='failed'){ toast("Failed ❌"); await refresh(); openJob(created.id); return; }
      }
      toast("Still running… check Jobs");
      await refresh();
      go('jobs');
    }

    async function refresh(){
      try{
        await loadJobs();
        if(state.page==='overview') renderOverview();
        if(state.page==='jobs') renderJobs();
      }catch(e){
        toast("Set API key first");
      }
    }

    syncKey();
    loadJobs().then(()=>go('overview')).catch(()=>go('overview'));
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASH_HTML)

@app.get("/health")
def health():
    return {"ok": True, "app": APP_NAME}

@app.post("/v1/jobs/create")
async def create_job(
    job_type: str = "ocr",
    file: UploadFile = File(...),
    x_api_key: str = Header(default="")
):
    tenant = require_tenant(x_api_key)
    if job_type not in ("ocr", "face_verify"):
        raise HTTPException(status_code=400, detail="job_type must be ocr|face_verify")

    check_doc_limit(tenant)

    data = await file.read()
    input_uri = save_upload(data, file.filename)

    global JOB_ID
    with LOCK:
        JOB_ID += 1
        jid = JOB_ID
        JOBS[jid] = {
            "id": jid,
            "tenant_id": tenant,
            "job_type": job_type,
            "status": "queued",
            "input_uri": input_uri,
            "result": None,
            "error": "",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

    job_q.put(jid)
    return {"id": jid, "status": "queued"}

@app.get("/v1/jobs")
def list_jobs(
    limit: int = 50,
    offset: int = 0,
    x_api_key: str = Header(default="")
):
    tenant = require_tenant(x_api_key)
    items = [j for j in JOBS.values() if j["tenant_id"] == tenant]
    items.sort(key=lambda x: x["id"], reverse=True)
    total = len(items)
    sliced = items[offset: offset + limit]
    # lightweight list view
    return {"total": total, "items": [{
        "id": j["id"], "job_type": j["job_type"], "status": j["status"],
        "input_uri": j["input_uri"], "created_at": j["created_at"]
    } for j in sliced]}

@app.get("/v1/jobs/{job_id}")
def get_job(job_id: int, x_api_key: str = Header(default="")):
    tenant = require_tenant(x_api_key)
    job = JOBS.get(job_id)
    if not job or job["tenant_id"] != tenant:
        raise HTTPException(status_code=404, detail="Not found")
    return job
