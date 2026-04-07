import uuid
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils import load_model, generate

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths — all relative to the repo root ────────────────────────────────────
DATA_ROOT  = Path("training_data")
IMAGES_DIR = DATA_ROOT / "images"
CSV_PATH   = DATA_ROOT / "session_table.csv"
OUTPUT_DIR = Path("output/predict")
CHECKPOINT_PATH = "checkpoints/latest (1).pth"

# Create folders if they don't exist yet
DATA_ROOT.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create the CSV with headers if it doesn't exist yet
if not CSV_PATH.exists():
    pd.DataFrame(columns=[
        "patient_id", "session_date", "interval_months",
        "FLAIR", "PRE", "POST", "T2"
    ]).to_csv(CSV_PATH, index=False)

# ── Load the model once when the server starts ────────────────────────────────
model_info = load_model(CHECKPOINT_PATH)
logger.info(f"Model loaded on {model_info['device']} | sequences: {model_info['sequences']}")

# ── In-memory task store ──────────────────────────────────────────────────────
# Stores status + results for each prediction job keyed by task_id
tasks: dict[str, dict] = {}

# ── FastAPI app setup ─────────────────────────────────────────────────────────
app = FastAPI(title="R-DDPM MRI Progression API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the output folder as static files so React can load result images by URL
app.mount("/output", StaticFiles(directory="output"), name="output")


# ── Response schemas ──────────────────────────────────────────────────────────
class SubmitResponse(BaseModel):
    task_id:    str
    status:     str
    patient_id: str
    n_sessions: int   # total sessions now in CSV for this patient

class StatusResponse(BaseModel):
    task_id:     str
    status:      str                   # queued | loading_sessions | running | done | error
    result_urls: list[str] | None = None
    error:       str        | None = None


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — POST /predict
# React sends the 4 MRI images + patient_id + interval here
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=SubmitResponse, status_code=202)
async def submit_prediction(
    background_tasks: BackgroundTasks,

    # The 4 MRI sequence images — field names must match exactly in the React form
    flair: UploadFile = File(...),
    pre:   UploadFile = File(...),
    post:  UploadFile = File(...),
    t2:    UploadFile = File(...),

    # Patient ID — e.g. "P001"
    patient_id: str = Form(...),

    # 1.0 = monthly scans, 0.5 = bi-weekly scans
    interval_months: float = Form(1.0),

    # How many future timepoints to generate
    n_future: int = Form(3),

    # How many existing sessions to use for GRU warm-up.
    # Leave as None to use ALL existing sessions for this patient.
    n_input: int = Form(None),
):
    # Validate file types
    allowed_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    for upload in (flair, pre, post, t2):
        if not upload.filename.lower().endswith(allowed_ext):
            raise HTTPException(400, f"Unsupported file: {upload.filename}. Use PNG or JPEG.")

    # ── Save uploaded images into training_data/images/<patient_id>/<session_id>/ ──
    session_id  = str(uuid.uuid4())[:8]
    session_dir = IMAGES_DIR / patient_id / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}
    for seq_name, upload in [("FLAIR", flair), ("PRE", pre), ("POST", post), ("T2", t2)]:
        ext      = Path(upload.filename).suffix.lower() or ".png"
        dest     = session_dir / f"{seq_name}{ext}"
        dest.write_bytes(await upload.read())
        # Store path relative to DATA_ROOT — this is what load_session_tensor expects
        saved_paths[seq_name] = str(dest.relative_to(DATA_ROOT))

    # ── Work out the session date based on previous sessions ──
    df           = pd.read_csv(CSV_PATH)
    patient_rows = df[df["patient_id"] == patient_id].sort_values("session_date")

    if patient_rows.empty:
        session_date = datetime.today().strftime("%Y-%m-%d")
    else:
        last_date    = datetime.strptime(patient_rows["session_date"].iloc[-1], "%Y-%m-%d")
        days_gap     = int(interval_months * 30)
        session_date = (last_date + timedelta(days=days_gap)).strftime("%Y-%m-%d")

    # ── Append new row to session_table.csv ──
    new_row = {
        "patient_id":      patient_id,
        "session_date":    session_date,
        "interval_months": interval_months,
        "FLAIR": saved_paths["FLAIR"],
        "PRE":   saved_paths["PRE"],
        "POST":  saved_paths["POST"],
        "T2":    saved_paths["T2"],
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    n_sessions = len(df[df["patient_id"] == patient_id])
    logger.info(f"Session saved for {patient_id} on {session_date}. Total sessions: {n_sessions}")

    # ── Queue background inference ──
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "queued"}

    background_tasks.add_task(
        run_inference,
        task_id, patient_id, n_input, n_future, interval_months
    )

    return SubmitResponse(
        task_id=task_id,
        status="queued",
        patient_id=patient_id,
        n_sessions=n_sessions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — GET /predict/{task_id}
# React polls this every few seconds to check if the result is ready
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/predict/{task_id}", response_model=StatusResponse)
def get_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    if task["status"] == "done":
        return StatusResponse(
            task_id=task_id,
            status="done",
            result_urls=task["result_urls"],
        )
    if task["status"] == "error":
        return StatusResponse(task_id=task_id, status="error", error=task["detail"])

    # Still running — return current status (queued / loading_sessions / running)
    return StatusResponse(task_id=task_id, status=task["status"])


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — GET /health
# Quick check that the server and model are running
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":    "ok",
        "device":    model_info["device"],
        "sequences": model_info["sequences"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND TASK — runs in the background after /predict returns
# This is where the actual model inference happens
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    task_id:         str,
    patient_id:      str,
    n_input:         int | None,
    n_future:        int,
    interval_months: float,
):
    try:
        tasks[task_id]["status"] = "running"

        task_out = OUTPUT_DIR / task_id
        task_out.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[{task_id}] patient={patient_id} | "
            f"n_input={n_input} | n_future={n_future}"
        )

        result = generate(
            model=model_info,
            data_path=str(DATA_ROOT),
            patient_id=patient_id,
            n_input=n_input,
            n_predict=n_future,
            interval_months=interval_months,
            out_dir=str(task_out),
        )

        result_urls = []
        for seq, paths in result.items():
            for p in paths:
                result_urls.append(f"/output/predict/{task_id}/{Path(p).name}")

        tasks[task_id] = {"status": "done", "result_urls": result_urls}
        logger.info(f"[{task_id}] Done — {len(result_urls)} images generated")

    except Exception as e:
        logger.error(f"[{task_id}] Failed: {e}", exc_info=True)
        tasks[task_id] = {"status": "error", "detail": str(e)}