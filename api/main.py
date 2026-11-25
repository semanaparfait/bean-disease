"""FastAPI service: prediction, bulk upload, retraining trigger, monitoring and evaluation.

Run locally:  uvicorn api.main:app --port 8000
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import io
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import config as C
from src import database as db
from src import preprocessing as pp
from src.prediction import Predictor

START_TIME = time.time()
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")
MAX_UPLOAD_MB = 200

app = FastAPI(
    title="Bean Leaf Disease Classifier API",
    description="Predict bean leaf diseases from images, upload new labelled data and retrain the model.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = Predictor()
state = {"current_run": None}
_feature_cache: dict = {}


@app.on_event("startup")
def startup():
    db.init_db()
    if C.MODEL_PATH.exists():
        predictor.load()


# ---------------------------------------------------------------- monitoring
@app.get("/", tags=["monitoring"])
def root():
    return {"service": "bean-disease-api", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["monitoring"])
def health():
    """Model up-time / liveness information used by the dashboard."""
    return {
        "status": "ok" if predictor.ready else "model_not_loaded",
        "model_loaded": predictor.ready,
        "model_version": predictor.version,
        "started_at": STARTED_AT,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "retraining": state["current_run"] is not None,
        "server_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.get("/stats", tags=["monitoring"])
def stats():
    return {
        "dataset": {"train": pp.count_images(C.TRAIN_DIR), "test": pp.count_images(C.TEST_DIR)},
        "uploads": db.upload_summary(),
        "predictions": db.prediction_summary(),
        "retrain_threshold": C.RETRAIN_THRESHOLD,
        "auto_retrain": C.AUTO_RETRAIN,
        "classes": C.CLASS_NAMES,
    }


@app.get("/features", tags=["insights"])
def features():
    """Per-image interpretable features (green area, lesion area, texture...) for the visualisations."""
    if not C.FEATURES_CSV.exists():
        raise HTTPException(404, "Feature table not built yet (run the notebook or pp.build_feature_table())")
    mtime = C.FEATURES_CSV.stat().st_mtime
    if _feature_cache.get("mtime") != mtime:
        import pandas as pd
        _feature_cache.update(mtime=mtime, rows=pd.read_csv(C.FEATURES_CSV).round(4).to_dict(orient="records"))
    return _feature_cache["rows"]


# ---------------------------------------------------------------- prediction
@app.post("/predict", tags=["prediction"])
def predict(file: UploadFile = File(..., description="A bean leaf image (jpg/png)")):
    if not predictor.ready:
        raise HTTPException(503, "Model not loaded")
    data = file.file.read()
    if not data or not pp.validate_image(data):
        raise HTTPException(400, "File is not a valid image")
    result = predictor.predict(data)
    db.log_prediction(result["predicted_class"], result["confidence"], result["latency_ms"], result["model_version"])
    result["filename"] = file.filename
    return result


# ---------------------------------------------------------------- upload (bulk) + database
def _store(data: bytes, filename: str, label: str) -> Optional[dict]:
    if label not in C.CLASS_NAMES:
        return None
    ext = Path(filename).suffix.lower()
    if ext not in C.ALLOWED_EXT or not pp.validate_image(data):
        return None
    dest_dir = C.UPLOAD_DIR / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex[:10]}{ext}"
    dest.write_bytes(data)
    upload_id = db.add_upload(Path(filename).name, label, str(dest))
    return {"id": upload_id, "filename": Path(filename).name, "label": label}


def _label_from_zip_path(name: str) -> Optional[str]:
    for part in PurePosixPath(name.replace("\\", "/")).parts[:-1][::-1]:
        if part.lower() in C.CLASS_NAMES:
            return part.lower()
    return None


@app.post("/upload", tags=["retraining"])
def upload(background: BackgroundTasks,
           files: List[UploadFile] = File(..., description="Images, or .zip files with one folder per class"),
           label: Optional[str] = Form(None, description=f"Class of the images: one of {C.CLASS_NAMES}. "
                                                         "Optional for zips organised in class folders.")):
    """Bulk upload of labelled images for retraining. Files are validated, saved and recorded in SQLite."""
    if label is not None and label not in C.CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {C.CLASS_NAMES}")
    saved, rejected = [], []
    for f in files:
        data = f.file.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            rejected.append({"filename": f.filename, "reason": "too large"})
            continue
        if f.filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for info in z.infolist():
                        if info.is_dir() or "__MACOSX" in info.filename:
                            continue
                        lbl = _label_from_zip_path(info.filename) or label
                        rec = _store(z.read(info), info.filename, lbl) if lbl else None
                        (saved if rec else rejected).append(
                            rec or {"filename": info.filename, "reason": "no valid label or not an image"})
            except zipfile.BadZipFile:
                rejected.append({"filename": f.filename, "reason": "invalid zip"})
        elif label is None:
            rejected.append({"filename": f.filename, "reason": "label required for single images"})
        else:
            rec = _store(data, f.filename, label)
            (saved if rec else rejected).append(rec or {"filename": f.filename, "reason": "not a valid image"})

    summary = db.upload_summary()
    auto = None
    if C.AUTO_RETRAIN and summary["pending_total"] >= C.RETRAIN_THRESHOLD and state["current_run"] is None:
        auto = _launch_retrain("auto_threshold")
    return {"saved": len(saved), "rejected": rejected, "files": saved, "pending_for_retraining": summary["pending_total"],
            "retrain_threshold": C.RETRAIN_THRESHOLD, "auto_retrain_started": auto}


@app.get("/uploads", tags=["retraining"])
def uploads():
    return {"summary": db.upload_summary(), "pending": db.pending_uploads()[-200:]}


# ---------------------------------------------------------------- retraining
def _run_retrain(run_id: int, trigger: str):
    from src.model import retrain
    try:
        retrain(trigger=trigger, run_id=run_id,
                on_done=lambda status: predictor.load() if status == "completed" else None)
    except Exception as e:  # already recorded in the DB by retrain()
        print("Retraining failed:", e)
    finally:
        state["current_run"] = None


def _launch_retrain(trigger: str) -> int:
    run_id = db.start_run(trigger)
    state["current_run"] = run_id
    threading.Thread(target=_run_retrain, args=(run_id, trigger), daemon=True).start()
    return run_id


@app.post("/retrain", tags=["retraining"])
def trigger_retrain():
    """Trigger retraining on the uploaded data. Runs in the background; poll /retrain/status/{run_id}."""
    if state["current_run"] is not None:
        raise HTTPException(409, f"Retraining already running (run {state['current_run']})")
    if db.upload_summary()["pending_total"] == 0:
        raise HTTPException(400, "No new uploaded data. Upload labelled images first.")
    run_id = _launch_retrain("manual")
    return {"run_id": run_id, "status": "running", "status_url": f"/retrain/status/{run_id}"}


@app.get("/retrain/status/{run_id}", tags=["retraining"])
def retrain_status(run_id: int):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "Unknown run")
    return run


@app.get("/retrain/history", tags=["retraining"])
def retrain_history(limit: int = 20):
    return db.list_runs(limit)


# ---------------------------------------------------------------- model evaluation in production
@app.get("/model/metrics", tags=["evaluation"])
def model_metrics():
    from src.model import load_metrics
    m = load_metrics()
    if not m:
        raise HTTPException(404, "No trained model metrics found")
    return m


@app.post("/model/evaluate", tags=["evaluation"])
def evaluate_production_model():
    """Evaluate the model currently serving traffic on the held-out test set (live, in production)."""
    if not predictor.ready:
        raise HTTPException(503, "Model not loaded")
    from src.model import evaluate_low_memory
    t0 = time.time()
    m = evaluate_low_memory(predictor.model, pp.list_images(C.TEST_DIR))
    m["model_version"] = predictor.version
    m["duration_s"] = round(time.time() - t0, 1)
    return m
