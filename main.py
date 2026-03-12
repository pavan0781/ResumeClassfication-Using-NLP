"""
FastAPI Backend — Resume Classification API
--------------------------------------------
Endpoints:
  GET  /              → API info
  GET  /health        → Health check
  POST /predict       → Classify a resume (file upload)
  POST /predict-text  → Classify raw text
  GET  /classes       → List all supported categories
"""

import re
import io
import pickle
from pathlib import Path
from typing import Optional

import docx
import pdfplumber
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Load Model ───────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent.parent / "model" / "model.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run: python model/train_and_save.py --data_dir ./resumes_data"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model_data = load_model()
    pipeline   = model_data["pipeline"]
    CLASSES    = model_data["classes"]
    print(f"✅ Model loaded. Classes: {CLASSES}")
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    pipeline = None
    CLASSES  = []


# ── FastAPI App ──────────────────────────────────────────────────

app = FastAPI(
    title="Resume Classification API",
    description="Classify resumes into categories: React Developer, Workday, Peoplesoft, SQL Developer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Text Extraction ──────────────────────────────────────────────

def extract_from_docx(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return " ".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        return ""

def extract_from_pdf(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return " ".join(pg.extract_text() or "" for pg in pdf.pages)
    except Exception:
        return ""

def extract_from_doc(file_bytes: bytes) -> str:
    try:
        text = file_bytes.decode("latin-1", errors="ignore")
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""

def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".docx":
        return extract_from_docx(file_bytes)
    elif ext == ".pdf":
        return extract_from_pdf(file_bytes)
    elif ext == ".doc":
        t = extract_from_docx(file_bytes)
        return t if len(t) > 50 else extract_from_doc(file_bytes)
    raise HTTPException(status_code=400, detail="Unsupported file type. Use .docx, .pdf, or .doc")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|\S+@\S+|\d{10}", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def predict(text: str) -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    try:
        proba = pipeline.predict_proba([cleaned])[0]
        confidence = {cls: round(float(p), 4) for cls, p in zip(pipeline.classes_, proba)}
    except AttributeError:
        confidence = {pred: 1.0}
    return {
        "prediction": pred,
        "confidence": confidence,
        "top_confidence": round(max(confidence.values()), 4)
    }


# ── Routes ───────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name":    "Resume Classification API",
        "version": "1.0.0",
        "status":  "running",
        "model":   "loaded" if pipeline else "not loaded",
        "classes": CLASSES,
        "endpoints": {
            "POST /predict":       "Upload a resume file (.docx/.pdf/.doc)",
            "POST /predict-text":  "Send raw text for classification",
            "GET  /classes":       "List all categories",
            "GET  /health":        "Health check",
            "GET  /docs":          "Interactive API docs (Swagger)",
        }
    }

@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.get("/classes", tags=["Info"])
def get_classes():
    return {"classes": CLASSES, "count": len(CLASSES)}


class TextRequest(BaseModel):
    text: str
    filename: Optional[str] = "resume.txt"


@app.post("/predict-text", tags=["Classify"])
def predict_text(req: TextRequest):
    """Classify resume from raw text input."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = predict(req.text)
    return {
        "filename": req.filename,
        "text_length": len(req.text),
        **result
    }


@app.post("/predict", tags=["Classify"])
async def predict_file(file: UploadFile = File(...)):
    """Upload a .docx / .pdf / .doc resume file and get its category."""
    allowed = {".docx", ".pdf", ".doc"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {allowed}"
        )

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    text = extract_text(file.filename, file_bytes)
    if len(text.strip()) < 30:
        raise HTTPException(status_code=422, detail="Could not extract enough text from the file.")

    result = predict(text)
    return {
        "filename":    file.filename,
        "file_size":   len(file_bytes),
        "text_length": len(text),
        **result
    }


# ── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
