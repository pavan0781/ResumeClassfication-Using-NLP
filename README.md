# 📄 Resume Classifier — Full Stack Deployment Guide

## 🏗️ Project Structure

```
resume_deploy/
├── model/
│   ├── train_and_save.py       # Train & save model to model.pkl
│   └── model.pkl               # Generated after training
│
├── api/
│   ├── main.py                 # FastAPI backend
│   └── requirements.txt
│
├── streamlit_app/
│   ├── app.py                  # Streamlit frontend
│   ├── requirements.txt
│   └── .streamlit/
│       └── secrets.toml        # Config for Streamlit Cloud
│
├── html_frontend/
│   └── index.html              # Standalone HTML UI → calls FastAPI
│
├── render.yaml                 # Render.com deploy config
└── README.md
```

---

## 🚀 Step-by-Step Deployment

---

### ① Train & Save the Model (Do this first!)

```bash
# Install deps
pip install python-docx pdfplumber scikit-learn

# Train model — point to your resume folder
python model/train_and_save.py --data_dir ./resumes_data

# model/model.pkl will be created
```

---

### ② Run FastAPI Locally

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:**       http://localhost:8000/redoc
- **Health:**      http://localhost:8000/health

---

### ③ Run Streamlit Locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```
Opens at: http://localhost:8501

---

### ④ Open HTML Frontend

Just open `html_frontend/index.html` in any browser.
Set the API URL to `http://localhost:8000` (or your deployed URL).

---

## ☁️ Deploy FastAPI → Render.com (Free)

1. Push your project to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Root Directory:** `api`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3.11
5. Add env var: `PYTHON_VERSION = 3.11.0`
6. Click Deploy → Your API will be live at:
   `https://resume-classifier-api.onrender.com`

> ⚠️ Important: Upload `model.pkl` to your repo or add a build step to retrain on deploy.

---

## 🎈 Deploy Streamlit → Streamlit Cloud (Free)

1. Push `streamlit_app/` folder to GitHub
2. Go to https://share.streamlit.io
3. Click **New app** → Connect your repo
4. Settings:
   - **Main file path:** `streamlit_app/app.py`
   - **Python version:** 3.11
5. Add Secrets (Settings → Secrets):
   ```toml
   USE_API = "true"
   API_URL = "https://your-api.onrender.com"
   ```
6. Click Deploy → Live at:
   `https://your-app.streamlit.app`

---

## 🔌 API Reference

| Method | Endpoint        | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/`             | API info and available endpoints     |
| GET    | `/health`       | Health check                         |
| GET    | `/classes`      | List all resume categories           |
| POST   | `/predict`      | Upload file (.docx/.pdf/.doc)        |
| POST   | `/predict-text` | Classify raw text as JSON            |

### Example — cURL
```bash
# File upload
curl -X POST "http://localhost:8000/predict" \
     -F "file=@resume.docx"

# Text input
curl -X POST "http://localhost:8000/predict-text" \
     -H "Content-Type: application/json" \
     -d '{"text": "5 years React JS Redux TypeScript..."}'
```

### Example — Python
```python
import requests

# File upload
with open("resume.docx", "rb") as f:
    r = requests.post("http://localhost:8000/predict", files={"file": f})
print(r.json())

# Text input
r = requests.post("http://localhost:8000/predict-text",
                  json={"text": "5 years React JS experience..."})
print(r.json())
# Output: {"prediction": "React Developer", "confidence": {...}, "top_confidence": 0.92}
```

---

## 🧩 Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    Users                             │
└────────┬─────────────────────────┬───────────────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐     ┌───────────────────┐
│  Streamlit App  │     │   HTML Frontend   │
│ (Streamlit Cloud)│     │  (any static host)│
└────────┬────────┘     └────────┬──────────┘
         │                       │
         └──────────┬────────────┘
                    ▼
         ┌──────────────────┐
         │   FastAPI API    │
         │  (Render.com)    │
         │                  │
         │  POST /predict   │
         │  POST /predict-  │
         │        text      │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │   model.pkl      │
         │ TF-IDF + Random  │
         │    Forest        │
         └──────────────────┘
```

---

## 📦 Categories Supported
- ⚛️ React Developer
- ☁️ Workday
- 🏢 Peoplesoft
- 🗄️ SQL Developer
