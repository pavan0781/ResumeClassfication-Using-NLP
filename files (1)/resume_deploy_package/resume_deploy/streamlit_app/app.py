"""
Streamlit Frontend — Resume Classifier
----------------------------------------
Standalone app: works with embedded model OR calls FastAPI backend.
Deploy to: Streamlit Cloud (https://streamlit.io/cloud)
"""

import re
import io
import os
import pickle
import warnings
from pathlib import Path

import docx
import pdfplumber
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set API URL — change this to your deployed FastAPI URL
API_URL = os.getenv("API_URL", "http://localhost:8000")
USE_API  = os.getenv("USE_API", "false").lower() == "true"

CATEGORY_COLORS = {
    "React Developer": "#4C72B0",
    "Workday":         "#DD8452",
    "Peoplesoft":      "#55A868",
    "SQL Developer":   "#C44E52",
}
CATEGORY_ICONS = {
    "React Developer": "⚛️",
    "Workday":         "☁️",
    "Peoplesoft":      "🏢",
    "SQL Developer":   "🗄️",
}


# ── Custom CSS ───────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
}
.main-header h1 { color: white !important; font-size: 2rem; margin: 0; }
.main-header p  { color: #b0b8d4; margin: 0.5rem 0 0; font-size: 1rem; }

.result-card {
    background: white;
    border-left: 6px solid #4C72B0;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin: 1rem 0;
}
.result-card .category {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
}
.result-card .confidence-label {
    font-size: 0.85rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-box {
    background: #f8f9ff;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #e8ecf5;
}
.metric-box .metric-value { font-size: 1.6rem; font-weight: 700; color: #302b63; }
.metric-box .metric-label { font-size: 0.78rem; color: #888; text-transform: uppercase; }

.upload-zone {
    border: 2.5px dashed #302b63;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: #f8f9ff;
    margin: 1rem 0;
}

.stButton button {
    background: linear-gradient(135deg, #302b63, #0f0c29) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    width: 100%;
}
.stButton button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px);
}

.history-item {
    display: flex;
    justify-content: space-between;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    background: #f8f9ff;
    margin: 0.3rem 0;
    border: 1px solid #eaeef8;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ── Text Extraction ──────────────────────────────────────────────

def extract_docx(b):
    try:
        doc = docx.Document(io.BytesIO(b))
        return " ".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        return ""

def extract_pdf(b):
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            return " ".join(pg.extract_text() or "" for pg in pdf.pages)
    except Exception:
        return ""

def extract_doc(b):
    try:
        text = b.decode("latin-1", errors="ignore")
        return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)).strip()
    except Exception:
        return ""

def get_text(filename, file_bytes):
    ext = Path(filename).suffix.lower()
    if ext == ".docx":  return extract_docx(file_bytes)
    if ext == ".pdf":   return extract_pdf(file_bytes)
    if ext == ".doc":
        t = extract_docx(file_bytes)
        return t if len(t) > 50 else extract_doc(file_bytes)
    return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|\S+@\S+|\d{10}", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Model / API Prediction ────────────────────────────────────────

@st.cache_resource
def load_local_model():
    model_path = Path("model/model.pkl")
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_local(text):
    m = load_local_model()
    if m is None:
        return None
    pipeline = m["pipeline"]
    cleaned  = clean_text(text)
    pred     = pipeline.predict([cleaned])[0]
    try:
        proba = pipeline.predict_proba([cleaned])[0]
        conf  = {cls: round(float(p), 4) for cls, p in zip(pipeline.classes_, proba)}
    except AttributeError:
        conf = {pred: 1.0}
    return {"prediction": pred, "confidence": conf}

def predict_via_api(text):
    try:
        r = requests.post(
            f"{API_URL}/predict-text",
            json={"text": text},
            timeout=15
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def classify(text):
    if USE_API:
        return predict_via_api(text)
    return predict_local(text)


# ── Sidebar ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.radio("Prediction Mode", ["Local Model", "FastAPI Backend"])
    USE_API = (mode == "FastAPI Backend")

    if USE_API:
        API_URL = st.text_input("API URL", value=API_URL)
        try:
            r = requests.get(f"{API_URL}/health", timeout=3)
            if r.ok:
                st.success("✅ API Connected")
            else:
                st.error("❌ API unreachable")
        except Exception:
            st.warning("⚠️ Cannot reach API")

    st.markdown("---")
    st.markdown("### 📚 Categories")
    for cat, color in CATEGORY_COLORS.items():
        icon = CATEGORY_ICONS[cat]
        st.markdown(
            f'<div style="padding:6px 10px;border-radius:6px;background:{color}20;'
            f'border-left:4px solid {color};margin:4px 0;font-size:0.9rem">'
            f'{icon} {cat}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🔗 API Docs")
    st.markdown(f"[Swagger UI →]({API_URL}/docs)")
    st.markdown(f"[ReDoc →]({API_URL}/redoc)")


# ── Main Header ──────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📄 Resume Classifier</h1>
    <p>AI-powered resume categorization · React · Workday · Peoplesoft · SQL</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📤 Upload Resume", "✍️ Paste Text", "📊 Batch Classify"])


# ── Tab 1: File Upload ───────────────────────────────────────────

with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### Upload Resume File")
        uploaded = st.file_uploader(
            "Drag & drop or browse",
            type=["docx", "pdf", "doc"],
            label_visibility="collapsed"
        )

        if uploaded:
            file_bytes = uploaded.read()
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">📄</div>
                <div style="font-weight:600;margin-top:4px">{uploaded.name}</div>
                <div class="metric-label">{len(file_bytes)/1024:.1f} KB · {uploaded.type}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔍 Classify Resume", key="file_btn"):
                with st.spinner("Extracting text & classifying..."):
                    text   = get_text(uploaded.name, file_bytes)
                    result = classify(text)

                if result:
                    st.session_state["last_result"]   = result
                    st.session_state["last_filename"]  = uploaded.name
                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].insert(0, {
                        "file": uploaded.name,
                        "label": result["prediction"]
                    })

    with col_result:
        if "last_result" in st.session_state:
            result   = st.session_state["last_result"]
            pred     = result["prediction"]
            conf     = result.get("confidence", {})
            color    = CATEGORY_COLORS.get(pred, "#555")
            icon     = CATEGORY_ICONS.get(pred, "📄")
            top_conf = max(conf.values()) if conf else 0

            st.markdown(f"""
            <div class="result-card" style="border-left-color:{color}">
                <div class="confidence-label">Predicted Category</div>
                <div class="category" style="color:{color}">{icon} {pred}</div>
                <div style="margin-top:0.8rem">
                    <div class="confidence-label">Confidence</div>
                    <div style="font-size:1.3rem;font-weight:700;color:#333">{top_conf*100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if conf:
                st.markdown("#### 📊 Confidence per Category")
                fig, ax = plt.subplots(figsize=(5, 3))
                cats   = list(conf.keys())
                vals   = list(conf.values())
                colors = [CATEGORY_COLORS.get(c, "#aaa") for c in cats]
                bars   = ax.barh(cats, vals, color=colors)
                ax.bar_label(bars, fmt="%.1%%", padding=4, fontsize=10)
                ax.set_xlim(0, 1.2)
                ax.set_xlabel("Confidence")
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


# ── Tab 2: Text Input ────────────────────────────────────────────

with tab2:
    st.markdown("#### Paste Resume Text")
    text_input = st.text_area(
        "Resume text",
        placeholder="Paste resume content here...\n\nExample:\n5 years of experience in React JS, Redux, TypeScript...",
        height=250,
        label_visibility="collapsed"
    )

    if st.button("🔍 Classify Text", key="text_btn") and text_input.strip():
        with st.spinner("Classifying..."):
            result = classify(text_input)

        if result:
            pred     = result["prediction"]
            conf     = result.get("confidence", {})
            color    = CATEGORY_COLORS.get(pred, "#555")
            icon     = CATEGORY_ICONS.get(pred, "📄")
            top_conf = max(conf.values()) if conf else 0

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="result-card" style="border-left-color:{color}">
                    <div class="confidence-label">Result</div>
                    <div class="category" style="color:{color}">{icon} {pred}</div>
                    <div style="margin-top:0.6rem;font-size:1.2rem;font-weight:700">{top_conf*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                if conf:
                    df_conf = pd.DataFrame({
                        "Category":   list(conf.keys()),
                        "Confidence": [f"{v*100:.1f}%" for v in conf.values()]
                    })
                    st.dataframe(df_conf, use_container_width=True, hide_index=True)


# ── Tab 3: Batch ─────────────────────────────────────────────────

with tab3:
    st.markdown("#### Batch Classify Multiple Resumes")
    batch_files = st.file_uploader(
        "Upload multiple resumes",
        type=["docx", "pdf", "doc"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if batch_files and st.button("🚀 Classify All", key="batch_btn"):
        results = []
        progress = st.progress(0)
        for i, f in enumerate(batch_files):
            fb   = f.read()
            text = get_text(f.name, fb)
            res  = classify(text)
            pred = res["prediction"] if res else "Error"
            conf = max(res.get("confidence", {}).values()) if res else 0
            results.append({
                "Filename":   f.name,
                "Category":   pred,
                "Confidence": f"{conf*100:.1f}%",
            })
            progress.progress((i + 1) / len(batch_files))

        st.success(f"✅ Classified {len(results)} resumes!")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Summary chart
        counts = df["Category"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        colors  = [CATEGORY_COLORS.get(c, "#aaa") for c in counts.index]
        bars    = ax.barh(counts.index, counts.values, color=colors)
        ax.bar_label(bars, padding=4)
        ax.set_title("Batch Classification Summary")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results CSV", csv, "batch_results.csv", "text/csv")


# ── History ──────────────────────────────────────────────────────

if st.session_state.get("history"):
    st.markdown("---")
    st.markdown("#### 🕓 Recent Classifications")
    for item in st.session_state["history"][:5]:
        color = CATEGORY_COLORS.get(item["label"], "#888")
        icon  = CATEGORY_ICONS.get(item["label"], "📄")
        st.markdown(
            f'<div class="history-item">'
            f'<span>📄 {item["file"]}</span>'
            f'<span style="color:{color};font-weight:600">{icon} {item["label"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
