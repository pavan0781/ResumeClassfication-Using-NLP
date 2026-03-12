"""
train_and_save.py
-----------------
Run this ONCE to train the model and save it as model.pkl
Usage:
    python model/train_and_save.py --data_dir ./resumes_data
"""

import os
import re
import pickle
import argparse
import warnings
from pathlib import Path

import docx
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


# ── Text Extraction ──────────────────────────────────────────────

def extract_docx(path):
    try:
        doc = docx.Document(path)
        return " ".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        return ""

def extract_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            return " ".join(pg.extract_text() or "" for pg in pdf.pages)
    except Exception:
        return ""

def extract_doc(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("latin-1", errors="ignore")
        return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)).strip()
    except Exception:
        return ""

def extract_text(path):
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return extract_docx(path)
    elif ext == ".pdf":
        return extract_pdf(path)
    elif ext == ".doc":
        t = extract_docx(path)
        return t if len(t) > 50 else extract_doc(path)
    return ""


# ── Label Assignment ─────────────────────────────────────────────

def assign_label(filepath):
    s = str(filepath).lower()
    n = Path(filepath).name.lower()
    if "workday"     in s: return "Workday"
    if "peoplesoft"  in s: return "Peoplesoft"
    if "sql"         in s: return "SQL Developer"
    if any(k in n for k in ["react", "reactjs", "react js", "react dev"]): return "React Developer"
    if "internship"  in n: return "React Developer"
    return None


# ── Text Cleaning ────────────────────────────────────────────────

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|\S+@\S+|\d{10}", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Build Dataset ────────────────────────────────────────────────

def build_dataset(root_dir):
    X, y = [], []
    for fpath in Path(root_dir).rglob("*"):
        if fpath.suffix.lower() not in (".docx", ".doc", ".pdf"):
            continue
        label = assign_label(fpath)
        if not label:
            continue
        text = clean_text(extract_text(str(fpath)))
        if len(text) < 30:
            continue
        X.append(text)
        y.append(label)
    return X, y


# ── Train & Save ─────────────────────────────────────────────────

def train_and_save(data_dir: str, output_path: str = "model/model.pkl"):
    print(f"\n📂 Loading resumes from: {data_dir}")
    X, y = build_dataset(data_dir)
    print(f"✅ {len(X)} resumes loaded")

    from collections import Counter
    print("   Distribution:", dict(Counter(y)))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            stop_words="english"
        )),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
    print(f"   CV F1 Macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    pipeline.fit(X, y)
    print("✅ Model trained on full dataset")

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "pipeline": pipeline,
            "classes": pipeline.classes_.tolist(),
            "clean_text": clean_text,
        }, f)

    print(f"✅ Model saved to: {output_path}")
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./resumes_data", help="Path to resume folder")
    parser.add_argument("--output",   default="model/model.pkl", help="Save path for model")
    args = parser.parse_args()
    train_and_save(args.data_dir, args.output)
