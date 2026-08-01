"""
ui.py

Streamlit UI for the Skin Cancer ISIC Classifier.
Talks to the FastAPI backend (app.py) over HTTP.
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://skin-cancer-classifier-z0bg.onrender.com")

st.set_page_config(page_title="Skin Cancer Classifier MLOps Console", page_icon="🩺", layout="wide")
st.title("🩺 Skin Cancer Detection & MLOps Console")

# ---------------- SIDEBAR: SYSTEM STATUS ----------------
st.sidebar.header("System Status")
try:
    res = requests.get(f"{API_URL}/health", timeout=15).json()
    st.sidebar.success(f"Status: {res['status'].upper()}")
    st.sidebar.write(f"Model file present: {'✅' if res['model_exists'] else '❌'}")
    st.sidebar.write(f"Classes tracked: {res['num_classes']}")
except Exception:
    st.sidebar.error("API Offline — is the Render service awake?")

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📤 Upload & Retrain", "📊 Data Insights"])

# ---------------- TAB 1: SINGLE PREDICTION ----------------
with tab1:
    st.header("Predict a Single Image")
    file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

    if file:
        st.image(file, caption="Uploaded image", width=300)

    if file and st.button("Classify Image", type="primary"):
        with st.spinner("Running prediction..."):
            res = requests.post(f"{API_URL}/predict", files={"file": (file.name, file.getvalue())}, timeout=60)

        if res.status_code == 200:
            data = res.json()
            st.subheader(f"Prediction: {data['predicted_class']}")
            st.write(f"Confidence: **{data['confidence']:.2%}**")

            st.write("Full probability breakdown:")
            probs = data["all_probabilities"]
            sorted_probs = dict(sorted(probs.items(), key=lambda x: -x[1]))
            st.bar_chart(sorted_probs)
        else:
            st.error(f"Prediction failed: {res.json().get('detail', res.text)}")

# ---------------- TAB 2: UPLOAD + RETRAIN ----------------
with tab2:
    st.header("Upload Bulk Training Data")
    st.caption(
        "Upload a .zip file containing new images organized into folders matching "
        "your existing class names (e.g. `melanoma/`, `nevus/`, etc.)"
    )
    bulk_zip = st.file_uploader("Upload bulk data (.zip)", type=["zip"])

    if bulk_zip and st.button("Upload Zip"):
        with st.spinner("Uploading and extracting..."):
            res = requests.post(f"{API_URL}/upload", files={"file": (bulk_zip.name, bulk_zip.getvalue())}, timeout=60)
        if res.status_code == 200:
            st.success(res.json()["message"])
        else:
            st.error(f"Upload failed: {res.json().get('detail', res.text)}")

    st.divider()

    st.header("Trigger Retraining")
    st.caption("Retrains the model using everything currently in data/train, including any newly uploaded images.")
    if st.button("🔄 Trigger Retraining", type="primary"):
        with st.spinner("Retraining in progress — this can take a few minutes..."):
            res = requests.post(f"{API_URL}/retrain", timeout=600)
        if res.status_code == 200:
            st.success(res.json()["message"])
        else:
            st.error(f"Retraining failed: {res.json().get('detail', res.text)}")

# ---------------- TAB 3: DATA VISUALIZATIONS ----------------
with tab3:
    st.header("Dataset Insights")
    st.info(
        "Class distribution charts are available in the training notebook — "
        "this environment doesn't have direct access to the training data folder."
    )
