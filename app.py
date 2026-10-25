from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="InspectAI | Quality Intelligence",
    page_icon="🔍",
    layout="wide"
)

# Dashboard styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #101b30);
    color: #f1f5f9;
}
.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #142b4b, #203b63);
    border: 1px solid #31547e;
    margin-bottom: 22px;
}
.hero h1 {
    color: #ffffff;
    font-size: 42px;
    margin-bottom: 5px;
}
.hero p {
    color: #b9cce5;
}
.metric-card {
    background: #111f33;
    border: 1px solid #263d5b;
    border-radius: 14px;
    padding: 20px;
    min-height: 110px;
}
.metric-title {
    color: #9eb3ce;
    font-size: 14px;
}
.metric-value {
    color: #60a5fa;
    font-size: 23px;
    font-weight: 700;
    margin-top: 8px;
}
.result-card {
    background: #111f33;
    border: 1px solid #31547e;
    border-radius: 14px;
    padding: 22px;
    margin-top: 15px;
}
.small-note {
    color: #a9bad0;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <p>INDUSTRIAL AI • QUALITY CONTROL</p>
    <h1>🔍 InspectAI</h1>
    <p>AI-powered bottle inspection to help identify potential product defects.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">PRODUCT CATEGORY</div>
        <div class="metric-value">Bottle</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">MODEL TYPE</div>
        <div class="metric-value">Isolation Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">TRAINING IMAGES</div>
        <div class="metric-value">209 normal images</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📤 Upload Product Image")
    st.write("Choose a bottle image to analyze.")
    uploaded_file = st.file_uploader(
        "Select a JPG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown(
        '<p class="small-note">Prototype for educational demonstration. '
        'Use clear images for more meaningful results.</p>',
        unsafe_allow_html=True
    )

with right:
    st.subheader("📊 Inspection Result")

    model_path = Path(__file__).parent / "models" / "bottle_model.pkl"

    if not model_path.exists():
        st.error("Trained model not found. Run: python train_model.py")
    elif uploaded_file is None:
        st.markdown("""
        <div class="result-card">
            <h3>Waiting for an image</h3>
            <p>Upload a bottle image to view the model's prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        model = joblib.load(model_path)

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded product", use_container_width=True)

        resized_image = image.resize((32, 32))
        pixels = np.asarray(resized_image, dtype=np.float32) / 255.0
        features = pixels.flatten().reshape(1, -1)

        prediction = model.predict(features)[0]
        score = model.decision_function(features)[0]

        if prediction == -1:
            st.error("⚠️ Potential Anomaly Detected")
            st.write("The model has flagged this image for further inspection.")
        else:
            st.success("✅ No Anomaly Detected")
            st.write("The model did not flag this image as unusual.")

        st.markdown(
            f'<div class="result-card">'
            f'<div class="metric-title">MODEL DECISION SCORE</div>'
            f'<div class="metric-value">{score:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "The decision score is not a defect probability. "
            "This prototype can miss defects or flag normal images."
        )

st.divider()
st.markdown(
    '<p class="small-note">InspectAI • AI-Based Product Quality Inspection System '
    '• Educational prototype</p>',
    unsafe_allow_html=True
)
