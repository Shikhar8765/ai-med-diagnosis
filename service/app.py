# 3_service/app.py

import streamlit as st
import tempfile
from inference import InferenceEngine
from PIL import Image
import os

# Load your trained model
engine = InferenceEngine("../2_models/weights/resnet50_med.pth",
                         ["Normal", "Pneumonia"])

st.set_page_config(page_title="AI Medical Diagnosis", layout="centered")
st.title("🧠 AI-Powered Chest X-ray Diagnosis")

# Upload image
uploaded_file = st.file_uploader("Upload a PNG/JPG chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Show uploaded image
    st.image(Image.open(temp_path), caption="Uploaded Image", use_column_width=True)

    # Run prediction
    with st.spinner("Diagnosing..."):
        result = engine.predict(temp_path)

    st.subheader("🩺 Prediction Result")
    st.markdown(f"**Diagnosis:** `{result['label']}`")
    st.markdown(f"**Confidence:** `{result['confidence']*100:.2f}%`")

    # Show full probability breakdown
    st.write("Class Probabilities:")
    st.json(result["probabilities"])
