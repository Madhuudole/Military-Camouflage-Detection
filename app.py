# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🎯 Military Camouflage Detector")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    model = YOLO("yolov8n.pt")  # Replace with trained weights
    results = model(np.array(img))

    st.subheader("🔍 Detection Results")
    results.show()  # Displays bounding boxes

To run:
streamlit run app.py
