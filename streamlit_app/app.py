import os
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import requests

st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE      = "cpu"
IMG_SIZE    = 224
MODEL_PATH  = "models/best_model.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    in_f  = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    
    MODEL_URL = "YOUR_HUGGINGFACE_RESOLVE_LINK"
    MODEL_PATH = "best_model.pth"
    
    def download_model():
        if not os.path.exists(MODEL_PATH):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    
    download_model()
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model
    return None

def predict(image, model):
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()
    pred_idx       = int(np.argmax(probs))
    pneumonia_prob = float(probs[1])
    return {
        "prediction":     CLASS_NAMES[pred_idx],
        "pneumonia_prob": pneumonia_prob,
        "normal_prob":    float(probs[0]),
        "confidence":     float(probs[pred_idx]),
        "risk":           "High"     if pneumonia_prob >= 0.80
                          else "Moderate" if pneumonia_prob >= 0.50
                          else "Low",
    }

st.markdown("""
<style>
.result-box{padding:18px 20px;border-radius:10px;margin-bottom:16px;border-left:5px solid}
.pneumonia{background:#fff5f5;border-color:#e53e3e}
.normal{background:#f0fff4;border-color:#38a169}
.pred-title{font-size:22px;font-weight:700;margin:0 0 4px}
.pred-risk{font-size:14px;margin:0}
.metric-row{display:flex;gap:16px;margin-top:12px}
.metric-box{flex:1;background:#f7fafc;border-radius:8px;padding:12px;text-align:center}
.metric-val{font-size:20px;font-weight:700;color:#2d3748}
.metric-lbl{font-size:11px;color:#718096;margin-top:2px}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown("""
**Architecture:** DenseNet121
**Pre-training:** ImageNet
**Fine-tuning:** NIH Chest X-Ray
**Test AUC:** ~0.97
**Test Accuracy:** ~92%
**Classes:** NORMAL · PNEUMONIA
    """)
    st.markdown("---")
    st.header("📊 Risk Levels")
    st.markdown("""
🟢 **Low** — PNEUMONIA prob < 50%
🟡 **Moderate** — PNEUMONIA prob 50–80%
🔴 **High** — PNEUMONIA prob > 80%
    """)
    st.markdown("---")
    st.warning("⚠️ For research purposes only. Not a substitute for medical diagnosis.")

st.title("🫁 Chest X-Ray Pneumonia Classifier")
st.markdown("Upload a frontal chest X-ray — classified as **NORMAL** or **PNEUMONIA** using DenseNet121 transfer learning.")
st.markdown("---")

model = load_model()

if model is None:
    st.error("Model weights not found. Please check the model file.")
    st.stop()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload X-Ray Image")
    uploaded = st.file_uploader("Supported: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

with col2:
    st.subheader("🔍 Prediction Result")
    if uploaded:
        with st.spinner("Analysing X-ray..."):
            result = predict(image, model)

        pred  = result["prediction"]
        risk  = result["risk"]
        css   = "pneumonia" if pred == "PNEUMONIA" else "normal"
        color = "#e53e3e"   if pred == "PNEUMONIA" else "#38a169"
        icon  = "🔴" if risk == "High" else "🟡" if risk == "Moderate" else "🟢"

        st.markdown(f"""
        <div class="result-box {css}">
            <p class="pred-title" style="color:{color}">{pred}</p>
            <p class="pred-risk">{icon} Risk Level: <b>{risk}</b></p>
        </div>
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val" style="color:#e53e3e">{result['pneumonia_prob']*100:.1f}%</div>
                <div class="metric-lbl">PNEUMONIA Probability</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="color:#38a169">{result['normal_prob']*100:.1f}%</div>
                <div class="metric-lbl">NORMAL Probability</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{result['confidence']*100:.1f}%</div>
                <div class="metric-lbl">Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.progress(result["pneumonia_prob"],
                    text=f"PNEUMONIA: {result['pneumonia_prob']*100:.1f}%")
        st.progress(result["normal_prob"],
                    text=f"NORMAL: {result['normal_prob']*100:.1f}%")
    else:
        st.info("👈 Upload a chest X-ray on the left to get started.")

st.markdown("---")
st.caption("Built by Kiran Tej Reddy Odugu · Dataset: NIH Chest X-Ray (Kaggle) · PyTorch + Streamlit")
