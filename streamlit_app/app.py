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
MODEL_PATH  = "best_model.pth"  # Local file path where model will be saved
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

# REPLACE THIS WITH YOUR ACTUAL HUGGING FACE MODEL URL
# Your URL should look like: https://huggingface.co/username/repo-name/resolve/main/best_model.pth
MODEL_URL = "https://huggingface.co/kirantej1234/Chest-XRay-Classification/resolve/main/best_model.pth"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def download_model():
    """Download model from Hugging Face if not exists locally"""
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner("📥 Downloading model from Hugging Face... Please wait..."):
                response = requests.get(MODEL_URL, stream=True, timeout=60)
                response.raise_for_status()
                
                # Get file size for progress bar
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Download with progress
                downloaded = 0
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = downloaded / total_size
                                progress_bar.progress(progress)
                                progress_text.text(f"Downloaded: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB")
                
                progress_bar.empty()
                progress_text.empty()
                st.success("✅ Model downloaded successfully from Hugging Face!")
                return True
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to download model: {str(e)}")
            st.info("Please check your MODEL_URL and make sure it's accessible.")
            return False
    return True

@st.cache_resource
def load_model():
    """Load the trained model"""
    # Download model if needed
    if not download_model():
        return None
    
    # Create model architecture
    model = models.densenet121(weights=None)
    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    
    # Load model weights
    try:
        with st.spinner("🔄 Loading model weights..."):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def predict(image, model):
    """Run prediction on uploaded image"""
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()
    
    pred_idx = int(np.argmax(probs))
    pneumonia_prob = float(probs[1])
    
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "pneumonia_prob": pneumonia_prob,
        "normal_prob": float(probs[0]),
        "confidence": float(probs[pred_idx]),
        "risk": "High" if pneumonia_prob >= 0.80
                else "Moderate" if pneumonia_prob >= 0.50
                else "Low",
    }

# Custom CSS styling
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

# Sidebar
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
    
    # Show model status
    st.markdown("---")
    st.header("📦 Model Status")
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.success(f"✅ Model loaded ({file_size:.1f} MB)")
    else:
        st.info("⏳ Model will be downloaded from Hugging Face")

# Main content
st.title("🫁 Chest X-Ray Pneumonia Classifier")
st.markdown("Upload a frontal chest X-ray — classified as **NORMAL** or **PNEUMONIA** using DenseNet121 transfer learning.")
st.markdown("---")

# Load model
model = load_model()

if model is None:
    st.error("""
    ### ❌ Failed to load model
    
    Please check:
    1. Your Hugging Face URL is correct
    2. The model file exists on Hugging Face
    3. Your internet connection is stable
    
    **Required URL format:** `https://huggingface.co/username/repo-name/resolve/main/best_model.pth`
    """)
    st.stop()

# Create columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload X-Ray Image")
    uploaded = st.file_uploader("Supported: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

with col2:
    st.subheader("🔍 Prediction Result")
    if uploaded:
        with st.spinner("🔬 Analysing X-ray..."):
            result = predict(image, model)

        pred = result["prediction"]
        risk = result["risk"]
        css = "pneumonia" if pred == "PNEUMONIA" else "normal"
        color = "#e53e3e" if pred == "PNEUMONIA" else "#38a169"
        icon = "🔴" if risk == "High" else "🟡" if risk == "Moderate" else "🟢"

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
        
        # Display probability bars
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.metric("PNEUMONIA Probability", f"{result['pneumonia_prob']*100:.1f}%")
            st.progress(result["pneumonia_prob"])
        with col_prob2:
            st.metric("NORMAL Probability", f"{result['normal_prob']*100:.1f}%")
            st.progress(result["normal_prob"])
            
        # Interpretation guide
        if result['pneumonia_prob'] > 0.5:
            st.warning("⚠️ This X-ray shows signs consistent with PNEUMONIA. Please consult a healthcare professional.")
        else:
            st.success("✅ This X-ray appears NORMAL. Continue to monitor symptoms as needed.")
    else:
        st.info("👈 Upload a chest X-ray on the left to get started.")

st.markdown("---")
st.caption("Built by Kiran Tej Reddy Odugu · Dataset: NIH Chest X-Ray (Kaggle) · PyTorch + Streamlit")
