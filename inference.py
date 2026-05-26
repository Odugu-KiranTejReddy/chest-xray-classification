import torch, torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image

# ── Load model ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model  = models.densenet121(weights=None)
in_f   = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 2)
)
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.to(DEVICE).eval()
print("Model loaded ✓")

# ── Transform ────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Predict function ─────────────────────────────────────────
def predict(image_path):
    img    = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1).squeeze().cpu().numpy()
    pred  = ["NORMAL","PNEUMONIA"][np.argmax(probs)]
    pneu  = float(probs[1])
    risk  = "High" if pneu >= 0.80 else "Moderate" if pneu >= 0.50 else "Low"
    print(f"\n{'─'*40}")
    print(f"  Prediction       : {pred}")
    print(f"  PNEUMONIA Prob   : {pneu*100:.2f}%")
    print(f"  NORMAL Prob      : {float(probs[0])*100:.2f}%")
    print(f"  Confidence       : {float(np.max(probs))*100:.2f}%")
    print(f"  Risk Level       : {risk}")
    print(f"{'─'*40}\n")
    return pred, pneu

# ── Test on a sample image ────────────────────────────────────
import glob, random
pneumonia_imgs = glob.glob("data/chest_xray/test/PNEUMONIA/*.jpeg")
normal_imgs    = glob.glob("data/chest_xray/test/NORMAL/*.jpeg")

print("── PNEUMONIA sample ──")
predict(random.choice(pneumonia_imgs))

print("── NORMAL sample ──")
predict(random.choice(normal_imgs))