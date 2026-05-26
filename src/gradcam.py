import torch, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn, glob, random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Reload model ─────────────────────────────────────────────
model = models.densenet121(weights=None)
in_f  = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 2)
)
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.to(DEVICE).eval()

tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Grad-CAM ──────────────────────────────────────────────────
gradients, activations = {}, {}

def fwd_hook(m, inp, out):  activations['feat'] = out.detach()
def bwd_hook(m, gi, go):    gradients['feat']   = go[0].detach()

target_layer = model.features.denseblock4
target_layer.register_forward_hook(fwd_hook)
target_layer.register_full_backward_hook(bwd_hook)

def gradcam(img_path, true_label):
    orig  = Image.open(img_path).convert("RGB")
    t     = tf(orig).unsqueeze(0).to(DEVICE); t.requires_grad=True
    out   = model(t)
    probs = torch.softmax(out,1).squeeze()
    cls   = out.argmax(1).item()

    model.zero_grad()
    out[0, cls].backward()

    w   = gradients['feat'].mean(dim=(2,3), keepdim=True)
    cam = F.relu((w * activations['feat']).sum(1).squeeze())
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Resize heatmap
    hmap = np.array(Image.fromarray(np.uint8(255*cam)).resize(orig.size, Image.BILINEAR)) / 255.0
    color_hmap = cm.jet(hmap)[:,:,:3]
    overlay    = np.clip(0.55*np.array(orig)/255 + 0.45*color_hmap, 0, 1)

    pred_name = ["NORMAL","PNEUMONIA"][cls]
    conf      = float(probs[cls]) * 100

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"True: {true_label}  |  Pred: {pred_name}  ({conf:.1f}%)",
                 fontsize=14, fontweight="bold",
                 color="green" if pred_name==true_label else "red")
    axes[0].imshow(orig, cmap="gray"); axes[0].set_title("Original X-Ray"); axes[0].axis("off")
    axes[1].imshow(color_hmap);        axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay);           axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(f"results/gradcam_{true_label.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()

# ── Generate for both classes ──────────────────────────────────
gradcam(random.choice(glob.glob("data/chest_xray/test/PNEUMONIA/*.jpeg")), "PNEUMONIA")
gradcam(random.choice(glob.glob("data/chest_xray/test/NORMAL/*.jpeg")),    "NORMAL")