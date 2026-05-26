
import os, random, numpy as np, matplotlib.pyplot as plt, seaborn as sns

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings; warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

CONFIG = {
    "data_dir":      "data/chest_xray",
    "model_dir":     "models",
    "results_dir":   "results",
    "img_size":      224,
    "batch_size":    32,
    "epochs":        30,
    "lr":            1e-4,
    "weight_decay":  1e-5,
    "patience":      15,      # was 5 — too aggressive before
    "num_classes":   2,
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    # ── Key fix: penalise missing PNEUMONIA more than missing NORMAL ──
    "class_weights": [1.0, 2.5],
}
os.makedirs(CONFIG["model_dir"],   exist_ok=True)
os.makedirs(CONFIG["results_dir"], exist_ok=True)
print(f"[INFO] Device: {CONFIG['device']}")

MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def build_loaders(data_dir, batch_size):
    train_ds = datasets.ImageFolder(f"{data_dir}/train", train_tf)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val",   val_tf)
    test_ds  = datasets.ImageFolder(f"{data_dir}/test",  val_tf)

    class_counts   = np.bincount(train_ds.targets)
    sample_weights = [1.0/class_counts[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Classes: {train_ds.classes} | Counts: NORMAL={class_counts[0]}, PNEUMONIA={class_counts[1]}\n")

    return (
        DataLoader(train_ds, batch_size, sampler=sampler,  num_workers=2, pin_memory=True),
        DataLoader(val_ds,   batch_size, shuffle=False,    num_workers=2, pin_memory=True),
        DataLoader(test_ds,  batch_size, shuffle=False,    num_workers=2, pin_memory=True),
        train_ds.classes
    )

def build_model(num_classes, device):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False

    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable params (head only): {trainable:,}\n")
    return model

def unfreeze(model):
    for p in model.parameters(): p.requires_grad = True
    print(f"[MODEL] All layers unfrozen: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params\n")
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * X.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum/total, correct/total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out   = model(X)
            loss  = criterion(out, y)
            probs = torch.softmax(out, 1)[:, 1]
            preds = out.argmax(1)
            loss_sum += loss.item() * X.size(0)
            correct  += (preds == y).sum().item()
            total    += y.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return loss_sum/total, correct/total, auc, all_preds, all_labels, all_probs

def run(config):
    device = config["device"]
    train_loader, val_loader, test_loader, class_names = build_loaders(
        config["data_dir"], config["batch_size"]
    )
    model = build_model(config["num_classes"], device)

    # ── Weighted loss: penalise missing PNEUMONIA 2.5x more ──────────
    weights   = torch.tensor(config["class_weights"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler    = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    history      = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "val_auc":[]}
    best_auc     = 0.0
    best_recall  = 0.0
    patience_counter = 0
    UNFREEZE_EPOCH   = 5

    print("=" * 60); print("TRAINING"); print("=" * 60)

    for epoch in range(1, config["epochs"] + 1):
        if epoch == UNFREEZE_EPOCH:
            model = unfreeze(model)
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["lr"] / 10,
                weight_decay=config["weight_decay"]
            )
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_auc, vl_preds, vl_labels, _ = eval_epoch(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(vl_auc)
        new_lr = optimizer.param_groups[0]['lr']

        for k, v in zip(["train_loss","val_loss","train_acc","val_acc","val_auc"],
                        [tr_loss, vl_loss, tr_acc, vl_acc, vl_auc]):
            history[k].append(v)

        print(f"Ep[{epoch:02d}/{config['epochs']}] "
              f"TrLoss:{tr_loss:.4f} TrAcc:{tr_acc:.4f} | "
              f"VlLoss:{vl_loss:.4f} VlAcc:{vl_acc:.4f} VlAUC:{vl_auc:.4f} | "
              f"LR:{current_lr:.0e}")

        if new_lr < current_lr:
            print(f"  → LR reduced to {new_lr:.2e}")

        # ── Save best model based on AUC ─────────────────────────────
        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), f"{config['model_dir']}/best_model.pth")
            print(f"  ✓ Saved best model (AUC: {best_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n[EARLY STOP] No improvement for {config['patience']} epochs.")
                break

    print(f"\nBest Val AUC: {best_auc:.4f}")
    return model, history, test_loader, class_names

def evaluate_and_plot(model, test_loader, class_names, history, config):
    device    = config["device"]
    weights   = torch.tensor(config["class_weights"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model.load_state_dict(
        torch.load(f"{config['model_dir']}/best_model.pth", map_location=device)
    )
    _, acc, auc, preds, labels, probs = eval_epoch(model, test_loader, criterion, device)

    print("\n" + "="*60)
    print(f"TEST Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"TEST ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(labels, probs)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC={auc:.4f}")
    axes[1].plot([0,1],[0,1],"k--", lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    axes[1].set_title("ROC Curve", fontweight="bold")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].legend()

    ep = range(1, len(history["val_auc"])+1)
    axes[2].plot(ep, history["train_acc"], label="Train Acc", color="coral")
    axes[2].plot(ep, history["val_acc"],   label="Val Acc",   color="steelblue")
    axes[2].plot(ep, history["val_auc"],   label="Val AUC",   color="seagreen", linestyle="--")
    axes[2].set_title("Training History", fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{config['results_dir']}/evaluation_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlots saved → results/evaluation_plots.png")

if __name__ == "__main__":
    model, history, test_loader, class_names = run(CONFIG)
    evaluate_and_plot(model, test_loader, class_names, history, CONFIG)
