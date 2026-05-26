# 🫁 Chest X-Ray Pneumonia Classification

> Deep learning pipeline for pneumonia detection from chest X-rays using transfer learning (DenseNet121) with Grad-CAM explainability.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Problem Statement

Pneumonia causes over 2.5 million deaths annually. Chest X-rays are the primary diagnostic tool but require expert radiologists. This project builds a binary classifier (NORMAL vs PNEUMONIA) using deep learning to assist radiologists, with Grad-CAM explainability to highlight lung regions driving predictions.

---

## Dataset

**Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

---

## Results

| Metric            | Value     |
|-------------------|-----------|
| Test Accuracy     | **92.1%** |
| ROC-AUC           | **0.974** |
| PNEUMONIA F1      | **0.94**  |
| PNEUMONIA Recall  | **0.97**  |
| NORMAL Recall     | **0.90**  |

Baseline: naive majority-class predictor = 62.5% accuracy. Logistic regression AUC ≈ 0.73.

---

## Project Structure

```
chest-xray-classification/
├── streamlit_app/
│   └── app.py                  # Deployed web app
├── src/
│   ├── train.py                # Training pipeline
│   ├── inference.py            # Prediction script
│   └── gradcam.py              # Grad-CAM explainability
├── models/
│   └── best_model.pth          # Trained weights
├── results/
│   ├── evaluation_plots.png
│   └── gradcam_sample.png
├── notebooks/
│   └── eda.ipynb
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Usage

```bash
git clone https://github.com/Odugu-KiranTejReddy/chest-xray-classification
cd chest-xray-classification
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| DenseNet121 | Used in Stanford CheXNet; dense connections preserve fine-grained features |
| Progressive unfreezing | Avoids catastrophic forgetting of ImageNet features |
| WeightedRandomSampler | Handles 3:1 class imbalance without synthetic data |
| Class weights [1.0, 2.5] | Penalises missing pneumonia 2.5× more — maximises recall |
| Grad-CAM | Highlights lung regions — essential for clinical trust |

---

## References

1. Huang et al. (2017) — [DenseNet](https://arxiv.org/abs/1608.06993)
2. Rajpurkar et al. (2017) — [CheXNet](https://arxiv.org/abs/1711.05225)
3. Selvaraju et al. (2017) — [Grad-CAM](https://arxiv.org/abs/1610.02391)

---

> ⚠️ **Disclaimer:** For educational and research purposes only. Not a substitute for professional medical diagnosis.
