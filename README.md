# 🛡️ Smart Quality Inspection & Root Cause Intelligence Platform

> An industrial Vision AI platform for automated steel surface defect detection, severity analysis, explainable AI, and quality analytics.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Django](https://img.shields.io/badge/Django-4.2+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Overview

This platform goes beyond simple image classification to deliver a **complete industrial quality intelligence system**. It combines deep learning, computer vision, and analytics into an enterprise-grade web dashboard.

### Key Capabilities

| Feature | Technology |
|---------|-----------|
| **Defect Classification** | ResNet50 (Transfer Learning, Fine-tuned) |
| **Severity Estimation** | Mask-area analysis + Grad-CAM proxy |
| **Explainable AI** | Grad-CAM heatmap overlays |
| **Similarity Retrieval** | FAISS cosine similarity search |
| **Quality Analytics** | KPI engine with automated insights |
| **Batch Inspection** | Multi-image processing with CSV reports |
| **Web Dashboard** | Django + Bootstrap 5 + Plotly.js |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                 Django Dashboard                  │
│  ┌───────────┬───────────┬──────────┬──────────┐ │
│  │ Executive │ Inspection│Root Cause│  Batch   │ │
│  │ Dashboard │ Workspace │ Analysis │  Report  │ │
│  └─────┬─────┴─────┬─────┴────┬─────┴────┬─────┘ │
│        │           │          │          │        │
│  ┌─────▼───────────▼──────────▼──────────▼─────┐ │
│  │           Inference Engine (Singleton)        │ │
│  └──┬────────┬──────────┬──────────┬────────────┘ │
│     │        │          │          │               │
│  ┌──▼──┐ ┌──▼──┐  ┌────▼────┐ ┌──▼──────┐       │
│  │Class│ │Sev. │  │Grad-CAM │ │  FAISS  │       │
│  │ifier│ │Est. │  │         │ │ Search  │       │
│  └─────┘ └─────┘  └─────────┘ └─────────┘       │
│                                                   │
│  ┌──────────────────────────────────────────────┐ │
│  │  Analytics Engine (KPIs + Insights)          │ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Smart-Trace-Platfrom/
├── manage.py                    # Django entry point
├── requirements.txt             # Dependencies
├── config/                      # Django project settings
├── ml_pipeline/                 # All ML/AI code
│   ├── config.py                # Centralized configuration
│   ├── data/                    # Dataset processing (Module 1)
│   ├── classifier/              # ResNet50 classifier (Module 2)
│   ├── severity/                # Severity estimation (Module 3)
│   ├── retrieval/               # FAISS similarity search (Module 4)
│   ├── explainability/          # Grad-CAM (Module 5)
│   └── inference/               # Unified inference engine
├── analytics/                   # KPI calculator + insights (Module 6)
├── dashboard/                   # Django web app (Modules 7-8)
│   ├── models.py                # Database models
│   ├── views.py                 # Page views
│   ├── batch_processor.py       # Batch inspection
│   ├── templates/               # HTML templates
│   └── static/                  # CSS, JS, images
├── scripts/                     # CLI utilities
│   ├── train_model.py           # Train the classifier
│   ├── build_faiss_index.py     # Build similarity index
│   └── generate_sample_data.py  # Seed demo data
├── data/                        # Severstal dataset
├── saved_models/                # Trained model artifacts
├── faiss_index/                 # FAISS index files
├── media/                       # User uploads & Grad-CAM outputs
└── reports/                     # Generated inspection reports
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA-capable GPU for training

### 2. Setup

```bash
# Clone the repository
git clone https://github.com/Sonupraharshan/Smart-Trace-Platfrom.git
cd Smart-Trace-Platfrom

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Initialize Django
python manage.py migrate
```

### 3. Generate Demo Data (Optional)

```bash
# Seed the database with sample inspection records
python scripts/generate_sample_data.py
```

### 4. Train the Model (Optional)

```bash
# Train ResNet50 classifier (~30-60 min on GPU, longer on CPU)
python scripts/train_model.py

# Build FAISS similarity index
python scripts/build_faiss_index.py
```

### 5. Run the Dashboard

```bash
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser.

---

## 📊 Dataset

**Severstal Steel Defect Detection** (Kaggle)

| Property | Value |
|----------|-------|
| Train Images | 12,568 |
| Test Images | 5,506 |
| Image Size | 256 × 1600 px |
| Defect Classes | 4 types + No Defect |
| Annotations | RLE-encoded masks |

Download from: [Kaggle - Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

Place the dataset in `data/severstal-steel-defect-detection/`.

---

## 🖥️ Dashboard Pages

### Executive Dashboard
- KPI cards (inspected, pass rate, defect rate, severity)
- Quality Score & Risk Score gauges
- Defect & severity distribution charts
- Automated insights
- Recent inspections table

### Inspection Workspace
- Single image upload
- Real-time classification + confidence
- Severity estimation
- Grad-CAM explanation overlay
- Class probability distribution

### Root Cause Analysis
- Similar defect retrieval (FAISS top-5)
- Similarity scores
- Grad-CAM comparison
- Historical defect gallery

### Batch Report
- Multi-file batch upload
- Aggregated KPIs (total/pass/fail)
- Defect & severity distribution charts
- Downloadable CSV report

---

## 🧠 ML Pipeline

### Classification (ResNet50)
- Pretrained ImageNet backbone
- Frozen conv1–layer2, fine-tuned layer3–layer4
- Custom FC head (Dropout + Linear)
- Class-weighted CrossEntropy loss
- AdamW with differential learning rates

### Severity Estimation
- Mask-area based: `defect_area_pct = mask_pixels / total_pixels × 100`
- Thresholds: Low (0–5%), Medium (5–15%), High (>15%)
- Inference fallback: Grad-CAM heatmap as proxy mask

### Explainability (Grad-CAM)
- Targets ResNet50 layer4[-1]
- Produces: original, heatmap, overlay
- Explains *why* the model predicted a specific defect

### Similarity Retrieval (FAISS)
- 2048-dim embeddings from ResNet50 avgpool
- IndexFlatIP (cosine similarity)
- Top-k=5 retrieval

---

## 🔧 Configuration

All ML configuration is centralized in `ml_pipeline/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 25 | Maximum training epochs |
| `LEARNING_RATE` | 1e-4 | Backbone learning rate |
| `FC_LEARNING_RATE` | 1e-3 | FC head learning rate |
| `EARLY_STOPPING_PATIENCE` | 5 | Epochs without improvement |
| `INPUT_IMG_SIZE` | 256 | Model input resolution |
| `FAISS_TOP_K` | 5 | Similar images returned |

---

## 📈 Future Enhancements

- [ ] Multi-label classification (images with multiple defect types)
- [ ] U-Net segmentation model for pixel-level defect localization
- [ ] Real-time streaming from industrial cameras
- [ ] Time-series quality trend analysis
- [ ] Role-based authentication (operator / engineer / manager)
- [ ] REST API (Django REST Framework) for IoT integration
- [ ] MLflow experiment tracking
- [ ] ONNX export for edge deployment
- [ ] Email/Slack alerting when quality drops

---

## 📄 License

This project is for educational and portfolio purposes.

---

## 🙏 Acknowledgments

- [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) — Kaggle dataset
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Django](https://www.djangoproject.com/) — Web framework
- [FAISS](https://github.com/facebookresearch/faiss) — Similarity search
- [Grad-CAM](https://arxiv.org/abs/1610.02391) — Visual explanations