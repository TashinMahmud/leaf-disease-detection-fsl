# 🌿 Leaf Disease Detection using Few-Shot Learning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![EfficientNet](https://img.shields.io/badge/EfficientNet-Google-blue?style=for-the-badge&logo=google)](https://github.com/lukemelas/EfficientNet-PyTorch)
[![Few Shot Learning](https://img.shields.io/badge/FSL-Prototypical%20Networks-green?style=for-the-badge&logo=wikipedia)](https://en.wikipedia.org/wiki/Few-shot_learning)

A research-driven deep learning system built to detect plant leaf diseases using **Few-Shot Learning (FSL)**. This repository implements **Prototypical Networks (ProtoNets)**, enabling plant leaf disease classification with extremely minimal training examples. This approach addresses real-world agricultural scenarios where labeled dataset availability is scarce.

* **Few-Shot Architecture**: Implements 5-Way 5-Shot classification configurations using episodic task sampler strategies.
* **Dual Backbones**: Integrated with both ResNet18 and EfficientNet neural networks.

---

## 🏗️ ProtoNet Model Pipeline

The prototypical network maps support and query sets into a shared embedding space to perform metric-based classification.

```
                      [ Plant Leaf Image ]
                               │
               [ Backbone (ResNet18 / EfficientNet) ]
                               │
              [ Prototypes Computation (Support Set) ]
                               │
             [ Distance Metric (Euclidean Distance) ]
                               │
                  [ Query Class Assignment ]
```

### Episodic Training Setup
1. **Support Set**: A few sample images (5-shot) per class (5-way) used to compute the mean prototype representation of each class.
2. **Query Set**: Target leaf images classified based on the minimum Euclidean distance to class prototypes.

---

## ⚡ Tech Stack & Core Libraries

* **Few-Shot Framework**: [easyfsl](https://github.com/sicara/easy-few-shot-learning) — PyTorch library for few-shot learning research.
* **Deep Learning Engine**: PyTorch & Torchvision.
* **Feature Extractors**: EfficientNet (efficientnet-pytorch) & ResNet18.
* **Data Preprocessing**: PIL, Scikit-Learn, Pandas, NumPy.
* **Visualization**: Matplotlib, Seaborn, Tqdm.

---

## 📊 Dataset Configurations

### PlantVillage Dataset
Features leaf images categorized under multiple crop species (healthy and diseased states):
* **Training Set (Background)**: Apple, Corn, Grape, Potato, Strawberry, Tomato (healthy and common diseases).
* **Evaluation Set**: Blueberry, Cherry, Orange (Citrus greening), Pepper, Peach, Raspberry, Soybean, Squash.

### Trimmed Dataset (PlantVillage-Trim-100)
A specialized dataset subset created for rapid verification cycles:
* **Background Set**: 2,700 images (400 per major crop class).
* **Evaluation Set**: 1,200 images (100–200 per evaluation class).

---

## 🚀 Quick Start Guide

### 1. Installation
Install all training and modeling dependencies:
```bash
pip install easyfsl scikit-learn efficientnet_pytorch torch torchvision matplotlib pandas tqdm pillow
```

### 2. Model Training Configurations

#### EfficientNet Backbone
Run the respective Jupyter notebooks for training models with EfficientNet feature extractors:
```bash
# Full PlantVillage dataset
jupyter notebook PV-PN-efficientnet-5x5.ipynb

# Trimmed dataset
jupyter notebook PV-Trim-PN-efficientnet-5x5.ipynb
```

#### ResNet18 Backbone
Run the respective notebooks for ResNet18 backbones:
```bash
# Full PlantVillage dataset
jupyter notebook PV-PN-resnet18-5x5.ipynb

# Trimmed dataset
jupyter notebook PV-Trim-PN-resnet18-5x5.ipynb
```

---

## 🧭 Project Layout

```
leaf-disease-detection-fsl/
├── data/                                      # Full and trimmed datasets
├── model-saves/                               # Output weights and training checkpoints
│   ├── EfficientNet/                          # EfficientNet run checkpoints
│   └── ResNET18/                              # ResNet18 run checkpoints
├── report/                                    # Thesis documentation, flowcharts, resource assets
├── Prototypical-Network/                      # Source code scripts for network models
├── PV-PN-efficientnet-5x5.ipynb               # EfficientNet notebook (full data)
├── PV-Trim-PN-efficientnet-5x5.ipynb          # EfficientNet notebook (trimmed data)
├── PV-PN-resnet18-5x5.ipynb                   # ResNet18 notebook (full data)
├── PV-Trim-PN-resnet18-5x5.ipynb              # ResNet18 notebook (trimmed data)
└── README.md                                  # Project documentation
```

---

## 📝 Citation & Research Context

Developed as part of CSE 299 (Group 4) research project exploring few-shot visual classification in agricultural domains. If you reference this work, please use the following citation:

```
CSE 299 Group 4. "Leaf Disease Detection using Few-Shot Learning." 2026.
```
