# 🌿 Plant Leaf Disease Detection via Episodic Few-Shot Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Few-Shot Learning](https://img.shields.io/badge/FSL-Prototypical--Networks-2B5B2E?style=flat)](#mathematical-background)

---

An agricultural deep learning system that utilizes metric-based **Few-Shot Learning (FSL)** to classify leaf diseases from extremely scarce data volumes. Built on top of **Prototypical Networks (ProtoNets)**, the system is designed to classify plant anomalies using only 5 reference images (5-Shot) per crop type.

</div>

---

## 📈 Notebooks & Google Colab Demos

We provide distinct training files for the **ResNet18** and **EfficientNet** feature-extraction backbones, tuned for both the full and trimmed versions of the PlantVillage dataset. Run them directly in your browser:

| Backbone Model | Dataset Type | Colab Launch Link |
| :--- | :--- | :--- |
| **EfficientNet** | Full PlantVillage | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TashinMahmud/leaf-disease-detection-fsl/blob/master/Prototypical-Network/PV-PN-efficientnet-5x5.ipynb) |
| **EfficientNet** | Trimmed (100 imgs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TashinMahmud/leaf-disease-detection-fsl/blob/master/Prototypical-Network/PV-Trim-PN-efficientnet-5x5.ipynb) |
| **ResNet18** | Full PlantVillage | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TashinMahmud/leaf-disease-detection-fsl/blob/master/Prototypical-Network/PV-PN-resnet18-5x5.ipynb) |
| **ResNet18** | Trimmed (100 imgs) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TashinMahmud/leaf-disease-detection-fsl/blob/master/Prototypical-Network/PV-Trim-PN-resnet18-5x5.ipynb) |

---

## 🔬 Mathematical Background

Prototypical Networks map samples from support sets ($S$) and query sets ($Q$) into a shared embedding space using a convolutional neural network feature extractor $f_\theta$.

### 1. Prototype Computation
For each class $k \in K$, we calculate a prototype vector $c_k$ representing the mean vector of the embedded support points belonging to that class:
$$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\theta(x_i)$$

### 2. Distance Classification
Query images $x$ are classified by performing a softmax over the negative distance to the computed prototypes. The probability distribution is defined by:
$$p_\theta(y = k \mid x) = \frac{\exp(-d(f_\theta(x), c_k))}{\sum_{k'} \exp(-d(f_\theta(x), c_{k'}))}$$
*Where $d(\cdot, \cdot)$ is the squared Euclidean distance.*

---

## 📂 Research Documentation & Slides

Literature reviews, thesis reports, and presentations are organized inside the project:

*   **[`report/CSE_299_Grp_4_Report.pdf`](report/CSE_299_Grp_4_Report.pdf)**: Complete research paper describing empirical results and loss convergence benchmarks.
*   **[`references/Prototypical Networks for Few-shot Learning.pptx`](references/Prototypical%20Networks%20for%20Few-shot%20Learning.pptx)**: Slides summarizing the ProtoNet model paper.
*   **[`references/omniglot_train_5_way_1shot_main.pdf`](references/omniglot_train_5_way_1shot_main.pdf)**: Reference implementation notes for Omniglot classification.

---

## 🛠️ Getting Started

### 1. Prerequisites
Install all dependencies for running the training tasks:
```bash
pip install easyfsl scikit-learn efficientnet_pytorch torch torchvision matplotlib pandas tqdm pillow jupyter
```

### 2. Run Notebooks locally
To start training and evaluations:
```bash
cd Prototypical-Network/
jupyter notebook
```
Open any of the `.ipynb` files to review the training epochs.

---

## 🧭 Project Layout

```
leaf-disease-detection-fsl/
├── data/                                  # PlantVillage crop images datasets
├── Prototypical-Network/                  # Jupyter notebooks for training models
│   ├── PV-PN-efficientnet-5x5.ipynb       # EfficientNet model (Full)
│   ├── PV-Trim-PN-efficientnet-5x5.ipynb  # EfficientNet model (Trimmed)
│   ├── PV-PN-resnet18-5x5.ipynb           # ResNet18 model (Full)
│   └── PV-Trim-PN-resnet18-5x5.ipynb      # ResNet18 model (Trimmed)
├── report/                                # Main thesis reports and SVG diagrams
├── references/                            # Clutter-free reference research papers and slides
└── README.md                              # Main project documentation
```

---

## 👥 Contributors

*   **CSE 299 Group 4 Research Team** (Md. Tasnimul Hasan, Tashin Mahmud Khan, and colleagues).
