# Leaf Disease Detection using Few-Shot Learning

## Project Overview

This research project focuses on developing an advanced plant leaf disease detection system using Few-Shot Learning (FSL) techniques. The project implements Prototypical Networks to classify plant diseases with minimal training data, making it particularly useful for agricultural applications where labeled data is scarce.

## Research Context

This project was developed as part of CSE 299 (Group 4) research, exploring the application of deep learning and few-shot learning techniques in agricultural disease detection. The research addresses the challenge of training robust classification models with limited labeled data, which is common in real-world agricultural scenarios.

## Key Features

- **Few-Shot Learning Implementation**: Uses Prototypical Networks for 5-way 5-shot classification
- **Multiple Backbone Networks**: Implements both EfficientNet and ResNet18 architectures
- **Comprehensive Dataset Support**: Works with PlantVillage dataset and trimmed versions
- **Real-time Disease Classification**: Capable of classifying plant diseases from leaf images
- **Agricultural Focus**: Specifically designed for plant health monitoring

## Methodology

### Few-Shot Learning Approach
The project implements Prototypical Networks, a state-of-the-art few-shot learning method that:
- Learns to represent classes in a metric space
- Requires only a few examples per class (5-shot learning)
- Can generalize to new classes not seen during training
- Uses episodic training with task-based sampling

### Architecture
- **Backbone Networks**: 
  - EfficientNet (efficientnet_pytorch)
  - ResNet18 (torchvision)
- **Few-Shot Framework**: easyfsl library
- **Image Processing**: 64x64 RGB images with data augmentation
- **Training Strategy**: 10,000 training episodes with 1,000 evaluation tasks

## Dataset

### PlantVillage Dataset
The project uses the PlantVillage dataset, which contains images of healthy and diseased plant leaves across multiple species:

#### Background Set (Training)
- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Corn**: Cercospora leaf spot, Common rust, Healthy, Northern Leaf Blight
- **Grape**: Black rot, Esca, Healthy, Leaf blight
- **Potato**: Early blight, Healthy, Late blight
- **Strawberry**: Healthy, Leaf scorch
- **Tomato**: Bacterial spot, Early blight, Healthy, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Tomato mosaic virus, Tomato Yellow Leaf Curl Virus

#### Evaluation Set
- **Blueberry**: Healthy
- **Cherry**: Healthy, Powdery mildew
- **Orange**: Huanglongbing (Citrus greening)
- **Pepper**: Bacterial spot, Healthy
- **Peach**: Bacterial spot, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Healthy

### Trimmed Dataset
A reduced version (PlantVillage-Trim-100) is also available for faster experimentation:
- Background: 2,700 images (400 per major crop)
- Evaluation: 1,200 images (100-200 per evaluation crop)

## Implementation

### Dependencies
```bash
pip install easyfsl
pip install scikit-learn
pip install efficientnet_pytorch
pip install torch torchvision
pip install matplotlib pandas tqdm pillow
```

### Key Components

#### 1. Custom Dataset Class
- Implements VisionDataset for PlantVillage data
- Supports background/evaluation split
- Handles image loading and preprocessing

#### 2. Prototypical Network
- Metric learning-based approach
- Computes prototypes for each class
- Classifies query samples based on distance to prototypes

#### 3. Training Pipeline
- Task-based sampling with TaskSampler
- Episodic training strategy
- Support and query set separation
- Comprehensive evaluation metrics

### Model Configurations

#### 5-Way 5-Shot Setup
- **N_WAY**: 5 classes per task
- **N_SHOT**: 5 support images per class
- **N_QUERY**: 5 query images per class
- **Image Size**: 64x64 pixels
- **Color Mode**: RGB

## Usage

### Training Models

#### EfficientNet Backbone
```python
# Full PlantVillage dataset
python PV-PN-efficientnet-5x5.ipynb

# Trimmed dataset
python PV-Trim-PN-efficientnet-5x5.ipynb
```

#### ResNet18 Backbone
```python
# Full PlantVillage dataset
python PV-PN-resnet18-5x5.ipynb

# Trimmed dataset
python PV-Trim-PN-resnet18-5x5.ipynb
```

### Model Evaluation
The training process automatically evaluates models using:
- Accuracy metrics
- ROC curves and AUC scores
- Confusion matrices
- Classification reports

### Saved Models
Trained models are automatically saved in the `model-saves/` directory with timestamps:
- `EfficientNet/5-way 5-shot plant leaf classification [timestamp]/`
- `ResNET18/5-way 5-shot plant leaf classification [timestamp]/`

## Results and Performance

The project evaluates model performance across different architectures and datasets:
- **EfficientNet**: Generally higher accuracy but slower training
- **ResNet18**: Faster training with competitive performance
- **Full vs. Trimmed Dataset**: Trade-off between performance and training time

## Research Contributions

1. **Few-Shot Learning in Agriculture**: Demonstrates the effectiveness of FSL for plant disease detection
2. **Architecture Comparison**: Systematic evaluation of different backbone networks
3. **Dataset Efficiency**: Shows how to achieve good performance with limited labeled data
4. **Real-world Applicability**: Addresses practical challenges in agricultural AI

## Future Work

- **Multi-modal Learning**: Incorporating additional data sources (text, sensor data)
- **Transfer Learning**: Leveraging pre-trained models for better performance
- **Real-time Deployment**: Optimizing for edge devices and mobile applications
- **Additional Crops**: Expanding to more plant species and diseases

## Citation

If you use this work in your research, please cite:
```
CSE 299 Group 4. "Leaf Disease Detection using Few-Shot Learning." 
[Your Institution], [Year].
```

## Contact

For questions or collaboration opportunities, please contact the research team.

## License

This project is for research purposes. Please ensure compliance with the PlantVillage dataset license and any other relevant licenses.

---

**Note**: This project requires significant computational resources for training. Consider using GPU acceleration for optimal performance.
