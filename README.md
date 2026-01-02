# CNN for CIFAR-10 with Hyperparameter Ablation Study

A comprehensive deep learning project implementing a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset, featuring an extensive hyperparameter ablation study to identify optimal model configurations.

---


## Overview

This project presents a complete CNN analysis pipeline for the CIFAR-10 image classification task. The implementation features:

- **Adaptive CNN Architecture**: Flexible model that can adjust depth (layers) and capacity (filters) dynamically
- **Comprehensive Ablation Study**: Systematic evaluation of hyperparameters including learning rate, batch size, filter count, and network depth
- **Performance Optimization**: Comparison between baseline and optimized model configurations
- **Detailed Visualizations**: Training curves, confusion matrices, feature maps, and comparative analysis

The goal is to demonstrate the impact of different hyperparameters on model performance and identify the optimal configuration for CIFAR-10 classification.

---

## Dataset

**CIFAR-10** is a widely-used benchmark dataset for image classification containing:

| Attribute | Value |
|-----------|-------|
| Training Images | 50,000 |
| Test Images | 10,000 |
| Image Dimensions | 32 × 32 × 3 (RGB) |
| Number of Classes | 10 |

### Classes
| ID | Class Name |
|----|------------|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

The dataset is loaded directly from Hugging Face for easy access and preprocessing.

---

## Architecture

### Adaptive CNN Model (`CIFAR10CNN_Adaptive`)

The model uses a flexible architecture that adapts based on two key parameters:

```
Input (3 × 32 × 32)
        ↓
┌─────────────────────────────────────┐
│   Convolutional Block × N layers    │
│   • Conv2D (3×3, padding=1)         │
│   • Batch Normalization             │
│   • ReLU Activation                 │
│   • MaxPool2D (every 2 layers)      │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Adaptive Average Pooling (4×4)    │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Fully Connected Layer (→ 512)     │
│   • ReLU + Dropout (0.5)            │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Output Layer (512 → 10 classes)   │
└─────────────────────────────────────┘
```

### Key Architecture Features

| Feature | Description |
|---------|-------------|
| **Adaptive Pooling** | Ensures consistent output dimensions regardless of network depth |
| **Batch Normalization** | Stabilizes training and allows higher learning rates |
| **Progressive Filter Growth** | Filters double with each layer (32 → 64 → 128 → ...) |
| **Dropout Regularization** | 50% dropout to prevent overfitting |

---

## Features

- **Complete Pipeline**: Data loading, preprocessing, training, evaluation, and visualization
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Progress Tracking**: Real-time training progress with tqdm
- **Comprehensive Metrics**: Accuracy, Precision, Recall, and F1-Score
- **Ablation Study**: Systematic hyperparameter exploration
- **Feature Visualization**: Visual analysis of learned feature maps
- **Confusion Matrix Analysis**: Detailed per-class performance evaluation
- **Model Comparison**: Side-by-side baseline vs. optimized model analysis

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/anasbinrashid/CNN-for-CIFAR10-Hyperparameter-AblationStudy.git
cd CNN-for-CIFAR10-Hyperparameter-AblationStudy
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn datasets tqdm
```

---

## Usage

### Running the Notebook

1. Open Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```

2. Navigate to and open `CNN for CIFAR10.ipynb`

3. Run all cells to execute the complete analysis pipeline

### Running on Kaggle

This notebook was developed on Kaggle with GPU acceleration. To run on Kaggle:
1. Create a new notebook on Kaggle
2. Enable GPU acceleration in Settings
3. Upload or paste the notebook content
4. Run all cells

---

## Experiments & Results

### Baseline Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 3 layers, 32 filters |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Optimizer | Adam |
| Training Epochs | 25 |

### Baseline Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 81.88% |
| **Precision** | 81.89% |
| **Recall** | 81.88% |
| **F1-Score** | 81.74% |

### Hyperparameter Ablation Study

#### Learning Rate Impact
| Learning Rate | Accuracy | Observation |
|---------------|----------|-------------|
| 0.001 | ~73% | Stable convergence |
| 0.01 | ~63% | Faster but less stable |
| 0.1 | ~10% | Training failure (too high) |

#### Batch Size Impact
| Batch Size | Accuracy | Observation |
|------------|----------|-------------|
| 16 | ~76% | More variance, better generalization |
| 32 | ~77% | Balanced performance |
| 64 | ~76% | Stable gradients |

#### Number of Filters Impact
| Starting Filters | Accuracy | Observation |
|------------------|----------|-------------|
| 16 | ~72% | Limited capacity |
| 32 | ~77% | Balanced |
| **64** | **~79%** | **Best performance** |

#### Network Depth Impact
| Layers | Accuracy | Observation |
|--------|----------|-------------|
| 3 | ~77% | Fast training |
| **5** | **~80%** | **Optimal depth** |
| 7 | ~78% | Diminishing returns |

### Optimal Configuration Identified

| Parameter | Optimal Value |
|-----------|---------------|
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Number of Filters | 64 |
| Number of Layers | 5 |

---

## Key Findings

### 1. Learning Rate Sensitivity
> A learning rate of 0.1 caused complete training failure, demonstrating the critical importance of appropriate learning rate selection. The model converged to random guessing (~10% accuracy on 10 classes).

### 2. Depth vs. Performance Trade-off
> 5 layers provided the best balance between model capacity and training stability. Deeper networks (7 layers) showed diminishing returns due to increased optimization difficulty.

### 3. Filter Count Correlation
> Model performance consistently improved with more filters (16 → 32 → 64), indicating that CIFAR-10 benefits from higher model capacity.

### 4. Batch Size Stability
> Batch sizes between 16-64 performed similarly, with 32 providing a good balance between training speed and gradient quality.

### 5. Adaptive Pooling Benefits
> The adaptive average pooling layer was crucial for:
> - Maintaining consistent output dimensions across different depths
> - Preventing vanishing gradients in deeper configurations
> - Enabling flexible architecture experimentation

---

## Visualizations

The notebook generates comprehensive visualizations including:

1. **Training Progress**
   - Loss vs. Epochs curves
   - Accuracy vs. Epochs curves
   - Normalized training progress correlation

2. **Confusion Matrices**
   - Baseline model confusion matrix
   - Optimized model confusion matrix
   - Improvement matrix (difference analysis)

3. **Ablation Study Charts**
   - Learning rate impact bar chart
   - Batch size impact bar chart
   - Filter count impact bar chart
   - Network depth impact bar chart

4. **Feature Map Visualization**
   - Layer-by-layer feature activation maps
   - Visual representation of learned features

5. **Per-Class Analysis**
   - Class-wise accuracy comparison
   - Baseline vs. Optimized model per-class breakdown

---

## Project Structure

```
CNN-for-CIFAR10-Hyperparameter-AblationStudy/
│
├── CNN for CIFAR10.ipynb    # Main notebook with complete analysis
├── data/                    # Dataset directory
└── README.md                # Project documentation
```

---

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 2.0 |
| torchvision | ≥ 0.15 |
| numpy | ≥ 1.21 |
| pandas | ≥ 1.3 |
| matplotlib | ≥ 3.4 |
| seaborn | ≥ 0.11 |
| scikit-learn | ≥ 1.0 |
| datasets | ≥ 2.0 |
| tqdm | ≥ 4.62 |

### Hardware Requirements
- **Recommended**: GPU with CUDA support (e.g., NVIDIA GTX 1060 or better)
- **Minimum**: CPU (training will be significantly slower)

---

## Theoretical Background

### Why These Hyperparameters Matter

| Hyperparameter | Low Value Effect | High Value Effect |
|----------------|------------------|-------------------|
| **Learning Rate** | Slow but stable convergence | Fast but unstable, risk of divergence |
| **Batch Size** | Noisier gradients, better generalization | Smoother gradients, may overfit |
| **Filters** | Limited feature extraction | Rich feature representation |
| **Layers** | Simple features only | Complex hierarchical features |

### CNN Feature Hierarchy
- **Early Layers**: Detect low-level features (edges, corners, textures)
- **Middle Layers**: Combine features into patterns (shapes, parts)
- **Later Layers**: High-level semantic features (object parts, class-specific)

---

## Acknowledgments

- **CIFAR-10 Dataset**: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **PyTorch**: Facebook AI Research team
- **Hugging Face Datasets**: For easy dataset access
- **Kaggle**: For providing GPU computing resources

---

## Contact

For questions or feedback about this project, please open an issue on the GitHub repository.

---
