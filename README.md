# 🔬 Advanced Machine Learning - Iris Dataset Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-green?style=flat-square)](https://github.com)
[![Academic Project](https://img.shields.io/badge/Type-Academic-orange?style=flat-square)](https://github.com)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Features](#project-features)
- [Project Structure](#project-structure)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Module Documentation](#module-documentation)
- [Results & Output](#results--output)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This repository implements comprehensive machine learning algorithms and techniques on the **Iris dataset**, a foundational dataset in machine learning containing 150 samples of iris flowers with four features and three species classifications.

The project demonstrates practical implementations of:
- **Supervised Learning**: Classification using multiple algorithms and validation techniques
- **Unsupervised Learning**: Clustering analysis with density-based methods
- **Linear Algebra**: Discriminant function analysis and feature extraction

All implementations emphasize algorithmic understanding, statistical validation, and visualization-driven insights suitable for academic and professional audiences.

---

## ✨ Project Features

| Feature | Description | Status |
|---------|-------------|--------|
| 📊 **Classification Analysis** | Multiple classifiers (SVM, Random Forest, AdaBoost) with Leave-One-Out CV | ✅ Complete |
| 🎲 **Clustering** | DBSCAN algorithm with parameter optimization and evaluation metrics | ✅ Complete |
| 📐 **Discriminant Functions** | Linear and quadratic discriminant analysis with manual calculations | ✅ Complete |
| 🔄 **Feature Selection** | SelectKBest feature importance and correlation analysis | ✅ Complete |
| 📈 **Comprehensive Evaluation** | Multiple metrics (Accuracy, Precision, Recall, F1, Silhouette Score, etc.) | ✅ Complete |
| 📊 **Data Visualization** | Confusion matrices, ROC curves, clustering visualizations | ✅ Complete |

---

## 📁 Project Structure

```
AML-2026-RigonPira/
│
├── 📄 README.md                    # This file
├── 📄 main.py                      # Project entry point
│
├── 📂 src/                         # Source code modules
│   ├── Classification.py           # Classification algorithms & validation
│   ├── Clustering.py               # DBSCAN clustering analysis
│   ├── DiscriminantFunctions.py    # Discriminant analysis
│   └── create_clean_iris.py        # Data preparation script
│
├── 📂 data/                        # Dataset directory
│   └── iris_clean.data             # Cleaned Iris dataset (CSV format)
│
├── 📂 results/                     # Output and results directory
│   ├── plots                       # Generated visualizations
│    
└── 📂 venv/                        # Python virtual environment

```

### 📊 Workflow Diagram

```
                    Raw Iris Dataset
                           ↓
                  ┌─────────────────┐
                  │ Data Preparation│
                  │(create_clean... │
                  └────────┬────────┘
                           ↓
                ┌──────────────────────┐
                │  Cleaned Dataset     │
                └──┬─────────┬─────────┘
                   │         │
        ┌──────────┘         └──────────┐
        ↓                                ↓
    ┌─────────────┐          ┌──────────────────┐
    │Classification│         │ Clustering (DBSCAN)│
    │  - SVM       │         │ - Parameter Search │
    │  - RF        │         │ - Evaluation      │
    │  - AdaBoost  │         └──────────────────┘
    │  - LOO CV    │
    └──────┬──────┘
           ↓
    ┌─────────────────┐
    │Feature Selection│
    │- SelectKBest   │
    │- Correlation   │
    └──────┬──────────┘
           ↓
    ┌─────────────────────────┐
    │Discriminant Functions   │
    │- LDA Analysis           │
    │- Manual Calculations    │
    └──────────┬──────────────┘
               ↓
        ┌─────────────────┐
        │ Results & Reports│
        │ - Metrics       │
        │ - Visualizations│
        └─────────────────┘
```

---

## 🛠️ Technical Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | v1.7.2 | ML algorithms, metrics, preprocessing |
| **NumPy** | v2.2.6 | Numerical computing & linear algebra |
| **Pandas** | v2.3.3 | Data manipulation & analysis |
| **Matplotlib** | 3.10.8 | Static visualizations |
| **Seaborn** | 0.13.2 | Statistical data visualization |
| **SciPy** | 1.15.3 | Scientific computing (optimization, linear algebra) |

### Development Tools

```python
# Python Version: 3.10.9
# Environment Manager: venv
# Package Manager: pip
```

---

## 📦 Installation

### Prerequisites
- Python 3.10.9 or higher
- pip or conda package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AML-2026-RigonPira.git
   cd AML-2026-RigonPira
   ```

2. **Create a virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   
   # Or using conda
   conda create -n aml-2026 python=3.9
   ```

3. **Activate the virtual environment**
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   
   # Or with conda
   conda activate aml-2026
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Prepare the dataset**
   ```bash
   python src/create_clean_iris.py
   ```

---

## 🚀 Usage

### Running Individual Modules

**Classification Only:**
```bash
python src/Classification.py
```

**Clustering Only:**
```bash
python src/Clustering.py
```

**Discriminant Functions Only:**
```bash
python src/DiscriminantFunctions.py
```

**Generate Clean Dataset:**
```bash
python src/create_clean_iris.py
```

### Expected Output

- **Console Output**: Detailed algorithm execution, metrics, and analysis
- **Generated Files** (in `results/`):
  - Confusion matrices and classification reports
  - Clustering quality metrics
  - Feature importance plots
  - ROC curves and performance visualizations
  - Discriminant function analysis results

---

## 📚 Module Documentation

### 🏷️ Classification.py

**Purpose**: Implements supervised learning classification algorithms with rigorous validation.

**Key Functions**:
- Multiple classifier implementations (SVM, Random Forest, AdaBoost)
- Leave-One-Out Cross-Validation (LOOCV)
- Feature selection using SelectKBest and mutual information
- Comprehensive performance metrics evaluation

**Output Metrics**:
```
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- Cross-validation Scores
- Feature Importance Rankings
```

**Example Analysis**:
```python
# Performed on Iris dataset with:
# - 150 samples, 4 features, 3 classes
# - Leave-One-Out Cross-Validation (LOOCV)
# - Multiple feature selection strategies
```

---

### 🎲 Clustering.py

**Purpose**: Explores unsupervised learning through DBSCAN clustering with comprehensive evaluation.

**Key Features**:
- DBSCAN implementation with parameter grid search (eps, min_samples)
- Optimal parameter selection based on silhouette score
- Multiple evaluation metrics:
  - Silhouette Score
  - Davies-Bouldin Index
  - Adjusted Rand Index
  - Normalized Mutual Information
  - Homogeneity, Completeness, V-Measure

**PCA Visualization**:
```
- 2D and 3D projections of cluster results
- Cluster boundary visualization
- Density distribution analysis
```

---

### 📐 DiscriminantFunctions.py

**Purpose**: Demonstrates linear discriminant analysis (LDA) with manual mathematical calculations.

**Analysis Components**:
- Manual computation of discriminant functions
- Mean vectors and covariance matrices
- Linear and quadratic discriminant boundaries
- Scikit-learn LDA validation
- Mathematical formula verification

**Output**:
- Discriminant function equations
- Threshold calculations
- Classification accuracy on toy dataset

---

### 🧹 create_clean_iris.py

**Purpose**: Data preparation and cleaning script.

**Operations**:
- Loads standard Iris dataset from scikit-learn
- Converts class labels to meaningful names
- Exports to CSV format
- Ensures consistent data formatting

**Output Format**:
```
sepal_length,sepal_width,petal_length,petal_width,class
5.1,3.5,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
...
```

---

## 📊 Results & Output


### Generated Visualizations

✅ Confusion matrices for each classifier  
✅ Feature importance bar plots  
✅ Cluster visualization (2D/3D)  
✅ ROC curves for multi-class classification  
✅ Correlation heatmaps  
✅ Discriminant function boundaries  

---

## 🧑‍💻 Author

Developed by Rigon Pira @UBT

---

## Feedback

If you have any feedback, please reach out to me at **rigon.pira@ubt-uni.net**.

