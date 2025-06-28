# 🩺 Breast Cancer Classification using Neural Networks

This project implements a neural network to classify breast cancer tumors as malignant or benign using the Wisconsin Breast Cancer Diagnostic dataset. The model achieves high accuracy in predicting tumor malignancy based on 30 computed features from digitized images.

---

## Dataset

The dataset used is from `sklearn.datasets.load_breast_cancer()` consisting of 569 samples with 30 features each. Key features include:

| Feature Category       | Examples                          |
|------------------------|-----------------------------------|
| Mean Values            | radius, texture, perimeter, area  |
| Standard Errors        | smoothness error, concavity error |
| Worst Values           | worst symmetry, worst concavity   |

Target variable:
- 0: Malignant
- 1: Benign

Dataset characteristics:
- 212 Malignant cases
- 357 Benign cases
- No missing values

---

## Libraries Used

- `numpy` – Numerical operations  
- `pandas` – Data manipulation  
- `matplotlib` – Visualization  
- `scikit-learn` – Data splitting and preprocessing  
- `tensorflow/keras` – Neural network implementation  

---

## Workflow

### 1. Data Loading & Exploration
- Load dataset from sklearn
- Convert to pandas DataFrame
- Check class distribution (212 malignant vs 357 benign)

### 2. Data Preprocessing
- Separate features (X) and target (y)
- Split into train/test sets (80/20 ratio)
- Standardize features using StandardScaler

### 3. Neural Network Architecture
- Input layer: 30 neurons (for 30 features)
- Hidden layers: 
  - Dense(20, activation='relu')
  - Dense(10, activation='relu')
- Output layer: Dense(2, activation='sigmoid')

### 4. Model Training
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 20
- Validation split: 10% of training data

### 5. Evaluation
- Test accuracy: ~95.6%
- Visualization of training vs validation accuracy/loss

### 6. Prediction System
- Takes input feature values
- Predicts malignancy with probability

---

## Model Performance

**Training Results:**
- Final Training Accuracy: 98.3%
- Validation Accuracy: 97.8%
- Test Accuracy: 95.6%

---

## Example Prediction

**Input Features:** 
```
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
```
**Output:** 
```
[[0.82823884 0.04604259]]
[np.int64(0)]
The breast cancer is Malignant
```

---

## How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/breast-cancer-classification.git
cd breast-cancer-classification
```
2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Run the Notebook**
```bash
jupyter notebook Breast_Cancer_Classification_NN.ipynb
```
Play around and make changes trying different inputs and predictions from the dataset! 
