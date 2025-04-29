# 🌸 Iris Flower Classification

## 📌 Objective
Develop a classification model to accurately identify Iris flowers into one of three species based on their sepal and petal dimensions.

---

## 📊 Dataset Overview
- **Source**: `IRIS.csv`
- **Records**: 150 flower samples
- **Features**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target**: Species (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`)

---

## 🧪 Workflow

### 1. Data Preprocessing
- Label Encoding of species
- Feature scaling using `StandardScaler`

### 2. Model Training
- Algorithm: `RandomForestClassifier`
- Train/test split: 80/20
- Evaluated with:
  - Classification Report
  - Confusion Matrix

### 3. Feature Importance
- Extracted from Random Forest
- Visualized with a bar chart

---

## 📈 Results
- Achieved high accuracy on the test set
- Most important features:
  - Petal Length
  - Petal Width

---

## 🚀 Run the Project

### Step 1: Clone the repo
```bash
git clone https://github.com/your-username/iris-classifier.git
cd iris-classifier
