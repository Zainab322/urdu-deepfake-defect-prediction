# urdu-deepfake-defect-prediction
End-to-end ML project for Urdu Deepfake Audio Detection and Multi-Label Software Defect Prediction, featuring classical ML, deep learning, and a real-time Streamlit app for predictions. Full workflow from preprocessing to evaluation.

---

## 1️⃣ Urdu Deepfake Audio Detection (Binary Classification)

### 🎯 Objective
Classify an Urdu audio file as **Bonafide** or **Deepfake**.

### 📊 Dataset
- **Source:** [`CSALT/deepfake_detection_dataset_urdu`](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu) (via Hugging Face Datasets).
- Audio format: `.wav`
- Labels: `bonafide` (real), `deepfake` (synthetic)

### 🛠 Preprocessing
- Extracted **MFCCs** and **Spectrograms**
- Normalized feature length (padding/truncation)
- Standard scaling before feeding to models

### 🤖 Models Used
| Model | Key Features |
|-------|--------------|
| SVM | RBF kernel, tuned `C` & `gamma` |
| Logistic Regression | One-vs-Rest, `liblinear` solver |
| Perceptron | Single-layer, no hidden layers |
| DNN | ≥2 Dense hidden layers, ReLU activations, dropout regularization |

### 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
---

## 2️⃣ Multi-Label Software Defect Prediction

### 🎯 Objective
Predict **multiple defect labels** for a software component based on feature vectors.

### 📊 Dataset
- **Format:** CSV file (provided with assignment)
- Preprocessing steps:
  - Missing value handling
  - Feature selection
  - Min-Max scaling
- Label distribution analysis (imbalance handling considered)

### 🛠 Models Used
| Model | Multi-Label Strategy |
|-------|----------------------|
| Logistic Regression | One-vs-Rest |
| SVM | Multi-label adaptation |
| Perceptron | Online learning via `partial_fit` |
| DNN | Multi-label sigmoid output layer |

### 📈 Evaluation Metrics
- Hamming Loss
- Micro-F1, Macro-F1
- Precision@k
---

## 3️⃣ Interactive Streamlit App

### 🎯 Objective
Provide a **real-time, user-friendly UI** to:
- Upload an audio file → Predict Bonafide vs Deepfake
- Input feature vector for defect data → Predict multiple labels

### 🖥 Features
- **Model selection at runtime** (SVM / Logistic Regression / DNN)
- Confidence score display
- Clean, intuitive layout

### 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run "03_Interactive Streamlit App/app.py"

📜 How to Reproduce Training
Part 1: Train Deepfake Detection
python "01_Urdu Deepfake Audio Detection (Binary Classification)/train.py" --model dnn

Part 2: Train Multi-Label Defect Prediction
python "02_Multi-Label Defect Prediction (Multi-Label Classification)/train.py" --model perceptron --online

📖 Learn More
For detailed explanation of methodology, challenges, and insights, read my Medium blog:
Cracking the Code of Deepfake Audio & Software Defect Detection
