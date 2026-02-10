## PMSI AI Coding Assistant

A **Deep Learning NLP application** that automates medical coding (PMSI/ICD-10) by analyzing clinical reports. Built with **Bio_ClinicalBERT**, this tool reads raw doctor's notes and predicts the correct diagnostic codes with high confidence.

### Limitations & Disclaimer

* **Experimental Prototype:** This project is an **experimental proof-of-concept** designed for educational purposes.
* **Limited Disease Scope:** The model is currently trained to recognize only the **Top 50** most common medical specialties/diseases from the dataset it is trained on. It does **not** cover the comprehensive list of 70,000+ ICD-10 codes used in real-world hospitals.
* **Data Source:** The model was trained on the **MTSamples** dataset (publicly available on Kaggle). This dataset contains transcribed medical reports from the US and is not able to work the same with other languages like French.

---

### Features

* **Intelligent Prediction:** Uses a fine-tuned **Bio_ClinicalBERT** model to understand medical language.
* **High Accuracy:** Implemented **Weighted Loss (BCEWithLogitsLoss)** to handle extreme data imbalance (e.g., rare diseases vs. common ones), boosting confidence.
* **Explainability (XAI):** Integrated the explainabilty with the famous **SHAP (SHapley Additive exPlanations)** library to visualize exactly *why* the model chose a specific code (highlighting key medical terms).
* **Web App:** Used a **Streamlit** interface for users to test predictions instantly.

---

### Tech Stack

* **Model:** `emilyalsentzer/Bio_ClinicalBERT` (Hugging Face Transformers)
* **Framework:** PyTorch
* **Interface:** Streamlit
* **Explainability:** SHAP
* **Data Handling:** Pandas, NumPy, Scikit-Learn

---

### Dataset

Trained on the **MTSamples** dataset, a collection of ~5,000 transcribed medical reports.
* **Input:** Raw clinical text (e.g., *"Patient presents with severe abdominal pain..."*).
* **Output:** Multi-label classification for 50 medical specialties (Orthopedic, Cardiovascular, etc.).
* **Imbalance Handling:** Rare classes (e.g., 'Hernia') were up-weighted by **50x** during training to ensure balanced detection.

---

## Demo

### 1. The Web Interface
_The app allows users to paste a medical report and get ICD-10 suggestions._

<img src="../images/Screenshot 2026-02-10 223937.png" alt="App Demo" width="600">

### 2. Explainability (SHAP)
_The model highlights the words that influenced its decision (Red = Evidence For, Blue = Evidence Against)._

<img src="../images/Screenshot 2026-02-08 220115.png" alt="App Demo" width="600">

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/PMSI-AI-Coding-Assistant.git](https://github.com/YOUR_USERNAME/PMSI-AI-Coding-Assistant.git)
cd PMSI-AI-Coding-Assistant