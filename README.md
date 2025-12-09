```markdown
<!-- PROJECT LOGO -->
<p align="center">
  <img src="https://img.icons8.com/?size=100&id=58722&format=png&color=000000" width="90" />
</p>

<h1 align="center">🏠 Real Estate Investment Advisor  
Predicting Property Profitability & Future Price (5 Years)</h1>

<p align="center">
  <b>ML-powered decision support system for real estate investors</b><br>
  Classification + Regression + Streamlit App + MLflow Tracking
</p>

---

<p align="center">
  <!-- Badges -->
  <img src="https://img.shields.io/badge/Python-3.11-blue" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red" />
  <img src="https://img.shields.io/badge/ML-ScikitLearn%20%7C%20XGBoost-green" />
  <img src="https://img.shields.io/badge/Tracking-MLflow-orange" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" />
</p>

---

# 🚀 Project Overview

This project builds an intelligent **Real Estate Investment Advisor** capable of:

### ✔️ Classification  
**“Is this property a Good Investment?”**

### ✔️ Regression  
**“What will be the estimated price after 5 years?”**

The system uses:
- Cleaned housing data (250,000+ rows)
- Feature-engineered investment metrics
- Trained ML models (RandomForest & XGBoost)
- A Streamlit-based interactive web interface
- MLflow experiment tracking

---

# 🌐 Live Demo (Streamlit Cloud)
👉 **Live App:** *Add your deployed link here*  
👉 **GitHub Repo:** *Add your repo link here*

---

# 🎯 Business Use Cases
- 🏢 **Real Estate Agencies**: Automated valuation & investment scoring  
- 🧾 **Property Portals**: Price forecasting for listings  
- 🧍‍♂️ **Home Buyers**: Decide if the property is worth buying  
- 🏦 **Investors**: Long-term return and risk evaluation  

---

# 🧱 Project Features

### 📌 Machine Learning  
- RandomForest + XGBoost  
- 2 Tasks:  
  - Classification → *Good Investment (0/1)*  
  - Regression → *Future Price (5Y)*  
- MLflow experiment tracking  
- Production feature alignment using `used_feature_list.txt`

### 📊 Streamlit App  
- Property input form  
- Prediction output with confidence  
- Market What-If Simulator  
- EDA dashboard (histograms, trends, correlations)  
- Model performance page  

### 🛠️ Engineering  
- End-to-end pipeline  
- Preprocessing + Feature Engineering  
- Outlier handling  
- One-Hot Encoding  
- Leakage-free training  
- Safe inference using ProductionModel wrapper  

---


# 🗂️ Project Folder Structure

<details>
<summary><b>Click to expand full structure</b></summary>

```text
📦 Real-Estate-Investment-Advisor-ML-model/
├── 📁 data/                # Raw, processed, and external datasets
│   ├── 📁 raw/
│   ├── 📁 processed/
│   └── 📁 external/
├── 📁 deployment/          # Dockerfile, requirements, Procfile
│   ├── Dockerfile
│   ├── requirements.txt
│   └── Procfile
├── 📁 docs/                # Documentation and references
├── 📁 models/              # Saved models, feature importance, etc.
│   ├── 📁 classification/
│   ├── 📁 regression/
│   └── 📁 feature_importance/
├── 📁 notebooks/           # Jupyter notebooks for EDA and prototyping
├── 📁 src/
│   ├── 📁 data_preprocessing/   # Data cleaning, encoding, feature engineering
│   ├── 📁 eda/                  # Exploratory Data Analysis scripts
│   ├── 📁 mlflow_tracking/      # MLflow experiment tracking
│   ├── 📁 models/               # Model training, evaluation, utilities
│   ├── 📁 predictions/          # Prediction service and input schema
│   └── 📁 utils/                # Helper functions, config, constants
├── 📁 streamlit_app/       # Streamlit web application
│   ├── 📁 assets/
│   ├── 📁 pages/
│   └── 📁 utils/
├── 📁 workFlow/            # Workflow scripts
├── README.md
├── LICENSE
├── .gitignore
```
</details>

---

# 🔧 Installation & Setup

### 1️⃣ Create Virtual Environment
```bash
python -m venv .venv
.venv/Scripts/activate
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Preprocess Data

```bash
python -m src.data_preprocessing.run_preprocessing
```

### 4️⃣ Train ML Models

```bash
python -m src.models.train_classification
python -m src.models.train_regression
```

### 5️⃣ Save Production Feature List

```bash
python -m src.models.save_used_features
```

### 6️⃣ Run Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

---

# 📐 Architecture Diagram

```mermaid
flowchart LR
  A[Raw Housing CSV] --> B[Preprocessing & Feature Engineering]
  B --> C[Processed Dataset]
  C --> D1[Classification Training]
  C --> D2[Regression Training]
  D1 --> E[MLflow Tracking]
  D2 --> E
  D1 --> F[Saved Models]
  D2 --> F
  F --> G[ProductionModel (inference.py)]
  G --> H[Streamlit App]
```

---

# 🔄 Data Flow

1. **Raw CSV** → Cleaning, imputations
2. **Feature Engineering** → price per sqft, z-score, investment score
3. **Encoding** (One-Hot)
4. **Leakage Removal**
5. **Model Training** (RF + XGBoost)
6. **Best model saved** to `models/`
7. **Feature list saved** for inference
8. **Streamlit UI** loads ProductionModel → Predicts

---

# 📊 Sample Performance of Project

| Model        | Task           | Train Size | Test Size | Metric   | Value      |
| ------------ | -------------- | ---------- | --------- | -------- | ---------- |
| RandomForest | Classification | 200k       | 50k       | Accuracy | **0.9939** |
| XGBoost      | Classification | 200k       | 50k       | F1       | **0.9865** |
| RandomForest | Regression     | 200k       | 50k       | RMSE     | **13.17**  |
| XGBoost      | Regression     | 200k       | 50k       | RMSE     | **13.18**  |

---



# 🚀 Deploying on Streamlit Cloud

### Step-by-step:

1. Push repo to GitHub
2. Open: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New App**
4. Choose repo → branch: `main`
5. Set *Main file*:

```
src/app/streamlit_app.py
```

6. Add secrets (if any)
7. Deploy 🎉

---

# 🧪 Smoke Tests (Optional But Recommended)

Run:

```python
from src.models.inference import ProductionModel
pm = ProductionModel()
sample = {"BHK": 3, "Size_in_SqFt": 1200, "Price_in_Lakhs": 80}
pm.predict_all(sample)
```

If this works → Streamlit will work.

---

# 🧰 Tech Stack

* 🐍 **Python**
* 📊 **Pandas, NumPy**
* 🤖 **scikit-learn, XGBoost**
* 📈 **MLflow**
* 🖥️ **Streamlit**
* 📉 **Plotly / Matplotlib**
* 🗂️ **Joblib**
* 🧪 **pytest (optional)**

---

# 📚 References

* Streamlit Docs
* MLflow Docs
* Scikit-learn Docs
* XGBoost Docs

---

# 🤝 Contributing

PRs are welcome!
Please open an issue first for significant changes.

---

# 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---




