
<div align="center">
  <img src="" width="90" />
  <h1>🏠 Real Estate Investment Advisor</h1>
  <h3>Predicting Property Profitability & Future Price (5 Years)</h3>
  <p><b>ML-powered decision support system for real estate investors</b></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue" />
    <img src="https://img.shields.io/badge/Framework-Streamlit-red" />
    <img src="https://img.shields.io/badge/ML-ScikitLearn%20%7C%20XGBoost-green" />
    <img src="https://img.shields.io/badge/Tracking-MLflow-orange" />
    <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" />
  </p>
</div>

---

<div align="center">
<b>Classification & Regression | Streamlit App | MLflow Tracking | Docker Ready</b>
</div>

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/SantoshGorebal007/Real-Estate-Investment-Advisor-ML-model.git
cd Real-Estate-Investment-Advisor-ML-model

# 2. Install dependencies
pip install -r deployment/requirements.txt

# 3. Preprocess data & train models
python -m src.data_preprocessing.run_preprocessing
python -m src.models.train_classification
python -m src.models.train_regression
python -m src.models.save_used_features

# 4. Launch the Streamlit app
streamlit run streamlit_app/Home.py
```

---

## 📂 Project Structure

<details>
<summary><b>Click to expand full structure</b></summary>

```text
📦 Real-Estate-Investment-Advisor-ML-model/
├── data/                # Raw, processed, and external datasets
│   ├── raw/
│   ├── processed/
│   └── external/
├── deployment/          # Dockerfile, requirements, Procfile
├── docs/                # Documentation and references
├── models/              # Saved models, feature importance, etc.
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── src/
│   ├── data_preprocessing/   # Data cleaning, encoding, feature engineering
│   ├── eda/                  # Exploratory Data Analysis scripts
│   ├── mlflow_tracking/      # MLflow experiment tracking
│   ├── models/               # Model training, evaluation, utilities
│   ├── predictions/          # Prediction service and input schema
│   └── utils/                # Helper functions, config, constants
├── streamlit_app/       # Streamlit web application
├── workFlow/            # Workflow scripts
├── README.md
├── LICENSE
├── .gitignore
```
</details>

---

## 🎯 Business Use Cases

- 🏢 Real Estate Agencies: Automated valuation & investment scoring
- 🧾 Property Portals: Price forecasting for listings
- 🧍‍♂️ Home Buyers: Decide if the property is worth buying
- 🏦 Investors: Long-term return and risk evaluation

---

## 🧱 Features

### 🤖 Machine Learning
- RandomForest & XGBoost (Classification & Regression)
- MLflow experiment tracking
- Production feature alignment

### 📊 Streamlit App
- Property input form, prediction output, market what-if simulator
- EDA dashboard (histograms, trends, correlations)
- Model performance page

### 🛠️ Engineering
- End-to-end pipeline: preprocessing, feature engineering, outlier handling, encoding, leakage-free training
- Safe inference using ProductionModel wrapper

---

## 🛠️ Installation & Setup (Detailed)

1. **Create Virtual Environment**
    ```bash
    python -m venv .venv
    .venv/Scripts/activate
    ```
2. **Install Dependencies**
    ```bash
    pip install -r deployment/requirements.txt
    ```
3. **Preprocess Data**
    ```bash
    python -m src.data_preprocessing.run_preprocessing
    ```
4. **Train ML Models**
    ```bash
    python -m src.models.train_classification
    python -m src.models.train_regression
    ```
5. **Save Production Feature List**
    ```bash
    python -m src.models.save_used_features
    ```
6. **Run Streamlit App**
    ```bash
    streamlit run streamlit_app/Home.py
    ```

---

## 📐 Architecture Diagram

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

## 🔄 Data Flow

1. **Raw CSV** → Cleaning, imputations
2. **Feature Engineering** → price per sqft, z-score, investment score
3. **Encoding** (One-Hot)
4. **Leakage Removal**
5. **Model Training** (RF + XGBoost)
6. **Best model saved** to `models/`
7. **Feature list saved** for inference
8. **Streamlit UI** loads ProductionModel → Predicts

---

## 📊 Sample Performance

| Model        | Task           | Train Size | Test Size | Metric   | Value      |
| ------------ | -------------- | ---------- | --------- | -------- | ---------- |
| RandomForest | Classification | 200k       | 50k       | Accuracy | **0.9939** |
| XGBoost      | Classification | 200k       | 50k       | F1       | **0.9865** |
| RandomForest | Regression     | 200k       | 50k       | RMSE     | **13.17**  |
| XGBoost      | Regression     | 200k       | 50k       | RMSE     | **13.18**  |

---

## 🚀 Deploying on Streamlit Cloud

1. Push repo to GitHub
2. Open: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New App**
4. Choose repo → branch: `main`
5. Set *Main file*:
    ```
    streamlit_app/Home.py
    ```
6. Add secrets (if any)
7. Deploy 🎉

---

## 🧪 Smoke Test (Optional)

```python
from src.models.inference import ProductionModel
pm = ProductionModel()
sample = {"BHK": 3, "Size_in_SqFt": 1200, "Price_in_Lakhs": 80}
pm.predict_all(sample)
```

If this works → Streamlit will work.

---

## 🧰 Tech Stack

| 🐍 Python | 📊 Pandas, NumPy | 🤖 scikit-learn, XGBoost | 📈 MLflow | 🖥️ Streamlit | 📉 Plotly/Matplotlib | 🗂️ Joblib |
|-----------|------------------|-------------------------|----------|-------------|---------------------|-----------|

---

## 📚 References

- Streamlit Docs
- MLflow Docs
- Scikit-learn Docs
- XGBoost Docs

---

## 🤝 Contributing

PRs are welcome! Please open an issue first for significant changes.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---




