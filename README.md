# Real Estate Investment Advisor ML Model

This project provides a machine learning-powered advisor for real estate investment decisions. It includes data preprocessing, model training, MLflow tracking, and a Streamlit web app for user interaction.

## Features
- Data preprocessing and EDA
- Classification and regression models
- MLflow experiment tracking
- Streamlit app for predictions and analysis
- Docker deployment support

## Getting Started
1. Install dependencies from `deployment/requirements.txt`
2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app/Home.py
   ```
3. For Docker:
   ```bash
   docker build -t real-estate-advisor .
   docker run -p 8501:8501 real-estate-advisor
   ```

## Project Structure
- `src/` - Source code
- `data/` - Data files
- `models/` - Model files
- `streamlit_app/` - Streamlit web app
- `deployment/` - Deployment files

## License
See `LICENSE` for details.
