# 🛵 Rapido: Intelligent Mobility Insights

**ML-powered ride-hailing analytics dashboard** built on 100,000 booking records from Rapido.

---

## 🌐 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]
> **Deployed on Streamlit Cloud** — Fully interactive dashboard with 4 live ML models.
> Click the badge above or visit:
> `https://rapido-intelligent-mobility-insights-ml-project.streamlit.app`

---

## 🎯 Project Overview

This capstone ML project predicts ride outcomes, fares, cancellation risk, and driver delays using 4 trained models across 15 algorithms — built on 100,000 Rapido ride-hailing booking records.

| # | Model | Type | Algorithm | Score | Benchmark |
|---|-------|------|-----------|-------|-----------|
| 1 | Ride Outcome Prediction | Multi-class Classification | XGBoost | ~87% Accuracy | ✅ Meets |
| 2 | Fare Prediction | Regression | XGBoost | 9.92% RMSE | ✅ Meets |
| 3 | Customer Cancellation Risk | Binary Classification | LightGBM | 92.71% Accuracy | ✅ Exceeds |
| 4 | Driver Delay Prediction | Binary Classification | LightGBM | 83.04% Accuracy | ⚠️ Close |

---

## 📁 Project Structure

```
Rapido_Project/
├── streamlit.py                          # Local Streamlit app
├── streamlit_cloud.py                    # Streamlit Cloud app (Git LFS CSV)
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
├── .gitignore                            # Git ignore rules
├── .streamlit/
│   └── config.toml                       # Streamlit dark orange theme
├── models/                               # Trained ML models (pkl files)
│   ├── model1_ride_outcome_best.pkl
│   ├── model1_scaler.pkl
│   ├── model1_features.pkl
│   ├── model1_label_encoders.pkl
│   ├── model2_fare_prediction_best.pkl
│   ├── model2_scaler.pkl
│   ├── model2_features.pkl
│   ├── model2_label_encoders.pkl
│   ├── model3_customer_cancellation_best.pkl
│   ├── model3_scaler.pkl
│   ├── model3_features.pkl
│   ├── model3_label_encoders.pkl
│   ├── model4_driver_delay_best.pkl
│   ├── model4_scaler.pkl
│   ├── model4_features.pkl
│   └── model4_label_encoders.pkl
├── data/                                 # Raw data CSVs
│   ├── bookings.csv
│   ├── customers.csv
│   ├── drivers.csv
│   ├── location_demand.csv
│   └── time_features.csv
├── master_dataset_engineered.csv         # Final engineered dataset (Git LFS)
├── 01_Data_Cleaning_and_EDA(Rapido).ipynb
├── 02_Feature_Engineering(Rapido).ipynb
├── 03_Model_Training_and_Evaluation(Rapido).ipynb
├── 01_database_normal.py                 # MariaDB normalized schema (5 tables)
└── 02_database_master.py                 # MariaDB master table (ML-ready)
```

---

## ⚙️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/GeekyVishweshNeelesh/Rapido-Intelligent-Mobility-Insights_ML_Project.git
cd Rapido-Intelligent-Mobility-Insights_ML_Project

# 2. Pull large files via Git LFS
git lfs pull

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run streamlit.py
```

---

## ☁️ Streamlit Cloud Deployment

The app is deployed on **Streamlit Cloud** using `streamlit_cloud.py`.

### Steps to Deploy Your Own Instance:
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your forked repo
5. Set **Main file path** → `streamlit_cloud.py`
6. Click **Deploy**

### Requirements for Deployment:
- `models/` folder with all 16 pkl files ✅
- `master_dataset_engineered.csv` pulled via Git LFS ✅
- `requirements.txt` with all dependencies ✅
- `.streamlit/config.toml` for dark orange theme ✅

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.10 | Core language |
| 🤖 Scikit-learn | Preprocessing, Logistic Regression |
| ⚡ XGBoost | Best model for Ride Outcome & Fare |
| 💡 LightGBM | Best model for Cancellation & Delay |
| 🌐 Streamlit | Dashboard & Cloud deployment |
| 📊 Plotly | Interactive visualizations |
| 🗄️ MariaDB | Database (local analysis) |
| 🐙 Git LFS | Large file storage for CSV |

---

## 📊 Features Engineered (115+)

- **Time features** — rush hour flags, peak periods, day/week patterns
- **Customer features** — loyalty score, cancellation rate, value segment
- **Driver features** — reliability score, delay rate, performance tier
- **Interaction features** — pairing quality score, risk factors, surge sensitivity
- **Location features** — pickup/drop demand scores, route popularity

---

## 💼 Business Use Cases

| Use Case | Model | Impact |
|----------|-------|--------|
| Reduce cancellations | Model 3 | Flag high-risk bookings proactively |
| Dynamic fare pricing | Model 2 | Estimate accurate fares before trip |
| Improve driver ETA | Model 4 | Identify delay-prone drivers |
| Ride quality prediction | Model 1 | Predict completion vs cancellation |

---

## 👨‍💻 Author

**VishweshN** — GUVI-HCL Project

[![GitHub](https://img.shields.io/badge/GitHub-GeekyVishweshNeelesh-black?logo=github)](https://github.com/GeekyVishweshNeelesh)
