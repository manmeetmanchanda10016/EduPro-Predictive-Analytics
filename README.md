# 🎓 EduPro – Machine Learning-Based Predictive Analytics Platform

<p align="center">

**An Interactive Machine Learning Application for Educational Analytics and Forecasting**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Application-red?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Deployed-success)](https://edupro-predictive-analytics-n6ewzdgssbu2gvmzwz4cge.streamlit.app/)

</p>

## 🌐 Live Application

**Try EduPro here:**

🔗 https://edupro-predictive-analytics-n6ewzdgssbu2gvmzwz4cge.streamlit.app/

## 📌 Project Overview

EduPro is a Machine Learning-Based Predictive Analytics Platform designed to analyze educational platform data and forecast future course enrollments and revenue.

Educational platforms generate large amounts of information through course enrollments, transactions, course details, and user activity. EduPro transforms this historical information into meaningful insights using data preprocessing, feature engineering, and machine learning.

The application provides an interactive dashboard where users can explore historical performance, select courses and categories, and generate multi-month forecasts.

## 🎯 Project Objectives

- Analyze historical educational platform data.
- Identify enrollment and revenue trends.
- Predict future course enrollments.
- Forecast revenue at course and category levels.
- Compare machine learning model performance.
- Provide interactive visualizations through Streamlit.
- Enable users to download forecast results.

## 📊 Dataset Description

The project uses four datasets:

| Dataset | Description |
|---|---|
| Users | User information and demographic details |
| Teachers | Teacher profiles, expertise, experience and ratings |
| Courses | Course categories, prices, duration, levels and ratings |
| Transactions | Enrollment transactions, payment information and amounts |

### Data Preprocessing

The following steps were performed:

- Data validation and consistency checks.
- Missing-value and duplicate detection.
- Foreign-key relationship validation.
- Merging relevant datasets.
- Monthly aggregation at course and category levels.
- Feature engineering using lag and rolling-window features.
- Chronological train-test splitting.

### Feature Engineering

Important features include:

- Enrollment lag features.
- Revenue lag features.
- Rolling averages.
- Month and quarter.
- Month sine and cosine transformations.
- Course metadata and category information.

A chronological split was used to preserve the time-series ordering of observations and evaluate predictions on later periods.

## 🤖 Machine Learning Models

EduPro uses three regression models for different prediction tasks.

| Prediction Task | Algorithm |
|---|---|
| Course Enrollment | Ridge Regression |
| Course Revenue | Random Forest Regressor |
| Category Revenue | Gradient Boosting Regressor |

### Model Evaluation

The models were evaluated using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

### Evaluation Results

The following metrics were obtained on the original held-out test set (October–December 2025).

| Prediction Task | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Course Enrollment | Ridge Regression | 2.81 | 3.69 | -0.016 |
| Course Revenue | Random Forest | 247.82 | 619.06 | 0.920 |
| Category Revenue | Gradient Boosting | 1016.56 | 1529.15 | 0.936 |

**Note:** The enrollment model showed weaker performance compared with the revenue models. Its results are close to the mean baseline and should be interpreted cautiously.

The saved final models were subsequently refitted using the available engineered observations. The metrics above represent the original held-out evaluation, not a new evaluation of the refitted models or a validation of future multi-month forecasts.

## 📈 Application Features

### 1. Overview Dashboard

- Historical enrollment statistics.
- Historical revenue statistics.
- Total courses and categories.
- Monthly enrollment trends.
- Monthly revenue trends.
- Forecast summaries.
- Revenue distribution by category.
- Top-performing courses.

### 2. Course Enrollment Forecast

- Select an individual course.
- Predict future enrollments.
- View monthly enrollment forecasts.
- Compare historical enrollment trends.
- Download predictions as CSV.

### 3. Course Revenue Forecast

- Select an individual course.
- Predict future course revenue.
- View monthly revenue trends.
- Explore forecast summaries.
- Download revenue predictions as CSV.

### 4. Category Revenue Forecast

- Select a course category.
- Predict category-level revenue.
- View predicted category enrollments.
- Explore monthly forecast visualizations.
- Download category predictions as CSV.

### 5. About Models

Provides information about the machine learning algorithms, evaluation metrics, and forecasting methodology.

### 6. Dynamic Forecast Horizon

Users can select their preferred forecasting horizon:

- 1 month
- 3 months
- 6 months
- 12 months

The forecast dates are automatically calculated based on the latest available historical observations.

The application uses recursive multi-step forecasting, where predictions from earlier months become inputs for subsequent months.

Forecast uncertainty can accumulate as the forecast horizon increases.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning models and preprocessing |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web application |
| Matplotlib / Plotly | Data visualization during analysis |
| Git & GitHub | Version control and project management |
| Streamlit Community Cloud | Application deployment |

## 📁 Project Structure

```text
EduPro_Project/
│
└── edupro_app/
    │
    ├── app.py
    │
    ├── data/
    │   ├── courses.csv
    │   ├── course_history.csv
    │   └── category_history.csv
    │
    ├── saved_models/
    │   ├── final_enrollment_model.pkl
    │   ├── final_course_revenue_model.pkl
    │   └── final_category_revenue_model.pkl
    │
    ├── requirements.txt
    │
    └── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/manmeetmanchanda10016/EduPro-Predictive-Analytics.git
```

### 2. Navigate to the Application Directory

```bash
cd EduPro-Predictive-Analytics
```

If the application files are inside a nested `edupro_app` directory, navigate into that directory before running the following commands.

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 📦 Requirements

The main dependencies include:

```text
streamlit
pandas
numpy
scikit-learn==1.6.1
joblib
```

The Scikit-learn version is pinned to maintain compatibility with the saved model artifacts.

## 🚀 Deployment

EduPro is deployed using **Streamlit Community Cloud**.

The deployment process involves:

1. Maintaining the application source code on GitHub.
2. Including the required dependencies.
3. Connecting the GitHub repository to Streamlit Community Cloud.
4. Selecting `app.py` as the application entry point.
5. Deploying and publishing the application.

**Live Deployment:**

https://edupro-predictive-analytics-n6ewzdgssbu2gvmzwz4cge.streamlit.app/

## ⚠️ Limitations

- Forecast quality depends on the quality and availability of historical data.
- The enrollment prediction model has limited predictive improvement over a mean baseline.
- Category-level evaluation uses a relatively small held-out test set.
- Recursive forecasting can accumulate prediction errors over longer horizons.
- Forecasts are based on available historical observations and should not be treated as guaranteed outcomes.

## 🔮 Future Improvements

Potential improvements include:

- Incorporating larger and more recent educational datasets.
- Improving enrollment prediction performance.
- Experimenting with advanced time-series forecasting models.
- Adding explainable AI techniques such as SHAP.
- Introducing real-time data integration.
- Adding automated model retraining.
- Developing more advanced forecasting and comparison tools.

## 👨‍💻 Author

**Manmeet Singh Manchanda**

Computer Science and Engineering Student

GitHub: [manmeetmanchanda10016](https://github.com/manmeetmanchanda10016)

## 🙏 Acknowledgement

This project was developed as part of a practical machine learning and predictive analytics learning experience.

The project provided hands-on experience in data preprocessing, feature engineering, model development, evaluation, and deployment.

---

<p align="center">

**EduPro – Turning Educational Data into Predictive Insights.** 🎓

</p>