import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EduPro | Predictive Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}
[data-testid="stSidebar"] {
    background-color: #151B27;
}
.main-title {
    font-size: 36px;
    font-weight: 750;
    color: #FFFFFF;
    margin-bottom: 0;
}
.subtitle {
    color: #A0AEC0;
    font-size: 15px;
    margin-bottom: 25px;
}
.section-title {
    font-size: 23px;
    font-weight: 650;
    margin-top: 15px;
    margin-bottom: 15px;
}
div[data-testid="stMetric"] {
    background-color: #192333;
    border: 1px solid #29384D;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricLabel"] {
    color: #A0AEC0;
}
div[data-testid="stMetricValue"] {
    color: #FFFFFF;
}
.stButton button {
    background-color: #3977F6;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}
.stButton button:hover {
    background-color: #255ACB;
    color: white;
}
.footer {
    color: #718096;
    text-align: center;
    padding: 20px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    model_dir = os.path.join(BASE_DIR, "saved_models")

    enrollment_model = joblib.load(
        os.path.join(model_dir, "final_enrollment_model.pkl")
    )
    course_revenue_model = joblib.load(
        os.path.join(model_dir, "final_course_revenue_model.pkl")
    )
    category_revenue_model = joblib.load(
        os.path.join(model_dir, "final_category_revenue_model.pkl")
    )

    return enrollment_model, course_revenue_model, category_revenue_model


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    data_dir = os.path.join(BASE_DIR, "data")

    courses = pd.read_csv(os.path.join(data_dir, "courses.csv"))
    course_history = pd.read_csv(
        os.path.join(data_dir, "course_history.csv")
    )
    category_history = pd.read_csv(
        os.path.join(data_dir, "category_history.csv")
    )

    course_history["MonthStart"] = pd.to_datetime(
        course_history["MonthStart"]
    )
    category_history["MonthStart"] = pd.to_datetime(
        category_history["MonthStart"]
    )

    return courses, course_history, category_history


# =========================================================
# INITIALIZE APPLICATION
# =========================================================

st.title("🎓 EduPro")
st.caption("PREDICTIVE ANALYTICS PLATFORM")

try:
    (
        enrollment_model,
        course_revenue_model,
        category_revenue_model
    ) = load_models()

    courses, course_history, category_history = load_data()

except Exception as error:
    st.error(f"Unable to load application files: {error}")
    st.exception(error)
    st.stop()


# =========================================================
# VALIDATE DATA
# =========================================================

required_course_columns = {
    "CourseID", "CourseName", "CourseCategory",
    "CourseType", "CourseLevel", "CoursePrice",
    "CourseDuration", "CourseRating"
}
required_course_history_columns = {
    "MonthStart", "CourseID", "CourseName", "CourseCategory",
    "EnrollmentCount", "CourseRevenue"
}
required_category_history_columns = {
    "MonthStart", "CourseCategory",
    "CategoryEnrollments", "CategoryRevenue"
}

missing_columns = (
    required_course_columns - set(courses.columns)
) | (
    required_course_history_columns - set(course_history.columns)
) | (
    required_category_history_columns - set(category_history.columns)
)

if missing_columns:
    st.error(
        "Missing required CSV columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

if course_history.empty or category_history.empty or courses.empty:
    st.error("One or more input CSV files are empty.")
    st.stop()


# =========================================================
# FORECAST SETTINGS
# =========================================================

last_course_month = course_history["MonthStart"].max()
last_category_month = category_history["MonthStart"].max()
last_observed_month = max(last_course_month, last_category_month)

forecast_start = (
    last_observed_month.to_period("M").to_timestamp()
    + pd.offsets.MonthBegin(1)
)

st.sidebar.markdown("# 🎓 EduPro")
st.sidebar.caption("PREDICTIVE ANALYTICS PLATFORM")
st.sidebar.divider()

st.sidebar.markdown("### Forecast Settings")

forecast_horizon = st.sidebar.selectbox(
    "Forecast horizon",
    options=[1, 3, 6, 12],
    index=3,
    format_func=lambda x: f"{x} month" if x == 1 else f"{x} months"
)

forecast_months = pd.date_range(
    start=forecast_start,
    periods=forecast_horizon,
    freq="MS"
)

forecast_end = forecast_months[-1]
forecast_label = (
    f"{forecast_start.strftime('%b %Y')} – "
    f"{forecast_end.strftime('%b %Y')}"
)

st.sidebar.info(
    f"Historical data through: {last_observed_month.strftime('%B %Y')}\n\n"
    f"Forecast period: {forecast_label}"
)

st.sidebar.caption("Powered by Machine Learning")
st.sidebar.caption("EduPro © 2026")

if last_observed_month < pd.Timestamp.today().to_period("M").to_timestamp():
    st.warning(
        f"Your latest historical data is {last_observed_month.strftime('%B %Y')}. "
        "The forecast starts after that date, not after today's date. "
        "Update the CSV history with newer actual data for forecasts "
        "that begin in the current future period."
    )


# =========================================================
# COURSE-LEVEL RECURSIVE FORECASTING
# =========================================================

def generate_course_forecasts():
    predictions = []

    for _, metadata in courses.iterrows():
        course_id = str(metadata["CourseID"])

        history = course_history[
            course_history["CourseID"].astype(str) == course_id
        ].sort_values("MonthStart")

        if len(history) < 3:
            continue

        enrollment_values = (
            history["EnrollmentCount"].astype(float).tail(3).tolist()
        )
        revenue_values = (
            history["CourseRevenue"].astype(float).tail(3).tolist()
        )

        for forecast_date in forecast_months:
            month = int(forecast_date.month)
            quarter = int(forecast_date.quarter)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            common_features = {
                "CourseID": metadata["CourseID"],
                "CourseCategory": metadata["CourseCategory"],
                "CourseType": metadata["CourseType"],
                "CourseLevel": metadata["CourseLevel"],
                "CoursePrice": metadata["CoursePrice"],
                "CourseDuration": metadata["CourseDuration"],
                "CourseRating": metadata["CourseRating"],
                "Month": month,
                "Quarter": quarter,
                "Month_Sin": month_sin,
                "Month_Cos": month_cos
            }

            enrollment_features = pd.DataFrame([{
                **common_features,
                "Enrollment_Lag1": enrollment_values[-1],
                "Enrollment_Lag2": enrollment_values[-2],
                "Enrollment_Rolling3": np.mean(enrollment_values[-3:])
            }])

            revenue_features = pd.DataFrame([{
                **common_features,
                "Revenue_Lag1": revenue_values[-1],
                "Revenue_Lag2": revenue_values[-2],
                "Revenue_Rolling3": np.mean(revenue_values[-3:])
            }])

            predicted_enrollment = max(
                0.0,
                float(enrollment_model.predict(enrollment_features)[0])
            )

            predicted_revenue = max(
                0.0,
                float(course_revenue_model.predict(revenue_features)[0])
            )

            predictions.append({
                "MonthStart": forecast_date,
                "CourseID": course_id,
                "CourseName": metadata["CourseName"],
                "CourseCategory": metadata["CourseCategory"],
                "PredictedEnrollment": predicted_enrollment,
                "PredictedRevenue": predicted_revenue
            })

            enrollment_values.append(predicted_enrollment)
            revenue_values.append(predicted_revenue)

    return pd.DataFrame(predictions)


# =========================================================
# CATEGORY-LEVEL RECURSIVE FORECASTING
# =========================================================

def generate_category_forecasts(course_forecasts):
    if course_forecasts.empty:
        return pd.DataFrame()

    monthly_category_enrollment = (
        course_forecasts.groupby(
            ["MonthStart", "CourseCategory"],
            as_index=False
        )["PredictedEnrollment"].sum()
    )

    predictions = []

    categories = sorted(
        category_history["CourseCategory"].dropna().unique()
    )

    for category in categories:
        history = category_history[
            category_history["CourseCategory"] == category
        ].sort_values("MonthStart")

        if len(history) < 3:
            continue

        revenue_values = (
            history["CategoryRevenue"].astype(float).tail(3).tolist()
        )
        enrollment_values = (
            history["CategoryEnrollments"].astype(float).tail(3).tolist()
        )

        for forecast_date in forecast_months:
            month = int(forecast_date.month)

            future_enrollment = monthly_category_enrollment[
                (monthly_category_enrollment["MonthStart"] == forecast_date)
                & (
                    monthly_category_enrollment["CourseCategory"]
                    == category
                )
            ]["PredictedEnrollment"].sum()

            features = pd.DataFrame([{
                "CourseCategory": category,
                "CategoryRevenue_Lag1": revenue_values[-1],
                "CategoryRevenue_Lag2": revenue_values[-2],
                "CategoryRevenue_Rolling3": np.mean(revenue_values[-3:]),
                "CategoryEnrollments_Lag1": enrollment_values[-1],
                "CategoryEnrollments_Lag2": enrollment_values[-2],
                "CategoryEnrollments_Rolling3": np.mean(enrollment_values[-3:]),
                "Month": month
            }])

            predicted_revenue = max(
                0.0,
                float(category_revenue_model.predict(features)[0])
            )

            predictions.append({
                "MonthStart": forecast_date,
                "CourseCategory": category,
                "PredictedEnrollment": float(future_enrollment),
                "PredictedRevenue": predicted_revenue
            })

            revenue_values.append(predicted_revenue)
            enrollment_values.append(float(future_enrollment))

    return pd.DataFrame(predictions)


# =========================================================
# GENERATE FORECASTS
# =========================================================

st.info("Loading models and generating forecasts. Please wait...")

try:
    course_forecasts = generate_course_forecasts()
    category_forecasts = generate_category_forecasts(course_forecasts)

    if course_forecasts.empty:
        st.error("No course forecasts were generated.")
        st.stop()

    if category_forecasts.empty:
        st.error("No category forecasts were generated.")
        st.stop()

except Exception as error:
    st.error(f"Forecast generation failed: {error}")
    st.exception(error)
    st.stop()

st.success(f"Forecasts generated for {forecast_label}.")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_month(value):
    return pd.Timestamp(value).strftime("%B %Y")


def get_course_forecast(course_id):
    return course_forecasts[
        course_forecasts["CourseID"].astype(str) == str(course_id)
    ].sort_values("MonthStart")


def get_category_forecast(category):
    return category_forecasts[
        category_forecasts["CourseCategory"] == category
    ].sort_values("MonthStart")


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Course Enrollment",
        "Course Revenue",
        "Category Revenue",
        "About Models"
    ]
)



# =========================================================
# PAGE 1 - OVERVIEW
# =========================================================

if page == "Overview":
    st.markdown(
        '<p class="main-title">EduPro Dashboard</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Monitor educational platform performance '
        'and explore future predictions.</p>',
        unsafe_allow_html=True
    )

    total_enrollments = int(course_history["EnrollmentCount"].sum())
    total_revenue = float(course_history["CourseRevenue"].sum())
    total_courses = courses["CourseID"].nunique()
    total_categories = courses["CourseCategory"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical Enrollments", f"{total_enrollments:,}")
    col2.metric("Historical Revenue", f"₹{total_revenue:,.0f}")
    col3.metric("Total Courses", total_courses)
    col4.metric("Categories", total_categories)

    st.divider()
    st.markdown("### Historical Platform Performance")

    col1, col2 = st.columns(2)

    monthly_enrollment = (
        course_history.groupby("MonthStart")["EnrollmentCount"]
        .sum().reset_index()
    )
    monthly_revenue = (
        course_history.groupby("MonthStart")["CourseRevenue"]
        .sum().reset_index()
    )

    with col1:
        st.markdown("#### Monthly Enrollments")
        st.line_chart(
            monthly_enrollment.set_index("MonthStart")
        )

    with col2:
        st.markdown("#### Monthly Revenue")
        st.line_chart(
            monthly_revenue.set_index("MonthStart")
        )

    st.divider()
    st.markdown(f"### Forecast Summary: {forecast_label}")

    monthly_enrollment_forecast = (
        course_forecasts.groupby("MonthStart")
        ["PredictedEnrollment"].sum().reset_index()
    )
    monthly_revenue_forecast = (
        course_forecasts.groupby("MonthStart")
        ["PredictedRevenue"].sum().reset_index()
    )

    annual_enrollment = monthly_enrollment_forecast[
        "PredictedEnrollment"
    ].sum()

    total_forecast_revenue = monthly_revenue_forecast[
        "PredictedRevenue"
    ].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast Enrollments", f"{annual_enrollment:,.0f}")
    col2.metric("Forecast Revenue", f"₹{total_forecast_revenue:,.0f}")
    col3.metric("Forecast Months", forecast_horizon)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Monthly Enrollment Forecast")
        st.line_chart(
            monthly_enrollment_forecast.set_index("MonthStart")
        )

    with col2:
        st.markdown("#### Monthly Revenue Forecast")
        st.line_chart(
            monthly_revenue_forecast.set_index("MonthStart")
        )

    st.divider()
    st.markdown("### Historical Revenue by Category")

    historical_category_summary = (
        course_history.groupby("CourseCategory")
        ["CourseRevenue"].sum().reset_index()
        .sort_values("CourseRevenue", ascending=False)
    )

    st.bar_chart(
        historical_category_summary.set_index("CourseCategory")
    )

    st.markdown("### Forecast Revenue by Category")

    category_annual_summary = (
        category_forecasts.groupby("CourseCategory")
        ["PredictedRevenue"].sum().reset_index()
        .sort_values("PredictedRevenue", ascending=False)
    )

    st.bar_chart(
        category_annual_summary.set_index("CourseCategory")
    )

    st.markdown("### Top Courses by Historical Enrollments")

    top_courses = (
        course_history.groupby("CourseName")
        .agg(
            Enrollments=("EnrollmentCount", "sum"),
            Revenue=("CourseRevenue", "sum")
        )
        .sort_values("Enrollments", ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(
        top_courses,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Multi-step forecasts are recursive: earlier predictions become "
        "inputs for later months. Uncertainty can accumulate as the "
        "forecast horizon increases."
    )


# =========================================================
# PAGE 2 - COURSE ENROLLMENT
# =========================================================

elif page == "Course Enrollment":
    st.title("📊 Course Enrollment Forecast")
    st.write(f"Enrollment forecast for {forecast_label}.")

    course_options = courses["CourseID"].astype(str).tolist()

    course_lookup = dict(
        zip(courses["CourseID"].astype(str), courses["CourseName"])
    )

    selected_course = st.selectbox(
        "Choose a Course",
        course_options,
        format_func=lambda x: f"{course_lookup[x]} ({x})"
    )

    course_data = courses[
        courses["CourseID"].astype(str) == selected_course
    ].iloc[0]

    forecast = get_course_forecast(selected_course)

    col1, col2, col3 = st.columns(3)
    col1.metric("Course", course_data["CourseName"])
    col2.metric("Category", course_data["CourseCategory"])
    col3.metric("Course Rating", course_data["CourseRating"])

    if not forecast.empty:
        total_predicted = forecast["PredictedEnrollment"].sum()
        average_predicted = forecast["PredictedEnrollment"].mean()

        col1, col2 = st.columns(2)
        col1.metric(
            "Total Forecast Enrollments",
            f"{total_predicted:,.0f}"
        )
        col2.metric(
            "Average Monthly Enrollments",
            f"{average_predicted:,.1f}"
        )

        st.markdown("### Monthly Enrollment Forecast")

        st.line_chart(
            forecast.set_index("MonthStart")[["PredictedEnrollment"]]
        )

        display = forecast.copy()
        display["Month"] = display["MonthStart"].apply(format_month)
        display["PredictedEnrollment"] = display[
            "PredictedEnrollment"
        ].round(2)

        st.dataframe(
            display[["Month", "PredictedEnrollment"]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Enrollment Forecast CSV",
            forecast.to_csv(index=False).encode("utf-8"),
            file_name="course_enrollment_forecast.csv",
            mime="text/csv"
        )

        st.markdown("### Historical Enrollment")

        history = course_history[
            course_history["CourseID"].astype(str) == selected_course
        ].sort_values("MonthStart")

        st.line_chart(
            history.set_index("MonthStart")[["EnrollmentCount"]]
        )



        # =========================================================
# PAGE 3 - COURSE REVENUE
# =========================================================

elif page == "Course Revenue":
    st.title("💰 Course Revenue Forecast")
    st.write(f"Revenue forecast for {forecast_label}.")

    course_options = courses["CourseID"].astype(str).tolist()

    course_lookup = dict(
        zip(courses["CourseID"].astype(str), courses["CourseName"])
    )

    selected_course = st.selectbox(
        "Choose a Course",
        course_options,
        format_func=lambda x: f"{course_lookup[x]} ({x})"
    )

    course_data = courses[
        courses["CourseID"].astype(str) == selected_course
    ].iloc[0]

    forecast = get_course_forecast(selected_course)

    col1, col2, col3 = st.columns(3)
    col1.metric("Course", course_data["CourseName"])
    col2.metric("Category", course_data["CourseCategory"])
    col3.metric("Course Price", f"₹{course_data['CoursePrice']:,.0f}")

    if not forecast.empty:
        total_predicted = forecast["PredictedRevenue"].sum()
        average_predicted = forecast["PredictedRevenue"].mean()

        col1, col2 = st.columns(2)
        col1.metric("Total Forecast Revenue", f"₹{total_predicted:,.2f}")
        col2.metric("Average Monthly Revenue", f"₹{average_predicted:,.2f}")

        st.markdown("### Monthly Revenue Forecast")

        st.line_chart(
            forecast.set_index("MonthStart")[["PredictedRevenue"]]
        )

        display = forecast.copy()
        display["Month"] = display["MonthStart"].apply(format_month)
        display["PredictedRevenue"] = display[
            "PredictedRevenue"
        ].round(2)

        st.dataframe(
            display[["Month", "PredictedRevenue"]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Revenue Forecast CSV",
            forecast.to_csv(index=False).encode("utf-8"),
            file_name="course_revenue_forecast.csv",
            mime="text/csv"
        )

        st.markdown("### Historical Revenue")

        history = course_history[
            course_history["CourseID"].astype(str) == selected_course
        ].sort_values("MonthStart")

        st.line_chart(
            history.set_index("MonthStart")[["CourseRevenue"]]
        )


# =========================================================
# PAGE 4 - CATEGORY REVENUE
# =========================================================

elif page == "Category Revenue":
    st.title("📈 Category Revenue Forecast")
    st.write(f"Category forecast for {forecast_label}.")

    categories = sorted(
        category_history["CourseCategory"].dropna().unique()
    )

    selected_category = st.selectbox(
        "Choose a Course Category",
        categories
    )

    category_courses = courses[
        courses["CourseCategory"] == selected_category
    ]

    historical = category_history[
        category_history["CourseCategory"] == selected_category
    ]

    forecast = get_category_forecast(selected_category)

    historical_revenue = historical["CategoryRevenue"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Category", selected_category)
    col2.metric("Courses", category_courses["CourseID"].nunique())
    col3.metric("Historical Revenue", f"₹{historical_revenue:,.0f}")

    if not forecast.empty:
        total_revenue = forecast["PredictedRevenue"].sum()
        total_enrollment = forecast["PredictedEnrollment"].sum()

        col1, col2 = st.columns(2)
        col1.metric("Forecast Revenue", f"₹{total_revenue:,.2f}")
        col2.metric("Forecast Enrollments", f"{total_enrollment:,.0f}")

        st.markdown("### Monthly Category Revenue Forecast")

        st.line_chart(
            forecast.set_index("MonthStart")[["PredictedRevenue"]]
        )

        st.markdown("### Monthly Category Enrollment Forecast")

        st.line_chart(
            forecast.set_index("MonthStart")[["PredictedEnrollment"]]
        )

        display = forecast.copy()
        display["Month"] = display["MonthStart"].apply(format_month)
        display["PredictedRevenue"] = display[
            "PredictedRevenue"
        ].round(2)
        display["PredictedEnrollment"] = display[
            "PredictedEnrollment"
        ].round(2)

        st.dataframe(
            display[
                ["Month", "PredictedEnrollment", "PredictedRevenue"]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Category Forecast CSV",
            forecast.to_csv(index=False).encode("utf-8"),
            file_name="category_forecast.csv",
            mime="text/csv"
        )

        st.markdown("### Historical Category Revenue")

        st.line_chart(
            historical.sort_values("MonthStart")
            .set_index("MonthStart")[["CategoryRevenue"]]
        )


# =========================================================
# PAGE 5 - ABOUT MODELS
# =========================================================

elif page == "About Models":
    st.title("🤖 Machine Learning Models")

    st.write(
        "EduPro uses three regression models to forecast course "
        "enrollment and revenue."
    )

    model_data = pd.DataFrame({
        "Prediction Task": [
            "Course Enrollment",
            "Course Revenue",
            "Category Revenue"
        ],
        "Algorithm": [
            "Ridge Regression",
            "Random Forest Regressor",
            "Gradient Boosting Regressor"
        ],
        "Evaluation Metric": [
            "MAE: 2.81 | RMSE: 3.69 | R²: -0.016",
            "MAE: 247.82 | RMSE: 619.06 | R²: 0.920",
            "MAE: 1016.56 | RMSE: 1529.15 | R²: 0.936"
        ]
    })

    st.dataframe(
        model_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.markdown("### Model Information")

    st.markdown("""
    **1. Ridge Regression**
    - Used for course enrollment forecasting.
    - Applies L2 regularization.
    - Helps control model complexity.

    **2. Random Forest Regressor**
    - Used for course revenue forecasting.
    - Combines multiple decision trees.
    - Captures nonlinear feature relationships.

    **3. Gradient Boosting Regressor**
    - Used for category revenue forecasting.
    - Builds sequential trees to reduce prediction errors.
    """)

    st.warning(
        "Evaluation metrics shown above come from the original held-out "
        "October–December 2025 test set. The final saved models were "
        "subsequently refitted on all available historical observations. "
        "These metrics do not validate the current multi-month forecasts."
    )

    st.markdown("### Forecasting Method")

    st.markdown("""
    EduPro uses recursive multi-step forecasting:

    1. The first forecast month uses the latest available historical data.
    2. Its predictions become lag inputs for the following month.
    3. The process repeats for the selected forecast horizon.
    4. Future category enrollments are aggregated from course-level
       enrollment predictions.

    Forecast uncertainty can accumulate as the horizon increases.
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="footer">EduPro | Machine Learning Predictive Analytics '
    '| Dynamic Forecasting</div>',
    unsafe_allow_html=True
)