
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

    enrollment_model = joblib.load(
        os.path.join(
            BASE_DIR,
            "saved_models",
            "final_enrollment_model.pkl"
        )
    )

    course_revenue_model = joblib.load(
        os.path.join(
            BASE_DIR,
            "saved_models",
            "final_course_revenue_model.pkl"
        )
    )

    category_revenue_model = joblib.load(
        os.path.join(
            BASE_DIR,
            "saved_models",
            "final_category_revenue_model.pkl"
        )
    )

    return (
        enrollment_model,
        course_revenue_model,
        category_revenue_model
    )


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    courses = pd.read_csv(
        os.path.join(BASE_DIR, "data", "courses.csv")
    )

    course_history = pd.read_csv(
        os.path.join(BASE_DIR, "data", "course_history.csv")
    )

    category_history = pd.read_csv(
        os.path.join(BASE_DIR, "data", "category_history.csv")
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

try:

    (
        enrollment_model,
        course_revenue_model,
        category_revenue_model
    ) = load_models()

    courses, course_history, category_history = load_data()

except Exception as error:

    st.error(f"Unable to load application files: {error}")
    st.stop()


# =========================================================
# COMMON SETTINGS
# =========================================================

last_observed_month = course_history["MonthStart"].max()

forecast_month = (
    last_observed_month + pd.offsets.MonthBegin(1)
)

forecast_month_number = forecast_month.month
forecast_quarter = forecast_month.quarter

forecast_month_sin = np.sin(
    2 * np.pi * forecast_month_number / 12
)

forecast_month_cos = np.cos(
    2 * np.pi * forecast_month_number / 12
)

forecast_label = forecast_month.strftime("%B %Y")


# =========================================================
# FEATURE GENERATION - COURSE LEVEL
# =========================================================

def create_course_features(course_id, prediction_type):

    course_info = courses[
        courses["CourseID"].astype(str) == str(course_id)
    ]

    history = course_history[
        course_history["CourseID"].astype(str) == str(course_id)
    ].sort_values("MonthStart")

    if course_info.empty or len(history) < 3:
        raise ValueError("Insufficient course history.")

    metadata = course_info.iloc[0]
    last_three = history.tail(3)

    if prediction_type == "enrollment":

        feature_row = {
            "CourseID": metadata["CourseID"],
            "CourseCategory": metadata["CourseCategory"],
            "CourseType": metadata["CourseType"],
            "CourseLevel": metadata["CourseLevel"],
            "CoursePrice": metadata["CoursePrice"],
            "CourseDuration": metadata["CourseDuration"],
            "CourseRating": metadata["CourseRating"],

            "Enrollment_Lag1": last_three.iloc[-1]["EnrollmentCount"],
            "Enrollment_Lag2": last_three.iloc[-2]["EnrollmentCount"],
            "Enrollment_Rolling3": last_three["EnrollmentCount"].mean(),

            "Month": forecast_month_number,
            "Quarter": forecast_quarter,
            "Month_Sin": forecast_month_sin,
            "Month_Cos": forecast_month_cos
        }

    elif prediction_type == "revenue":

        feature_row = {
            "CourseID": metadata["CourseID"],
            "CourseCategory": metadata["CourseCategory"],
            "CourseType": metadata["CourseType"],
            "CourseLevel": metadata["CourseLevel"],
            "CoursePrice": metadata["CoursePrice"],
            "CourseDuration": metadata["CourseDuration"],
            "CourseRating": metadata["CourseRating"],

            "Revenue_Lag1": last_three.iloc[-1]["CourseRevenue"],
            "Revenue_Lag2": last_three.iloc[-2]["CourseRevenue"],
            "Revenue_Rolling3": last_three["CourseRevenue"].mean(),

            "Month": forecast_month_number,
            "Quarter": forecast_quarter,
            "Month_Sin": forecast_month_sin,
            "Month_Cos": forecast_month_cos
        }

    else:
        raise ValueError("Invalid prediction type.")

    return pd.DataFrame([feature_row])


# =========================================================
# FEATURE GENERATION - CATEGORY LEVEL
# =========================================================

def create_category_features(category):

    history = category_history[
        category_history["CourseCategory"] == category
    ].sort_values("MonthStart")

    if len(history) < 3:
        raise ValueError("Insufficient category history.")

    last_three = history.tail(3)

    feature_row = {
        "CourseCategory": category,

        "CategoryRevenue_Lag1": last_three.iloc[-1]["CategoryRevenue"],
        "CategoryRevenue_Lag2": last_three.iloc[-2]["CategoryRevenue"],
        "CategoryRevenue_Rolling3": last_three["CategoryRevenue"].mean(),

        "CategoryEnrollments_Lag1": last_three.iloc[-1]["CategoryEnrollments"],
        "CategoryEnrollments_Lag2": last_three.iloc[-2]["CategoryEnrollments"],
        "CategoryEnrollments_Rolling3": last_three["CategoryEnrollments"].mean(),

        "Month": forecast_month_number
    }

    return pd.DataFrame([feature_row])


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("# 🎓 EduPro")

st.sidebar.caption("PREDICTIVE ANALYTICS PLATFORM")

st.sidebar.divider()

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

st.sidebar.divider()

st.sidebar.markdown("### Forecast Period")
st.sidebar.info(f"📅 {forecast_label}")

st.sidebar.markdown("---")

st.sidebar.caption("Powered by Machine Learning")
st.sidebar.caption("EduPro © 2026")


# =========================================================
# PAGE 1 - OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        '<p class="main-title">EduPro Dashboard</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Monitor educational platform performance and explore future predictions.</p>',
        unsafe_allow_html=True
    )

    total_enrollments = int(
        course_history["EnrollmentCount"].sum()
    )

    total_revenue = float(
        course_history["CourseRevenue"].sum()
    )

    total_courses = courses["CourseID"].nunique()

    total_categories = courses["CourseCategory"].nunique()

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Enrollments",
        f"{total_enrollments:,}"
    )

    col2.metric(
        "Total Revenue",
        f"₹{total_revenue:,.0f}"
    )

    col3.metric(
        "Total Courses",
        f"{total_courses}"
    )

    col4.metric(
        "Categories",
        f"{total_categories}"
    )

    st.divider()

    # MONTHLY TRENDS

    st.markdown(
        '<p class="section-title">Platform Performance</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    monthly_enrollment = (
        course_history.groupby("MonthStart")
        ["EnrollmentCount"]
        .sum()
        .reset_index()
    )

    monthly_revenue = (
        course_history.groupby("MonthStart")
        ["CourseRevenue"]
        .sum()
        .reset_index()
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

    # CATEGORY PERFORMANCE

    col1, col2 = st.columns(2)

    category_summary = (
        course_history.groupby("CourseCategory")
        .agg(
            Enrollments=("EnrollmentCount", "sum"),
            Revenue=("CourseRevenue", "sum")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )

    with col1:

        st.markdown("#### Revenue by Category")

        st.bar_chart(
            category_summary.set_index("CourseCategory")
            [["Revenue"]]
        )

    with col2:

        st.markdown("#### Top Courses")

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

    st.divider()

    st.markdown("### Machine Learning Modules")

    col1, col2, col3 = st.columns(3)

    col1.info("📊 Course Enrollment\n\nRidge Regression")

    col2.info("💰 Course Revenue\n\nRandom Forest")

    col3.info("📈 Category Revenue\n\nGradient Boosting")


# =========================================================
# PAGE 2 - COURSE ENROLLMENT
# =========================================================

elif page == "Course Enrollment":

    st.title("📊 Course Enrollment Prediction")

    st.write(
        f"Predict the expected enrollment for a selected course in {forecast_label}."
    )

    course_options = courses["CourseID"].astype(str).tolist()

    course_lookup = courses.set_index(
        courses["CourseID"].astype(str)
    )["CourseName"].to_dict()

    selected_course = st.selectbox(
        "Choose a Course",
        course_options,
        format_func=lambda x: f"{course_lookup[x]} ({x})"
    )

    course_data = courses[
        courses["CourseID"].astype(str) == selected_course
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Course", course_data["CourseName"])
    col2.metric("Category", course_data["CourseCategory"])
    col3.metric("Course Rating", course_data["CourseRating"])

    if st.button("Predict Enrollments", use_container_width=True):

        try:

            features = create_course_features(
                selected_course,
                "enrollment"
            )

            prediction = float(
                enrollment_model.predict(features)[0]
            )

            prediction = max(0, prediction)

            st.divider()

            st.success("Enrollment prediction generated!")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Enrollments",
                f"{round(prediction):,}"
            )

            col2.metric(
                "Forecast Month",
                forecast_label
            )

            st.caption(
                f"Raw model prediction: {prediction:.2f}"
            )

            st.markdown("#### Recent Enrollment History")

            history = course_history[
                course_history["CourseID"].astype(str)
                == selected_course
            ].sort_values("MonthStart").tail(6)

            st.line_chart(
                history.set_index("MonthStart")
                [["EnrollmentCount"]]
            )

            st.markdown("#### Features Used")

            st.dataframe(
                features,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# =========================================================
# PAGE 3 - COURSE REVENUE
# =========================================================

elif page == "Course Revenue":

    st.title("💰 Course Revenue Prediction")

    st.write(
        f"Estimate the revenue generated by a course in {forecast_label}."
    )

    course_options = courses["CourseID"].astype(str).tolist()

    course_lookup = courses.set_index(
        courses["CourseID"].astype(str)
    )["CourseName"].to_dict()

    selected_course = st.selectbox(
        "Choose a Course",
        course_options,
        format_func=lambda x: f"{course_lookup[x]} ({x})"
    )

    course_data = courses[
        courses["CourseID"].astype(str) == selected_course
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Course", course_data["CourseName"])
    col2.metric("Category", course_data["CourseCategory"])
    col3.metric("Course Price", f"₹{course_data['CoursePrice']:,.0f}")

    if st.button("Predict Course Revenue", use_container_width=True):

        try:

            features = create_course_features(
                selected_course,
                "revenue"
            )

            prediction = float(
                course_revenue_model.predict(features)[0]
            )

            prediction = max(0, prediction)

            st.divider()

            st.success("Course revenue prediction generated!")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Revenue",
                f"₹{prediction:,.2f}"
            )

            col2.metric(
                "Forecast Month",
                forecast_label
            )

            st.markdown("#### Recent Revenue History")

            history = course_history[
                course_history["CourseID"].astype(str)
                == selected_course
            ].sort_values("MonthStart").tail(6)

            st.line_chart(
                history.set_index("MonthStart")
                [["CourseRevenue"]]
            )

            st.markdown("#### Features Used")

            st.dataframe(
                features,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# =========================================================
# PAGE 4 - CATEGORY REVENUE
# =========================================================

elif page == "Category Revenue":

    st.title("📈 Category Revenue Prediction")

    st.write(
        f"Forecast revenue for a selected course category in {forecast_label}."
    )

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

    category_history_selected = category_history[
        category_history["CourseCategory"] == selected_category
    ]

    total_category_enrollments = int(
        category_history_selected["CategoryEnrollments"].sum()
    )

    total_category_revenue = float(
        category_history_selected["CategoryRevenue"].sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Category", selected_category)

    col2.metric(
        "Courses",
        category_courses["CourseID"].nunique()
    )

    col3.metric(
        "Historical Revenue",
        f"₹{total_category_revenue:,.0f}"
    )

    if st.button("Predict Category Revenue", use_container_width=True):

        try:

            features = create_category_features(
                selected_category
            )

            prediction = float(
                category_revenue_model.predict(features)[0]
            )

            prediction = max(0, prediction)

            st.divider()

            st.success("Category revenue prediction generated!")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Category Revenue",
                f"₹{prediction:,.2f}"
            )

            col2.metric(
                "Forecast Month",
                forecast_label
            )

            st.markdown("#### Recent Category Performance")

            recent_history = (
                category_history_selected
                .sort_values("MonthStart")
                .tail(6)
            )

            st.line_chart(
                recent_history.set_index("MonthStart")
                [["CategoryRevenue"]]
            )

            st.markdown("#### Features Used")

            st.dataframe(
                features,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# =========================================================
# PAGE 5 - ABOUT MODELS
# =========================================================

elif page == "About Models":

    st.title("🤖 Machine Learning Models")

    st.write(
        "EduPro uses three trained regression models "
        "to forecast enrollment and revenue."
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
            "MAE: 2.81 | R²: -0.016",
            "MAE: 247.82 | R²: 0.920",
            "MAE: 1016.56 | R²: 0.936"
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
        "Evaluation metrics shown above come from the original "
        "held-out October–December 2025 test set. "
        "The final saved models were subsequently refitted on "
        "all available historical data."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="footer">EduPro | Machine Learning Predictive Analytics | 2026</div>',
    unsafe_allow_html=True
)
