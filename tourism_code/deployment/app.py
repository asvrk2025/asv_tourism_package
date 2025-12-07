import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="asvravi/asv-tourism-package", filename="best_toursim_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction")
st.write("""
This application predicts the likelihood of a customer buying the new Tourism Package.
Please enter the data below to get a prediction.
""")

# User input
st.header("Section 1 – Basic Information")

# ---------- Row 1 ----------
col1, col2, col3, col7 = st.columns(4)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=150,
        value=25
    )

with col2:
    # Marital status alphabetically
    marital_status_options = sorted(["Married", "Single", "Divorced", "Unmarried"])
    marital_status = st.selectbox(
        "Marital Status",
        marital_status_options,
        index=0
    )

with col3:
    gender = st.radio(
        "Gender",
        ["Male", "Female"],
        index=0
    )

with col7:
    own_car = st.selectbox(
        "Own a Car",
        ["Yes", "No"],
        index=0
    )

# ---------- Row 2 ----------
col4, col5, col6 = st.columns(3)

with col4:
    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3],
        index=0
    )

with col5:
    total_family = st.number_input(
        "Total Family Members",
        min_value=1,
        max_value=50,
        value=1,
        step=1
    )

with col6:
    children = st.number_input(
        "No. of Children (age > 5)",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

st.header("Section 2 – Professional Details")

# ---------- Row 1 ----------
col1, col2, col3 = st.columns(3)

with col1:
    occupation_options = sorted(["Free Lancer", "Salaried", "Large Business", "Small Business"])
    occupation = st.selectbox(
        "Occupation",
        occupation_options,
        index=0
    )

with col2:
    designation_options = sorted(["AVP", "Manager", "Senior Manager", "Executive", "VP"])
    designation = st.selectbox(
        "Designation",
        designation_options,
        index=0
    )

with col3:
    monthly_salary = st.number_input(
        "Monthly Salary",
        min_value=1000,
        max_value=100000,
        value=1000,
        step=100
    )


st.header("Section 3 – Travel Preferences")

# ---------- Row 1 ----------
col1, col2, col3 = st.columns(3)

with col1:
    property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5],
        index=0
    )

with col2:
    trips_per_year = st.number_input(
        "Number of Trips per Year",
        min_value=1,
        max_value=50,
        value=1,
        step=1
    )

with col3:
    passport = st.selectbox(
        "Passport",
        ["Yes", "No"],
        index=0
    )

st.header("Section 4 – Sales Interaction Details")

# ---------- Row 1 ----------
col1, col2, col3 = st.columns(3)

with col1:
    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Enquiry"],
        index=0
    )

with col2:
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"],
        index=0
    )

with col3:
    pitch_duration = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=1,
        max_value=150,
        value=1,
        step=1
    )

# ---------- Row 2 ----------
col4, col5, col6 = st.columns(3)

with col4:
    followups = st.number_input(
        "Number of Follow-ups",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )

with col5:
    pitch_satisfaction = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5],
        index=0
    )


own_car = 1 if own_car == "Yes" else 0
passport = 1 if passport == "Yes" else 0

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'DurationOfPitch': pitch_duration,
    'NumberOfFollowups': followups,
    'PitchSatisfactionScore': pitch_satisfaction,
    'NumberOfPersonVisiting': total_family,
    'PreferredPropertyStar': property_star,
    'NumberOfTrips': trips_per_year,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': children,
    'MonthlyIncome': monthly_salary,
    'CityTier': city_tier,     
    'TypeofContact': type_of_contact,         
    'Occupation': occupation,            
    'Gender': gender,              
    'ProductPitched': product_pitched,
    'MaritalStatus': marital_status,         
    'Designation': designation                        
}])

classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "likely to buy" if prediction == 1 else "not likely to buy"
    st.write(f"Based on the information provided, the customer is {result} the new tourism package.")
