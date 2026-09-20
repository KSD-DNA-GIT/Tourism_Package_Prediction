import os
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.joblib')
    return joblib.load(model_path)

pipeline = load_model()

st.title("🛫 Wellness Tourism Package Predictor")
st.markdown("Evaluate customer probability of purchasing the new Wellness Package.")
st.info("Disclaimer: This is a predictive decision-support tool, not a guarantee of purchase.")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    OwnCar = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=25000.0)

with col2:
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, value=15.0)
    NumberOfPersonVisiting = st.number_input("Total Persons Visiting", min_value=1, value=2)
    NumberOfChildrenVisiting = st.number_input("Children (<5 yrs) Visiting", min_value=0.0, value=0.0)
    NumberOfTrips = st.number_input("Average Annual Trips", min_value=1.0, value=2.0)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    NumberOfFollowups = st.slider("Number of Follow-ups", 1.0, 6.0, 3.0)

if st.button("Predict Likelihood", type="primary"):
    input_df = pd.DataFrame([{
        'Age': Age, 'TypeofContact': TypeofContact, 'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch, 'Occupation': Occupation, 'Gender': Gender,
        'NumberOfPersonVisiting': NumberOfPersonVisiting, 'NumberOfFollowups': NumberOfFollowups,
        'ProductPitched': ProductPitched, 'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': MaritalStatus, 'NumberOfTrips': NumberOfTrips, 'Passport': Passport,
        'PitchSatisfactionScore': PitchSatisfactionScore, 'OwnCar': OwnCar,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting, 'Designation': Designation,
        'MonthlyIncome': MonthlyIncome
    }])
    
    prob = pipeline.predict_proba(input_df)[0][1]
    
    st.divider()
    if prob >= 0.45:
        st.success(f"🌟 **High Likelihood to Purchase!** (Probability: {prob:.1%})")
    else:
        st.warning(f"📉 **Low Likelihood to Purchase.** (Probability: {prob:.1%})")
