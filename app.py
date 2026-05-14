import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Calories Burned Predictor", layout="centered")

# Load the model and assets using the established "pro" methods
@st.cache_resource
def load_assets():
    # Loading the XGBRegressor saved with joblib
    model = joblib.load('calories_model.joblib')
    # Ensuring features match the exact training order
    features = joblib.load('features.pkl')
    # Manual mapping used to handle 8GB RAM constraints during training
    sex_mapping = {'female': 1, 'male': 0} 
    return model, features, sex_mapping

# Initialize assets
try:
    model, expected_features, sex_mapping = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

st.title("🔥 Calorie Expenditure Predictor")
st.markdown("Predict calories burned during exercise using your high-precision XGBoost model.")

# Create input form to batch interactions and prevent unnecessary re-runs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.selectbox("Sex", options=["female", "male"])
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    
    with col2:
        duration = st.number_input("Duration (min)", min_value=1.0, max_value=180.0, value=30.0)
        heart_rate = st.number_input("Heart Rate", min_value=40.0, max_value=220.0, value=100.0)
        body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)

    # Fixed the function name to streamlit's standard attribute
    submit = st.form_submit_button("Predict Calories")

if submit:
    # Prepare the input dictionary matching the manual mapping
    input_dict = {
        'Sex': sex_mapping[sex],
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp
    }
    
    # Convert to DataFrame and reorder columns to match 'features.pkl'
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[expected_features]
    
    # Run the prediction
    prediction = model.predict(input_df)
    
    # Apply np.clip logic to ensure outputs are compatible with MSLE metrics
    final_val = np.clip(prediction[0], 0, None)
    
    st.success(f"### Estimated Calories Burned: {final_val:.2f} kcal")
    st.info("Note: This model was built for the Kaggle S5E5 Playground Series.")