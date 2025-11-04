import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the Saved Models and Scaler ---

try:
    # Load the scaler
    scaler = joblib.load('standard_scaler.joblib')

    # Load the models
    heating_model = joblib.load('linear_regression_heating_model.joblib')
    cooling_model = joblib.load('linear_regression_cooling_model.joblib')
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Make sure the following files are in the same directory as the script: 'standard_scaler.joblib', 'linear_regression_heating_model.joblib', 'linear_regression_cooling_model.joblib'")
    st.stop() # Stop execution if files are missing
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()

# --- Streamlit App Interface ---

st.set_page_config(page_title="Building Energy Efficiency Predictor", layout="wide")

# Custom CSS for better styling (optional)
st.markdown("""
<style>
    /* Center title */
    .stApp > header {
        background-color: transparent;
    }
    h1 {
        text-align: center;
        color: #4CAF50; /* Green color */
    }
    /* Style sliders and inputs */
    .stSlider [data-baseweb="slider"] {
        background-color: #4CAF50;
    }
    /* Style the button */
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;
    }
    /* Style the results display */
    .result-box {
        background-color: #e8f5e9; /* Light green background */
        border-left: 6px solid #4CAF50; /* Green left border */
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
        color: #1b5e20; /* Dark green text */
    }
    .result-box h3 {
        color: #1b5e20;
        margin-bottom: 10px;
    }
    .result-box p {
        font-size: 1.5em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("🏢 Building Energy Efficiency Predictor")
st.markdown("Predict the Heating Load (Y1) and Cooling Load (Y2) of a building based on its features.")

# --- User Input Features in the Sidebar ---
st.sidebar.header("Building Features")
st.sidebar.markdown("Adjust the sliders and select options to input building characteristics.")

# Feature names (must match the order used during training)
features_order = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
                  'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution']

# Input Widgets
relative_compactness = st.sidebar.slider("Relative Compactness (X1)", 0.60, 1.00, 0.75, 0.01)
surface_area = st.sidebar.slider("Surface Area (X2)", 500.0, 810.0, 650.0, 0.5)
wall_area = st.sidebar.slider("Wall Area (X3)", 240.0, 420.0, 300.0, 0.5)
roof_area = st.sidebar.slider("Roof Area (X4)", 100.0, 225.0, 150.0, 0.25)
overall_height = st.sidebar.select_slider("Overall Height (X5)", options=[3.5, 7.0], value=7.0) # Two distinct values in dataset
orientation = st.sidebar.selectbox("Orientation (X6)", options=[2, 3, 4, 5], index=0) # 2=North, 3=East, 4=South, 5=West
glazing_area = st.sidebar.select_slider("Glazing Area (X7)", options=[0.0, 0.10, 0.25, 0.40], value=0.10) # Distinct values
glazing_area_dist = st.sidebar.selectbox("Glazing Area Distribution (X8)", options=[0, 1, 2, 3, 4, 5], index=1) # 0=Uniform, 1-5 patterns

# Store inputs in a dictionary (ensure the order matches 'features_order')
input_data = {
    'Relative_Compactness': relative_compactness,
    'Surface_Area': surface_area,
    'Wall_Area': wall_area,
    'Roof_Area': roof_area,
    'Overall_Height': overall_height,
    'Orientation': orientation,
    'Glazing_Area': glazing_area,
    'Glazing_Area_Distribution': glazing_area_dist
}

# --- Display User Input ---
st.subheader("Current Input Features")
input_df_display = pd.DataFrame([input_data]) # Create a DataFrame for nice display
st.dataframe(input_df_display.style.format("{:.2f}")) # Format to 2 decimal places

# --- Prediction Logic ---
if st.button("Predict Energy Loads"):
    try:
        # 1. Create a DataFrame from input, ensuring correct column order
        input_df = pd.DataFrame([input_data], columns=features_order)

        # 2. Scale the input data using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # 3. Make predictions using the loaded models
        heating_load_pred = heating_model.predict(input_scaled)
        cooling_load_pred = cooling_model.predict(input_scaled)

        # 4. Display the results
        st.subheader("Predicted Loads")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="result-box">
                <h3>Heating Load (Y1)</h3>
                <p>{heating_load_pred[0]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="result-box">
                <h3>Cooling Load (Y2)</h3>
                <p>{cooling_load_pred[0]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Add some explanation or footer
st.markdown("---")
st.markdown("This app uses Linear Regression models trained on the Energy Efficiency dataset.")
