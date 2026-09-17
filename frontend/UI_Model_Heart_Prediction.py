import streamlit as st
import pandas as pd
import numpy as np
import base64
import requests
import pickle


def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a class="download-link" href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href


st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.markdown("""
<style>

@keyframes doubleColorBlink {
    0% {
        border-color: crimson;
        color: white;
        box-shadow: 0 0 10px rgba(220, 20, 60, 0.8);
    }
    50% {
        border-color: white;
        color: rgb(7, 71, 69);
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.9);
    }
    100% {
        border-color: crimson;
        color: white;
        box-shadow: 0 0 10px rgba(220, 20, 60, 0.8);
    }
}

.stApp {
    background-color: rgb(7, 71, 69);
    color: white;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main {
    padding-top: 20px;
}

.title {
    text-align: center;
    font-size: 40px;
    color: white;
    font-style: oblique 40deg;
    font-weight: bold;
    margin-bottom: 5px;
    transition: all 0.3s ease;
}

.title:hover {
    color: black;
    text-shadow: 0 0 10px black, 0 0 20px white;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    font-style: oblique 40deg;
    color: lightgray;
    margin-bottom: 30px;
}

button[data-baseweb="tab"] {
    font-size: 6px !important;
    font-weight: bold !important;
    font-style: oblique 40deg !important;
    color: white !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.3s ease-in-out;
}

button[data-baseweb="tab"]:hover {
    animation: doubleColorBlink 1.2s infinite ease-in-out;
    border-radius: 8px;
}

label, div[data-testid="stMarkdownContainer"] p, .stWidgetLabel label {
    font-size: 22px !important;
    font-weight: 500 !important;
    font-style: oblique 40deg !important;
    color: white !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="select"] div,
div[data-baseweb="select"] span {
    font-size: 22px !important;
}

div[data-baseweb="input"]:hover > div, div[data-baseweb="select"]:hover > div {
    animation: doubleColorBlink 1.2s infinite ease-in-out;
}

.stNumberInput input {
    height: 45px;
    font-size: 20px !important;
    font-style: oblique 40deg !important;
    color: black !important;
    background-color: white;
}

/* SECTION TITLE (BORDER REMOVED) */
.section {
    font-size: 32px;
    color: #3300000;
    font-weight: bold;
    font-style: oblique 40deg;
    border-bottom: none !important;
    padding-bottom: 0px;
}

/* ALGORITHM SUBHEADERS (LOGISTIC REGRESSION, DECISION TREE, RANDOM FOREST) */
.stApp h3 {
    font-style: oblique 40deg !important;
    font-size: 28px !important;
    font-weight:600;
    color: black !important;
    font-weight: bold;
}

.stButton {
    display: flex;
    justify-content: center;
}

.stButton > button {
    width: 1000% !important;
    height: 55px !important;
    font-size: 24px !important;
    font-weight: bold !important;
    font-style: oblique 40deg !important;
    border-radius: 15px !important;
    background-color: black !important;
    color: black !important;
    border: 2px solid black !important;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.stButton > button:hover {
    animation: doubleColorBlink 1s infinite ease-in-out;
    background-color: black !important;
}

/* RESULT SAFE CARD  no heart disease*/
.result-safe {
    background-color: black;
    border: 1px solid white;
    color:white;
    padding: 12px;
    border-radius: 10px;
    font-size: 27px;
    font-weight: 700;
    font-style: oblique 40deg;
    text-align: center;
    transition: color 0.3s ease;
}

.result-safe:hover {
    color: white;
}

/* RESULT DANGER CARD heart disease */
.result-danger {
    background-color:white;
    border: blue;
    color: black;
    padding: 12px;
    border-radius: 8px;
    font-size: 27px;
    font-weight: 700;
    font-style: oblique 40deg;
    text-align: center;
    transition: color 0.3s ease;
}

.result-danger:hover {
    animation: doubleColorBlink 2s infinite;
}

/* Tab2 = Bulk Prediction css styles */

.stApp h1 {
    font-style: oblique 40deg !important;
    color: white !important;
    text-align: center;
}

div[data-testid="stAlert"] {
    background-color: black !important;
    border: 1px solid white !important;
    border-radius: 10px !important;
    padding: 15px !important;
    transition: all 0.3s ease;
}

div[data-testid="stAlert"]:hover {
    animation: doubleColorBlink 1.2s infinite ease-in-out;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] li,
div[data-testid="stAlert"] span {
    color: white !important;
    font-size: 21px !important;
    font-weight: 500 !important;
    font-style: oblique 40deg !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: black !important;
    border: 2px dashed white !important;
    border-radius: 10px !important;
    transition: all 0.3s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    animation: doubleColorBlink 1.2s infinite ease-in-out;
}

[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div {
    color: white !important;
    font-style: oblique 40deg !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background-color:#004526;
    color: black !important;
    border-radius: 8px !important;
    font-weight:600;
    font-style: oblique 40deg !important;
}

[data-testid="stFileUploaderFile"] {
    background-color: black !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid white !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid white !important;
    border-radius: 10px !important;
    overflow: hidden;
}

.download-link {
    display: inline-block;
    background-color: white;
    color: black !important;
    font-weight: bold;
    font-style: oblique 40deg;
    font-size: 18px;
    padding: 10px 22px;
    border-radius: 12px;
    text-decoration: none;
    margin-top: 15px;
    border: 2px solid black;
    transition: all 0.3s ease;
}

.download-link:hover {
    animation: doubleColorBlink 1s infinite ease-in-out;
}

[data-testid="stPlotlyChart"] {
    border: 1px solid white !important;
    border-radius: 10px !important;
    padding: 10px !important;
    background-color: black !important;
    transition: all 0.3s ease;
}

[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 0 15px rgba(220, 20, 60, 0.5);
}

</style>
""", unsafe_allow_html=True)

# Header of the UI
st.markdown('<div class="title">Heart Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Based Heart Disease Prediction System</div>', unsafe_allow_html=True)

# Three tabs prediction
# Tab1 = Is to predict single person at a time, Whether a person contains the heart disease or not
# Tab2 = Is to predict multiple person at a time , Whether a person contains the heart disease or not
# Tab3 = Gives the accuarry of the Model as[Logistic,Decision tree,Random forest]

tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Prediction", "Model Information"])

# Tab1 = Prediction of single person at a time
with tab1:
    # Input data
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=700)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["≤ 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting Electrocardiogram (ECG) Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Here,the model doesnot understand the label encoding, so we shifted into one hot endcoding
    sex_num = 0 if sex == "Male" else 1
    fasting_bs_num = 1 if fasting_bs == "> 120 mg/dl" else 0
    chest_pain_mapping = {"Atypical Angina": 0, "Non-Anginal Pain": 1, "Asymptomatic": 2, "Typical Angina": 3}
    chest_pain_num = chest_pain_mapping[chest_pain]
    exercise_angina_num = 0 if exercise_angina == "No" else 1
    resting_ecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    resting_ecg_num = resting_ecg_mapping[resting_ecg]
    st_slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    st_slope_num = st_slope_mapping[st_slope]

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_num],
        'ChestPainType': [chest_pain_num],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_num],
        'RestingECG': [resting_ecg_num],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_num],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_num]
    })

    # Sending data to backend
    if st.button("Submit"):
        st.markdown('<div class="section">Model Predictions</div>', unsafe_allow_html=True)

        # Convert DataFrame into dictionary
        data = input_data.iloc[0].to_dict()

        # Send data to FastAPI backend
        response = requests.post("https://studentdetails-2-gdiv.onrender.com/predict", json=data)

        # Get response from backend
        try:
            result = response.json()
        except requests.exceptions.JSONDecodeError:
            st.error(f"Backend returned a non-JSON response (status {response.status_code}). It may be waking up — try again in a few seconds.\n\nRaw: {response.text[:300]}")
            st.stop()
        print("Result of the predict response", result)

        # Check if the backend sent back an error instead of a prediction
        if "detail" in result:
            st.error(f"Prediction failed: {result['detail']}")
        else:
            # For all - No error — the backend sent back real predictions, so display them
            # Returns result for the logistic regression[Heart Disease/No Heart Disease ]
            st.subheader("Logistic Regression")
            if result["Logistic Regression"] == 0:
                st.markdown('<div class="result-safe">No Heart disease..........</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-danger">Heart disease........</div>', unsafe_allow_html=True)

            # Returns result for the decision tree[Heart Disease/No Heart Disease ]
            st.subheader("Decision Tree")
            if result["Decision Tree"] == 0:
                st.markdown('<div class="result-safe">No Heart disease..........</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-danger">Heart disease........</div>', unsafe_allow_html=True)

            # Returns result for the random forest[Heart Disease/No Heart Disease]
            st.subheader("Random Forest")
            if result["Random Forest"] == 0:
                st.markdown('<div class="result-safe">No Heart disease..........</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-danger">Heart disease........</div>', unsafe_allow_html=True)

# Tab2 = Bulk Prediction
with tab2:
    st.subheader("Upload CSV file")
    st.subheader("Information to note before uploading the CSV file")
    st.info("""
- No NaN values allowed.
- Total 11 features in this order: ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'].
- Check the spellings of the feature names.
- Feature values conventions:

  - Age: age of the patient [years]
  - Sex: sex of the patient [0: Male, 1: Female]
  - ChestPainType: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic]
  - RestingBP: resting blood pressure [mm Hg]
  - Cholesterol: serum cholesterol [mm/dl]
  - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
  - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversion)]
  - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
  - ExerciseAngina: exercise induced angina [1: Yes, 0: No]
  - Oldpeak: oldpeak = ST [Numeric value measured in depression]
  - ST_Slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping]
""")

    # Uploading the CSV File , --> This creates a file upload button.
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:

        ## File is not none , enter into the condition
        input_data = pd.read_csv(uploaded_file)
        expected_columns = [
            'Age',
            'Sex',
            'ChestPainType',
            'RestingBP',
            'Cholesterol',
            'FastingBS',
            'RestingECG',
            'MaxHR',
            'ExerciseAngina',
            'Oldpeak',
            'ST_Slope'
        ]

        if set(expected_columns).issubset(input_data.columns):
            data = input_data[expected_columns].to_dict(orient="records")
            response = requests.post("https://studentdetails-2-gdiv.onrender.com/predict-bulk", json=data)
            try:
                result = response.json()
            except requests.exceptions.JSONDecodeError:
                st.error(f"Backend returned a non-JSON response (status {response.status_code}). It may be waking up — try again in a few seconds.\n\nRaw: {response.text[:300]}")
                st.stop()
            prediction_data = pd.DataFrame(result)
            st.subheader("Predictions:")
            st.write(prediction_data)
            prediction_data.to_csv("PredictedHeartDisease.csv", index=False)
            st.markdown(get_binary_file_downloader_html(prediction_data), unsafe_allow_html=True)
            st.success(f"{len(prediction_data)} rows predicted successfully.")

        else:
            st.error("CSV file must contain all required columns.")

    else:
        st.info("Upload a CSV file to get predictions.")

# Tab3 - Model Accuracy
with tab3:
    st.markdown('<div class="section">Model Accuracy Comparison</div>', unsafe_allow_html=True)
    import plotly.express as px

    # Load model accuracies dynamically from the training notebook's output
    import json
    with open("../model_accuracies.json", "r") as f:
        data = json.load(f)

    # Get the model names from the dictionary
    Models = list(data.keys())

    # Get the accuracy values from the dictionary
    Accuracies = list(data.values())

    # Create a DataFrame using model names and accuracy values
    df = pd.DataFrame(zip(Models, Accuracies), columns=['Models', 'Accuracies'])

    # Create a bar chart
    # x-axis = Model names
    # y-axis = Accuracy values
    fig = px.bar(df, x='Models', y='Accuracies', title='Model Accuracy Comparison', text='Accuracies', color_discrete_sequence=['crimson'])

    # Display accuracy values on top of each bar
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    # Match the app's dark theme (black chart background, white text, crimson bars)
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white', style='italic', size=14), title_font=dict(color='white', size=22), xaxis=dict(showgrid=False, color='white', linecolor='white'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.15)', color='white', linecolor='white'), margin=dict(t=60, b=40))

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, width="stretch")

