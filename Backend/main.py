from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
import pickle
import pandas as pd
import numpy as np

# Create fastapi application
app = FastAPI()

# Loading the models
logistic_model = pickle.load(open("LogistiRegression.pkl", "rb"))
decision_tree_model = pickle.load(open("Decision_Tree.pkl", "rb"))
random_forest_model = pickle.load(open("Random_Forest.pkl", "rb"))
knn_imputer = pickle.load(open("KNN_Imputer.pkl", "rb"))


# Get the home page '/'
@app.get("/")
def home():
    return {
        "message": "Heart Disease Backend is running"
    }


# Single Prediction
@app.post("/predict")
def predict(data: dict):
    try:
        age = data["Age"]
        sex = data["Sex"]
        chest_pain_type = data["ChestPainType"]
        resting_bp = data["RestingBP"]
        cholesterol = data["Cholesterol"]
        fasting_bs = data["FastingBS"]
        resting_ecg = data["RestingECG"]
        max_hr = data["MaxHR"]
        exercise_angina = data["ExerciseAngina"]
        oldpeak = data["Oldpeak"]
        st_slope = data["ST_Slope"]

        # Create DataFrame using the SAME column names used during training
        input_data = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [chest_pain_type],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [fasting_bs],
            "RestingECG": [resting_ecg],
            "MaxHR": [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_slope]
        })

        # Replace invalid 0 values with NaN
        input_data["Cholesterol"] = input_data["Cholesterol"].replace(0, np.nan)
        input_data["RestingBP"] = input_data["RestingBP"].replace(0, np.nan)

        # Apply the knn imputer used during training
        input_data = pd.DataFrame(knn_imputer.transform(input_data), columns=input_data.columns)

        # Predictions
        logistic_prediction = logistic_model.predict(input_data)
        decision_tree_prediction = decision_tree_model.predict(input_data)
        random_forest_prediction = random_forest_model.predict(input_data)

        return {
            "Logistic Regression": int(logistic_prediction[0]),
            "Decision Tree": int(decision_tree_prediction[0]),
            "Random Forest": int(random_forest_prediction[0])
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Bulk Prediction
@app.post("/predict-bulk")
def predict_bulk(data: List[Dict[str, Any]]):
    try:
        input_data = pd.DataFrame(data)
        input_data["Cholesterol"] = input_data["Cholesterol"].replace(0, np.nan)
        input_data["RestingBP"] = input_data["RestingBP"].replace(0, np.nan)
        input_data = pd.DataFrame(knn_imputer.transform(input_data), columns=input_data.columns)

        logistic_prediction = logistic_model.predict(input_data)
        decision_tree_prediction = decision_tree_model.predict(input_data)
        random_forest_prediction = random_forest_model.predict(input_data)

        input_data["Logistic Regression"] = logistic_prediction
        input_data["Decision Tree"] = decision_tree_prediction
        input_data["Random Forest"] = random_forest_prediction

        return input_data.to_dict(orient="records")

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")