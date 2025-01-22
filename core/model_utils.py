import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
import joblib
from fastapi import HTTPException
import numpy as np
# Directory for storing trained models
MODEL_DIR = "app/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_models(user_id, task_id):
    """
    Train models for the given task ID and save the best one.
    """
    # Fetch dataset path
    dataset_files = [f for f in os.listdir("app/data") if f.startswith(f"{user_id}_{task_id}")]
    if not dataset_files:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    filepath = os.path.join("app/data", dataset_files[0])
    data = pd.read_csv(filepath)

    # Fetch metadata from the query file
    query_file = "app/queries.csv"
    query_data = pd.read_csv(query_file)
    query_row = query_data[(query_data["user_id"] == user_id) & (query_data["task_id"] == task_id)]

    if query_row.empty:
        raise HTTPException(status_code=404, detail="Task metadata not found.")

    task_type = "classification" if "classification" in dataset_files[0] else "regression"
    output_column = query_row["output_column"].iloc[0]

    if output_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Output column '{output_column}' not found in dataset.")

    X = data.drop(columns=[output_column])
    y = data[output_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    best_model = None
    best_metric = -float("inf") if task_type == "classification" else float("inf")

    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
        }

        for name, clf in models.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary')

            results.append({"model": name, "accuracy": acc, "f1_score": f1})

            if acc > best_metric:
                best_metric = acc
                best_model = clf

    elif task_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
        }

        for name, reg in models.items():
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({"model": name, "mse": mse, "r2_score": r2})

            if mse < best_metric:
                best_metric = mse
                best_model = reg

    if not best_model:
        raise HTTPException(status_code=500, detail="Failed to find a suitable model.")

    # Save the best model
    user_model_dir = os.path.join(MODEL_DIR, user_id)
    os.makedirs(user_model_dir, exist_ok=True)
    model_path = os.path.join(user_model_dir, f"{task_id}.pkl")
    joblib.dump(best_model, model_path)

    # Fetch query example for the task
    query_example = eval(query_row["query"].iloc[0])

    return results, query_example



def load_model(user_id, task_id):
    """
    Load the trained model for a given user and task.
    """
    model_path = os.path.join(MODEL_DIR, user_id, f"{task_id}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found.")
    return joblib.load(model_path)


def make_prediction(user_id, task_id, input_data):
    """
    Make predictions using the trained model.
    """
    # Load the model
    model = load_model(user_id, task_id)

    # Fetch query example for validation
    query_file = "app/queries.csv"
    query_data = pd.read_csv(query_file)
    query_row = query_data[(query_data["user_id"] == user_id) & (query_data["task_id"] == task_id)]

    if query_row.empty:
        raise HTTPException(status_code=404, detail="Task not found for the user.")

    # Validate the input structure
    expected_query = eval(query_row["query"].iloc[0])  # Convert string to dict

    if set(input_data.keys()) != set(expected_query.keys()):
        raise HTTPException(status_code=400, detail={
            "error": "Invalid input structure.",
            "expected_structure": expected_query,
        })

    # Convert input_data to the expected order of features
    input_array = np.array([input_data[feature] for feature in expected_query.keys()]).reshape(1, -1)

    # Make prediction
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_array)[0]
            prediction = np.argmax(probabilities)
            confidence = round(probabilities[prediction], 2)
            return {"prediction": int(prediction), "confidence": confidence}
        else:
            prediction = model.predict(input_array)[0]
            return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
