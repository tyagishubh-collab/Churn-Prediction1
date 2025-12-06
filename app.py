from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)
INPUT_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "tenure",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def load_and_train_model():
 
    df = pd.read_csv("Churn.csv")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])

    target_col = "Churn"
    df = df.dropna(subset=[target_col])
    y = df[target_col].map({"No": 0, "Yes": 1})
    X = df.copy()
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])
    available_features = [col for col in INPUT_FEATURES if col in X.columns]
    X = X[available_features]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=12,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X, y)

    stats = {
        "tenure_max": float(df["tenure"].max()) if "tenure" in df.columns else 72.0,
        "monthly_min": float(df["MonthlyCharges"].min())
        if "MonthlyCharges" in df.columns
        else 0.0,
        "monthly_max": float(df["MonthlyCharges"].max())
        if "MonthlyCharges" in df.columns
        else 120.0,
        "total_min": float(df["TotalCharges"].min())
        if "TotalCharges" in df.columns
        else 0.0,
        "total_max": float(df["TotalCharges"].max())
        if "TotalCharges" in df.columns
        else 8000.0,
    }

    return clf, available_features, stats


model, MODEL_FEATURES, TRAIN_STATS = load_and_train_model()



def build_input_row(payload: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame from frontend JSON
    using the same feature names and types as training.
    """
    row = {}

  
    for f in MODEL_FEATURES:
        row[f] = None

    mapping = {
        "gender": "gender",
        "SeniorCitizen": "SeniorCitizen",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "PhoneService": "PhoneService",
        "MultipleLines": "MultipleLines",
        "tenure": "tenure",
        "InternetService": "InternetService",
        "OnlineSecurity": "OnlineSecurity",
        "OnlineBackup": "OnlineBackup",
        "DeviceProtection": "DeviceProtection",
        "TechSupport": "TechSupport",
        "StreamingTV": "StreamingTV",
        "StreamingMovies": "StreamingMovies",
        "Contract": "Contract",
        "PaperlessBilling": "PaperlessBilling",
        "PaymentMethod": "PaymentMethod",
        "MonthlyCharges": "MonthlyCharges",
        "TotalCharges": "TotalCharges",
    }

    for key, col in mapping.items():
        if col not in MODEL_FEATURES:
            continue
        value = payload.get(key, None)
        if value is None:
            row[col] = None
        else:
            if key in ["SeniorCitizen", "tenure"]:
                try:
                    row[col] = int(value)
                except Exception:
                    row[col] = 0
            elif key in ["MonthlyCharges", "TotalCharges"]:
                try:
                    row[col] = float(value)
                except Exception:
                    row[col] = 0.0
            else:
                row[col] = value

    return pd.DataFrame([row], columns=MODEL_FEATURES)


def build_feature_contributions(payload: dict, probability: float):
    """
    Heuristic per-feature 'contributions' and explanations for the UI.
    These are not true SHAP values but reasonable interpretable scores
    based on domain intuition.
    Scores roughly in [-1, 1], where >0.1 = risk, <-0.1 = protective.
    """
    contribs = []

    tenure = float(payload.get("tenure", 0) or 0)
    tenure_max = TRAIN_STATS["tenure_max"] or 72.0
    tenure_norm = min(max(tenure / tenure_max, 0.0), 1.0)
  
    tenure_score = 0.5 - tenure_norm
    contribs.append(
        {
            "feature": "Tenure",
            "score": float(tenure_score),
            "explanation": "Shorter tenure customers tend to have higher churn risk.",
        }
    )

 
    mc = float(payload.get("MonthlyCharges", 0) or 0.0)
    mc_min, mc_max = TRAIN_STATS["monthly_min"], TRAIN_STATS["monthly_max"]
    if mc_max > mc_min:
        mc_norm = (mc - mc_min) / (mc_max - mc_min)
    else:
        mc_norm = 0.5
   
    charges_score = mc_norm - 0.5
    contribs.append(
        {
            "feature": "MonthlyCharges",
            "score": float(charges_score),
            "explanation": "Higher monthly charges may increase churn risk if value perception is low.",
        }
    )

    
    tc = float(payload.get("TotalCharges", 0) or 0.0)
    tc_min, tc_max = TRAIN_STATS["total_min"], TRAIN_STATS["total_max"]
    if tc_max > tc_min:
        tc_norm = (tc - tc_min) / (tc_max - tc_min)
    else:
        tc_norm = 0.5
    total_score = 0.5 - tc_norm
    contribs.append(
        {
            "feature": "TotalCharges",
            "score": float(total_score),
            "explanation": "Long-tenure, high-value customers are typically less likely to churn.",
        }
    )

  
    contract = payload.get("Contract", "Month-to-month")
    if contract == "Month-to-month":
        c_score = 0.4
        c_exp = "Month-to-month contracts are strongly associated with higher churn risk."
    elif contract == "One year":
        c_score = -0.1
        c_exp = "One-year contracts usually reduce churn risk."
    else: 
        c_score = -0.2
        c_exp = "Two-year contracts usually indicate lower churn risk."
    contribs.append({"feature": "Contract", "score": float(c_score), "explanation": c_exp})

    
    inet = payload.get("InternetService", "Fiber optic")
    if inet == "Fiber optic":
        i_score = 0.15
        i_exp = "Fiber customers sometimes show higher churn if service issues are perceived."
    elif inet == "DSL":
        i_score = 0.05
        i_exp = "DSL customers show moderate churn risk."
    else: 
        i_score = -0.05
        i_exp = "No internet service reduces churn risk for internet-related reasons."
    contribs.append(
        {"feature": "InternetService", "score": float(i_score), "explanation": i_exp}
    )

  
    tech = payload.get("TechSupport", "No")
    sec = payload.get("OnlineSecurity", "No")
    support_score = 0.0
    if tech == "No":
        support_score += 0.15
    if sec == "No":
        support_score += 0.15
    if tech == "Yes":
        support_score -= 0.05
    if sec == "Yes":
        support_score -= 0.05

    contribs.append(
        {
            "feature": "Support & Security",
            "score": float(support_score),
            "explanation": "Lack of tech support or online security can increase churn risk.",
        }
    )


    return contribs


def build_radar_scores(payload: dict):
    """
    Build radar_scores object expected by your JS.
    All values in [0, 1], where higher ~ worse risk.
    """
    tenure = float(payload.get("tenure", 0) or 0)
    tenure_max = TRAIN_STATS["tenure_max"] or 72.0
    tenure_risk = 1.0 - min(max(tenure / tenure_max, 0.0), 1.0)

    mc = float(payload.get("MonthlyCharges", 0) or 0.0)
    mc_min, mc_max = TRAIN_STATS["monthly_min"], TRAIN_STATS["monthly_max"]
    if mc_max > mc_min:
        charges_risk = (mc - mc_min) / (mc_max - mc_min)
    else:
        charges_risk = 0.5
    charges_risk = float(min(max(charges_risk, 0.0), 1.0))

    tc = float(payload.get("TotalCharges", 0) or 0.0)
    tc_min, tc_max = TRAIN_STATS["total_min"], TRAIN_STATS["total_max"]
    if tc_max > tc_min:
        total_risk = 1.0 - (tc - tc_min) / (tc_max - tc_min)
    else:
        total_risk = 0.5
    total_risk = float(min(max(total_risk, 0.0), 1.0))

    
    usage_features = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "StreamingTV",
        "StreamingMovies",
    ]
    usage_count = sum(1 for f in usage_features if payload.get(f, "No") == "Yes")
    service_usage = usage_count / len(usage_features)

    contract = payload.get("Contract", "Month-to-month")
    contract_type = 0.9 if contract == "Month-to-month" else (0.4 if contract == "One year" else 0.2)

    tech = payload.get("TechSupport", "No")
    sec = payload.get("OnlineSecurity", "No")
    support_avail = 0.0
    if tech == "No":
        support_avail += 0.5
    if sec == "No":
        support_avail += 0.5
    support_avail = float(min(max(support_avail, 0.0), 1.0))

    radar_scores = {
        "tenure": float(tenure_risk),
        "charges": float(charges_risk),
        "total_charges": float(total_risk),
        "service_usage": float(service_usage),
        "contract_type": float(contract_type),
        "support_availability": float(support_avail),
    }
    return radar_scores

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        X_new = build_input_row(payload)
        proba = model.predict_proba(X_new)[0, 1]
        pred = int(proba >= 0.5)

        feature_contributions = build_feature_contributions(payload, proba)
        radar_scores = build_radar_scores(payload)

        response = {
            "probability": float(proba),
            "prediction": pred,
            "feature_contributions": feature_contributions,
            "radar_scores": radar_scores,
        }
        return jsonify(response)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": "Prediction failed",
                    "details": str(e),
                }
            ),
            500,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
