import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import lightgbm as lgb
from imblearn.combine import SMOTEENN

import mlflow
import mlflow.sklearn


# ✅ Set experiment
mlflow.set_experiment("LGBM_SMOTE_Experiment")


# ✅ Start MLflow run
with mlflow.start_run():

    # Load data
    df = pd.read_csv("data/dataset.csv")

    X = df.drop("final_label", axis=1)
    y = df["final_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE-ENN
    smote_enn = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

    # Model parameters
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'random_state': 42
    }

    # Log parameters
    mlflow.log_params(params)

    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_res, y_train_res)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", pre)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # ✅ Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, "LGBM_Model")

    print("Model logged & registered in MLflow ✅")


# ✅ Save model locally for deployment (BEST PRACTICE)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")

print("Model saved at models/model.joblib ✅")

# Print one test sample for API testing
sample = X_test.iloc[0].tolist()
print("Test sample for API:")
print(sample)