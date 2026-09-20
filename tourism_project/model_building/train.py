import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Start local MLflow tracking
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("Wellness_Tourism_Purchase_Prediction")

# Load prepared splits
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = Xtrain.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=['object', 'category']).columns.tolist()

# Calculate scale_pos_weight for XGBoost to handle class imbalance natively
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Build Preprocessing Pipeline
num_transformer = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
cat_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False))

preprocessor = make_column_transformer(
    (num_transformer, numeric_features),
    (cat_transformer, categorical_features)
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric='logloss')
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameters
param_dist = {
    'xgbclassifier__n_estimators': [100, 200, 300],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1]
}

with mlflow.start_run(run_name="Tuned_XGBoost_Pipeline"):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(model_pipeline, param_distributions=param_dist, n_iter=5, cv=cv, scoring='f1', random_state=42)
    random_search.fit(Xtrain, ytrain)

    best_model = random_search.best_estimator_
    mlflow.log_params(random_search.best_params_)

    # Shift threshold slightly down to capture more true positive buyers (Recall priority)
    classification_threshold = 0.45
    y_pred_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred = (y_pred_proba >= classification_threshold).astype(int)

    mlflow.log_metrics({
        "accuracy": accuracy_score(ytest, y_pred),
        "precision": precision_score(ytest, y_pred),
        "recall": recall_score(ytest, y_pred),
        "f1-score": f1_score(ytest, y_pred),
        "roc_auc": roc_auc_score(ytest, y_pred_proba)
    })

    # CRITICAL FIX: Use CLOUDPICKLE serialization to bypass MLflow/skops UntrustedTypesFoundException
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )

    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(best_model, "tourism_project/deployment/best_model.joblib")
    
    print("\n--- MODEL EVALUATION ---")
    print(classification_report(ytest, y_pred))
    print("Model pipeline saved successfully to deployment directory.")
