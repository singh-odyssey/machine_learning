import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from data_processing.classical_ml.student_analysis_pipl import preprocessor
from src.utils.paths import DATA_DIR_PROCESS
from src.utils.supervised_metrics import evaluate_regression
from src.utils.tracker import log_experiment


data = pd.read_csv(DATA_DIR_PROCESS / "StudentPerformanceFactor.csv")

features_to_use = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Teacher_Quality",
    "Distance_from_Home",
    "Parental_Involvement",
]

X = data[features_to_use]
y = data["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("xgboost", xgb.XGBRegressor(random_state=42)),
    ]
)


param_grid = {
    "xgboost__n_estimators": [100, 150, 200],
    "xgboost__learning_rate": [0.01, 0.05, 0.1],
    "xgboost__max_depth": [3, 4, 5],
    "xgboost__subsample": [0.7, 0.8, 0.9],
    "xgboost__colsample_bytree": [0.7, 0.8, 0.9],
    "xgboost__reg_alpha": [0, 0.1, 0.5],
    "xgboost__reg_lambda": [0.5, 1.0, 2.0],
}


print("Starting grid search...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,                
    scoring="r2",        
    n_jobs=-1,            
    verbose=1,
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")


prediction = best_model.predict(X_test)
evaluation = evaluate_regression(y_test, prediction)

print(f"\nTuned XGBoost Evaluation Metrics:")
print(f"  R-squared: {evaluation.get('R-squared') or evaluation.get('r2')}")
print(f"  MSE: {evaluation.get('MSE') or evaluation.get('mse')}")

log_experiment(
    model=best_model,
    metrics=evaluation,
    experiment_name="marks_predict_XGBOOST_tuned_v1",
)