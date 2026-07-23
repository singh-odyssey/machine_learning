import pandas as pd
from sklearn.model_selection import train_test_split
from data_processing.classical_ml.student_analysis_pipl import preprocessor
from sklearn.ensemble import RandomForestRegressor
from src.utils.supervised_metrics import evaluate_regression
from src.utils.tracker import log_experiment
from sklearn.pipeline import Pipeline
import joblib
from src.utils.paths import DATA_DIR_PROCESS, CML_MODELS_DIR, RESULTS_DIR

# saving path for files

file_import_path = DATA_DIR_PROCESS / "StudentPerformanceFactor.csv"
Final_model_path = CML_MODELS_DIR / "marks_predict_RFR_v1.joblib"

# importing file
data = pd.read_csv(file_import_path)

features_to_use = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Teacher_Quality",
    "Distance_from_Home",
    "Parental_Involvement",
]

# spliting
X = data[features_to_use]
y = data["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# preprocessing (master pipeline)
model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100, n_jobs=-2, max_depth=10, min_samples_leaf=5
            ),
        ),
    ]
)

# time to train
model_pipeline.fit(X_train, y_train)

# predict
prediction = model_pipeline.predict(X_test)

# evaluate model
evaluation = evaluate_regression(y_test, prediction)

# saving metrics 
log_experiment(model_pipeline,evaluation,"marks_predict_RFR_v1")

# Saving the model
joblib.dump(model_pipeline, Final_model_path)
print(f"Model saved to: {Final_model_path}")
