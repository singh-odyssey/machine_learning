import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data_processing.classical_ml.student_analysis_pipl import preprocessor
from sklearn.svm import SVR
from src.utils.supervised_metrics import evaluate_regression
from src.utils.tracker import log_experiment
from src.utils.paths import DATA_DIR_PROCESS, CML_MODELS_DIR
import joblib

data = pd.read_csv(DATA_DIR_PROCESS / "StudentPerformanceFactor.csv")

# saving cols to pass as parameters
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

# splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# final model pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("svr", SVR(kernel="rbf", C=1, epsilon=0.5, gamma=0.1)),
    ]
)

# training model
model_pipeline.fit(X_train, y_train)
prediction = model_pipeline.predict(X_test)

# evaluating model
evaluation = evaluate_regression(y_test, prediction)

# saving model
joblib.dump(model_pipeline, CML_MODELS_DIR / "marks_predict_SVR_v1.joblib")
print("model saved")

# saving the metrics
log_experiment(model=model_pipeline, metrics=evaluation, experiment_name="marks_predict_SVR_v1")
