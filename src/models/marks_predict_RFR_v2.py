import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV , KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from src.data_processing.student_analysis_pipl import preprocessor
from src.utils.paths import DATA_DIR_PROCESS ,MODELS_DIR
from src.utils.supervised_metrics import evaluate_regression
from src.utils.tracker import log_experiment
import joblib

data = pd.read_csv(DATA_DIR_PROCESS / "StudentPerformanceFactor.csv")

# cols to use
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

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# builing model
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("rfr", RandomForestRegressor(random_state=42)),
    ]
)

# hyperparameters
rfr_param = {
    "rfr__n_estimators": [50, 100, 200],
    "rfr__max_depth": [None, 3, 5, 7, 10],
    "rfr__min_samples_split": [2, 5, 10],
    "rfr__max_features": ["sqrt", "log2", 0.5, 1.0],
}

# custom KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# grid_search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=rfr_param,
    cv=cv,
    scoring={
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    },
    refit="r2",
    verbose=1,
)
# training model using all parameters
grid_search.fit(X_train, y_train)

# best 
best_model=grid_search.best_estimator_
# evaluate
y_pred = best_model.predict(X_test)
metrics = evaluate_regression(y_test, y_pred)

# save result
log_experiment(best_model,metrics,"marks_predict_RFR_hypertuning")

# saving model
joblib.dump(best_model,MODELS_DIR/"marks_predict_RFR_v2(hypertuning).joblib")