import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data_processing.student_analysis_pipl import preprocessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib 
# saving path for files
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent

file_import_path = root_dir / "data" / "processed" / "StudentPerformanceFactor.csv"
Final_model_path = root_dir / "models" / "marks_predict_RFR_v1.joblib"

# importing file
data = pd.read_csv(file_import_path)

features_to_use = [
    "Hours_Studied", "Attendance", "Previous_Scores", "Tutoring_Sessions", 
    "Teacher_Quality", "Distance_from_Home", "Parental_Involvement"
]

# spliting
X = data[features_to_use]
y = data["Exam_Score"]

X_train, X_test , y_train , y_test =train_test_split(X , y , test_size=0.2,random_state=42)

# preprocessing (master pipeline)
model_pipeline=Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators = 100 , n_jobs = -2 , max_depth = 10 , min_samples_leaf = 5)) 
    ])

# time to train
model_pipeline.fit(X_train, y_train)

# predict 
prediction=model_pipeline.predict(X_test)

# test
mae=mean_absolute_error(y_test,prediction)
mse=mean_squared_error(y_test,prediction)
score = model_pipeline.score(X_test, y_test)
print(f"R-squared: {score:.4f}")

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# Saving the model

joblib.dump(model_pipeline, Final_model_path)
print(f"Model saved to: {Final_model_path}")