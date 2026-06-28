import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data_processing.student_analysis_pipl import preprocessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# saving path for files
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent

file_import_path = root_dir / "data" / "processed" / "StudentPerformanceFactor.csv"
Final_model_path = root_dir / "models" / "marks_predict_RFR.skops"

# importing file
data = pd.read_csv(file_import_path)

# spliting
X = data.drop(columns=["Exam_Score"])
y = data["Exam_Score"]

X_train, X_test , y_train , y_test =train_test_split(X , y , test_size=0.2,random_state=42)

# preprocessing (master pipeline)
model_pipeline=Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators = 100 , n_jobs = -2 , max_depth = 10 , min_samples_leaf = 3)) 
    ])

# time to train
model_pipeline.fit(X_train, y_train)

# test
score = model_pipeline.score(X_test, y_test)
print(f"Model R-squared Score: {score:.4f}")