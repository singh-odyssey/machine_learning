import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def clip_to_range(values):
    return np.clip(values, a_min=0, a_max=99)


numeric_features = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores", 
    "Tutoring_Sessions",
]
categorical_features = ["Teacher_Quality", "Distance_from_Home", "Parental_Involvement"]

# Corrected numeric_transformer (this pipeline includes the capper for now)
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("capper", FunctionTransformer(func=clip_to_range, validate=False)),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer=Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    ]) 

preprocessor=ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numeric_features),
        ('cat',categorical_transformer,categorical_features)
    ]
)

