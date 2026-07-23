import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils.paths import DATA_DIR_PROCESS
import pandas as pd

def get_preprocessed_data(target_col: str = "Exited", test_size: float = 0.2):
  
    try:
        data = pd.read_csv(DATA_DIR_PROCESS / "cust_churn_cc.csv")
    except FileNotFoundError:
        print("file not found")
        sys.exit(1)

    # 2. Feature Engineering (Zero-Balance Flag)
    if "Balance" in data.columns:
        data["is_zero_balance"] = (data["Balance"] == 0).astype(int)

    # Separate Features (X) and Target (y)
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 3. Dynamic Column Detection
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # # Move low-cardinality numerical columns (e.g. HasCrCard, is_zero_balance) to categorical
    # for col in numeric_features.copy():
    #     if X[col].nunique() < 10:
    #         numeric_features.remove(col)
    #         categorical_features.append(col)

    # 4. Define Sub-Pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # 5. Build Unified ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # 6. Train/Validation Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
    )

    # 7. Fit & Transform 
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_test)

    return X_train_scaled, X_val_scaled, y_train, y_test, preprocessor

if __name__ == "__main__":
    get_preprocessed_data()