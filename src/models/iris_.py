import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from src.utils.paths import MODELS_DIR

iris = load_iris(as_frame=True)

feature_cols = iris.data[["petal length (cm)", "petal width (cm)"]]

X = feature_cols
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4212
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train_scaled, y_train)

knn_predict = knn_model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, knn_predict))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, knn_predict))
print("\nClassification Report:\n", classification_report(y_test, knn_predict))


log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
log_reg_predict = log_reg.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, log_reg_predict))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, log_reg_predict))
print("\nClassification Report:\n", classification_report(y_test, log_reg_predict))

svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
svm_predict = svm_model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, svm_predict))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, svm_predict))
print("\nClassification Report:\n", classification_report(y_test, svm_predict))

joblib.dump(knn_model,MODELS_DIR/"iris_knn_v1.joblib")
joblib.dump(log_reg,MODELS_DIR/"iris_log_reg_v1.joblib")
joblib.dump(svm_model,MODELS_DIR/"iris_svm_v1.joblib")