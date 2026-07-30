import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.src.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.supervised_metrics import evaluate_regression
from src.utils.tracker import log_experiment

housing_data = fetch_california_housing(as_frame=True)
print(housing_data)

X = housing_data.data
y = housing_data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

model = Sequential()
input_dim = X_train.shape[1]
model.add(Dense(128, activation="relu", input_dim=input_dim))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="linear"))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=0,
    verbose=True,
    mode="auto",
    restore_best_weights=True,
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=1000,
    validation_split=0.2,
    callbacks=[early_stopping],
)
prediction = model.predict(X_test_scaled)

score = evaluate_regression(y_test, prediction)
log_experiment(model, score, "cali_housing_MLP_v1")
