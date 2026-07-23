from src.data_processing.deep_learning.cust_churn_cc_pipl import get_preprocessed_data
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from utils.supervised_metrics import evaluate_classification
from utils.tracker import log_experiment

X_train, X_test, y_train, y_test, preprocessor = get_preprocessed_data(
    target_col="Exited", test_size=0.2
)

input_dim = X_train.shape[1]

model = Sequential(
    [
        Input(shape=(X_train.shape[1],)),  # Defines input dimension automatically
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# model.summary()

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100)
prediction=model.predict(X_test)

# Convert probabilities to binary 0/1 predictions
y_pred_binary = (prediction > 0.5).astype(int)

score=evaluate_classification(y_test,y_pred_binary)
print(score)

log_experiment(model,score,"cust_churn_MLP_v1")