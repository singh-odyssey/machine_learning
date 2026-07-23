import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.ops import sparse_categorical_crossentropy
from src.utils.supervised_metrics import evaluate_classification
from src.utils.tracker import log_experiment
from torch import flatten

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# changing value of each image from 0 to 1
X_train = X_train / 255
X_test = X_test / 255

# model init
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(10, activation="softmax"))

# compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy" )

# train
model.fit(X_train, y_train, epochs=20, validation_split=0.2 , batch_size=1024)
# predict
y = model.predict(X_test,batch_size=1024)
y_pred = y.argmax(axis=1)

metrics = evaluate_classification(y_test, y_pred)
log_experiment(model, metrics, "mnist_ANN_v1")
