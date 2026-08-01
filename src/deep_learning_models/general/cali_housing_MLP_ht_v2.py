import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.callbacks import EarlyStopping
import keras_tuner as kt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.supervised_metrics import evaluate_regression
from src.utils.tracker import log_experiment
from src.utils.paths import LOGS_DIR

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


# hypertuning
def build_model(hp):
    model = Sequential()

    input_dim = X_train.shape[1]
    model.add(InputLayer(input_shape=(input_dim,)))

    # NUMBER OF HIDDEN LAYERS
    num_layers = hp.Int("num_layers", min_value=1, max_value=4)

    for i in range(num_layers):
        # neurons for each layer
        layer_units = hp.Int(f"units_{i}", min_value=16, max_value=512, step=16)
        model.add(keras.layers.Dense(units=layer_units, activation="relu"))

        # Tune dropout for each layer
        if hp.Boolean(f"dropout_{i}"):
            drop_rate = hp.Float(
                f"drop_rate_{i}", min_value=0.1, max_value=0.5, step=0.1
            )
            model.add(Dropout(rate=drop_rate))

    # THE OUTPUT LAYER
    model.add(Dense(1, activation="linear"))

    # COMPILE
    
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    hp_optimizer = hp.Choice("optimizer", values=["adam", "sgd", "rmsprop", "nadam"])

    
    if hp_optimizer == "adam":
        selected_optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    elif hp_optimizer == "sgd":
        selected_optimizer = keras.optimizers.SGD(
            learning_rate=hp_learning_rate, momentum=0.9
        )
    elif hp_optimizer == "rmsprop":
        selected_optimizer = keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
    elif hp_optimizer == "nadam":
        selected_optimizer = keras.optimizers.Nadam(learning_rate=hp_learning_rate)

    # 3. Compile the model
    model.compile(
        optimizer=selected_optimizer,
        loss="mse",
        metrics=["mae"],
    )

    return model


# tuner
tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=50,
    factor=3,
    directory=LOGS_DIR,
    project_name="cali_housing",
)

# earlystopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=0,
    verbose=True,
    mode="auto",
    restore_best_weights=True,
)

# Execute the search
print("Starting hyperparameter search...")
tuner.search(
    X_train_scaled,
    y_train,
    epochs=1000,
    validation_split=0.2,
    callbacks=[early_stopping],
)

# best hyperparameters found during the search
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("The optimal number of layers is:", best_hps.get("num_layers"))
print("The optimal learning rate is:", best_hps.get("learning_rate"))


# Build a new model using the best hyperparameters
final_model = tuner.hypermodel.build(best_hps)

# Train the final model
history = final_model.fit(
    X_train_scaled,
    y_train,
    epochs=1000,
    validation_split=0.2,
    callbacks=[early_stopping],
)

prediction = final_model.predict(X_test_scaled)

score = evaluate_regression(y_test, prediction)
log_experiment(final_model, score, "cali_housing_MLP_ht_v2")
