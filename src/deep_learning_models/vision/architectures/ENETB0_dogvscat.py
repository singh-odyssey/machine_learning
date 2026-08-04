import keras
from keras import Sequential
from keras.layers import Dense, GlobalAveragePooling2D , Dropout 
from keras.applications.efficientnet_v2 import EfficientNetV2B0
from keras.utils import image_dataset_from_directory


train_dataset = image_dataset_from_directory(
    directory='/content/catsvsdogs/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224)
)

test_dataset = image_dataset_from_directory(
    directory='/content/catsvsdogs/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224)
)

base_model = EfficientNetV2B0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    # Dense(64, activation='relu'),
    # Dropout(0.2),
    # Dense(32, activation='relu'),
    # Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history=model.fit(
    train_dataset,
    epochs=2,
    validation_data=test_dataset
)


# 99.24 Accuracy 
# # dataset
# !pip install --upgrade kaggle
# !kaggle datasets download salader/dogsvscats --unzip --force