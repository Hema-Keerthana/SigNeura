import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.saving import register_keras_serializable
from sklearn.model_selection import train_test_split
from dataset_pairs import create_signature_pairs

@register_keras_serializable()
def abs_diff(tensors):
    return tf.abs(tensors[0] - tensors[1])

(X1, X2), y = create_signature_pairs("D:/Project/augmented_pressure_datasets")

X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
    X1, X2, y, test_size=0.2, random_state=42
)

def base_network(shape):
    i = Input(shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(i)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    return Model(i, x)

shape = X1_tr.shape[1:]
base = base_network(shape)

a, b = Input(shape), Input(shape)
fa, fb = base(a), base(b)
dist = layers.Lambda(abs_diff)([fa, fb])
out = layers.Dense(1, activation="sigmoid")(dist)

model = Model([a, b], out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)


os.makedirs("D:/Project/models", exist_ok=True)
path = "D:/Project/models/signature_siamese_model.keras"

model.fit(
    [X1_tr, X2_tr], y_tr,
    validation_data=([X1_te, X2_te], y_te),
    epochs=30, batch_size=16,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(path, save_best_only=True)
    ]
)

model.save(path)
print("✅ Model trained & saved")
