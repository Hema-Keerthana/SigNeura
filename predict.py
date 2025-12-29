import cv2, tensorflow as tf, keras
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable

keras.config.enable_unsafe_deserialization()

@register_keras_serializable()
def abs_diff(tensors):
    return tf.abs(tensors[0] - tensors[1])

model = load_model(
    "D:/Project/models/signature_siamese_model.keras",
    custom_objects={"abs_diff": abs_diff}
)

def prep(p):
    img = cv2.imread(p, 0)
    img = cv2.resize(img, (100, 100))
    img = img.astype("float32") / 255.0
    return img.reshape(1, 100, 100, 1)

s = model.predict([
    prep("D:/Project/augmented_pressure_datasets/person1/genuine3.png"),
    prep("D:/Project/augmented_pressure_datasets/person1/forged5.png")
])[0][0]

print("Score:", s)
print("Result:", "Genuine" if s > 0.5 else "Forged")
