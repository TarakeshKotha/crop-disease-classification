import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("crop_disease_model.h5")

# Class names (ORDER IS IMPORTANT)
class_names = [
    "Potato__Late_blight",
    "Tomato___Early_blight",
    "Tomato_healthy"
]

# Load and preprocess image
img_path = "test_images/leaf.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print("Predicted Disease:", predicted_class)
print("Confidence:", round(confidence, 2), "%")