import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("crop_disease_model.h5")

# Class names (must match training folders)
class_names = [
    "Potato__Late_blight",
    "Tomato___Early_blight",
    "Tomato_healthy"
]

# App title
st.title(" Crop Disease Detection System")
st.write("Upload a crop leaf image to identify the disease")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.success(f" Predicted Disease: {predicted_class}")
    st.info(f"🔍 Confidence: {confidence:.2f}%")