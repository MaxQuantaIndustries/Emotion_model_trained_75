import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ====== CONFIG ======
MODEL_PATH = "best_model.h5"
IMG_SIZE = (224, 224)
CLASS_LABELS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise', 'disgust', 'fear']

# ====== LOAD MODEL ======
@st.cache_resource
def load_emotion_model():
    model = load_model(MODEL_PATH)
    return model

model = load_emotion_model()

# ====== STREAMLIT APP ======
st.title("🙂 Facial Emotion Recognition App")

st.markdown("Upload a **face image**, and the model will predict the **emotion**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ====== Preprocess Image ======
    img_array = np.array(image)
    resized_img = cv2.resize(img_array, IMG_SIZE)
    normalized_img = resized_img / 255.0
    input_tensor = np.expand_dims(normalized_img, axis=0)  # shape: (1, 224, 224, 3)

    # ====== Predict ======
    predictions = model.predict(input_tensor)
    predicted_index = np.argmax(predictions)
    predicted_emotion = CLASS_LABELS[predicted_index]

    st.subheader("🔍 Prediction Result:")
    st.success(f"**Emotion Detected: {predicted_emotion}**")

    # ====== Show Prediction Probabilities ======
    st.subheader("📊 Class Probabilities")
    pred_dict = {label: float(f"{prob:.4f}") for label, prob in zip(CLASS_LABELS, predictions[0])}
    st.bar_chart(pred_dict)
