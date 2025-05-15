import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64

# 🌱 Page config
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# Function to set background from local image
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: white !important;
    }}
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("bg.jpg")

# 🌱 Sidebar content
st.sidebar.title("🌿 Navigation")
st.sidebar.markdown("""
- 📤 Upload a Leaf Image
- 📈 View Prediction & Confidence
- 💡 Get Care Tips
""")

st.sidebar.markdown("""
*Let's heal the plants, one leaf at a time.*
""")
st.sidebar.info("Made with 💚 by Nature + AI")

# 🌿 Title and description
st.markdown(
    "<h1 style='text-align: center; color: #3E8E41;'>🌿 Plant Disease Diagnosis</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h5 style='text-align: center; color: #4B6043;'>Welcome to the AI-powered Plant Doctor! </h5>",
    unsafe_allow_html=True
)
st.markdown(
    "<h6 style='text-align: center; color: #4B6043;'>Let’s keep your garden green and healthy! 🌼</h6>",
    unsafe_allow_html=True
)

# 🧠 Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# 🌱 Tips dictionary (customize as needed)
plant_tips = {
    "Apple___Apple_scab": "💡 Tip: Prune your apple trees and apply fungicides early in the season.",
    "Apple___Black_rot": "🛡️ Tip: Remove infected branches and clean up fallen debris.",
    "Apple___healthy": "✅ Your plant looks healthy! Keep monitoring for changes.",
    "Corn_(maize)___Common_rust_": "🌽 Tip: Use resistant hybrids and rotate crops to prevent recurrence.",
    "Corn_(maize)___healthy": "🎉 Looks good! Maintain optimal spacing and soil conditions.",
    "Grape___Black_rot": "🍇 Tip: Apply fungicides during early growth and ensure proper air circulation.",
    "Grape___healthy": "👏 Healthy vine! Just maintain regular inspection and pruning."
}

# 📁 Get class names from training dataset directory
@st.cache_resource
@st.cache_data
def get_class_names():
    dataset_dir = "./dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    if os.path.exists(dataset_dir):
        class_names = sorted(os.listdir(dataset_dir))
        return class_names
    else:
        st.error("⚠️ Could not find the training directory to fetch class names.")
        return []

class_names = get_class_names()

# 📸 Upload image
uploaded_file = st.file_uploader("📤 Upload a plant leaf image 🌱", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and class_names:
    try:
        # Display the image
        img = Image.open(uploaded_file)
        st.image(img, caption="📷 Uploaded Leaf Image", use_container_width=True)

        # 🔍 Preprocess image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 🔎 Prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)

        if predicted_index < len(class_names):
            predicted_class = class_names[predicted_index]
            confidence = np.max(predictions)

            # 🌿 Styled output for prediction, confidence, and tips
            st.markdown(f"""
            <div style='padding: 1em; border-radius: 10px; background-color: #e0f2e9; border-left: 5px solid #2e7d32;'>
                <h4 style='color: #2e7d32;'>🌼 <strong>Prediction:</strong> {predicted_class}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='padding: 1em; border-radius: 10px; background-color: #e3f2fd; border-left: 5px solid #0d47a1;'>
                <h4 style='color: #0d47a1;'>📊 <strong>Confidence Level:</strong> {confidence * 100:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
            tip = plant_tips.get(predicted_class, "🪴 General Tip: Ensure proper watering, sunlight, and monitor regularly.")
            st.markdown(f"""
            <div style='padding: 1em; border-radius: 10px; background-color: #f0f4c3; border-left: 5px solid #33691e;'>
                <h4 style='color: #33691e;'>🌱 <strong>Care Tip:</strong></h4>
                <p style='color: #33691e;'>{tip}</p>
            </div>
            """, unsafe_allow_html=True)

            
            if confidence < 0.6:
                st.warning("The model is unsure. Please try a clearer image or different angle.")
        else:
            st.error("🚫 Prediction index out of range. Please check your model and class names.")
    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {e}")
elif not uploaded_file:
    st.info("📩 Upload a leaf image to start diagnosis.")

