import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ExifTags
from datetime import datetime
import geocoder
import tensorflow as tf

st.set_page_config(page_title="SwachhAI - Clean India", layout="wide")
st.title("🧹 SwachhAI – Clean India Detection App 🇮🇳")
st.markdown("### AI-powered cleanliness detection and reporting platform for a cleaner India.")

st.markdown("---")

if "count" not in st.session_state:
    st.session_state.count = 0

st.sidebar.header("📈 Dashboard Summary")
st.sidebar.metric("Images Analyzed", st.session_state.count)
st.sidebar.metric("Model", "MobileNetV2")
st.sidebar.metric("Estimated Accuracy", "92%")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Demo", "Future Scope"])

if page == "Demo":
    # existing app code
elif page == "Future Scope":
    st.header("🚀 Future Scalability of Swachh AI")
    st.markdown("""
    - **CCTV/Drone Integration:** Real-time monitoring of public places.
    - **Smart Routing:** AI to predict efficient waste collection paths.
    - **Citizen Portal:** App interface for reporting unclean areas.
    - **Data Insights:** Dashboards for authorities.
    """)


def load_model():
    model = tf.keras.models.load_model("model/clean_model_tf_v2.keras")
    return model

model = load_model()

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_exif_location(image):
    """Extract GPS metadata if available"""
    try:
        img = Image.open(image)
        exif = img._getexif()
        if not exif:
            return None
        gps_data = {}
        for tag, val in exif.items():
            if ExifTags.TAGS.get(tag) == "GPSInfo":
                gps_data = val
        return gps_data if gps_data else None
    except Exception:
        return None

def get_ip_location():
    """Approximate user location via IP"""
    g = geocoder.ip('me')
    return g.city

st.header("📸 Citizen Upload Portal")
uploaded_file = st.file_uploader("Upload a photo of your area:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Try to get location
    gps = extract_exif_location(uploaded_file)
    if gps:
        st.success("✅ GPS data found in photo! (Detected from EXIF)")
        city = "Detected from image metadata"
    else:
        city = get_ip_location()
        if city:
            st.info(f"📍 Detected current location: {city}")
        else:
            city = st.text_input("Enter city name manually:")

if st.button("Analyze Image"):
        st.session_state.count += 1

        # Preprocess and predict
        input_img = preprocess_image(uploaded_file)
        pred_prob = model.predict(input_img)[0][0]  # Single neuron output

        # Decide class based on threshold
        if pred_prob < 0.45:
            label = "🧼 CLEAN"
        elif pred_prob > 0.55:
            label = "🚮 DIRTY"
        else:
            label = "🤔 UNCERTAIN"

        st.markdown(f"### Prediction: {label}")
        st.progress(int(pred_prob * 100))
        st.write(f"Confidence: {pred_prob:.2f}")

        entry = {
            "timestamp": datetime.now(),
            "city": city if city else "Unknown",
            "label": label,
            "confidence": round(float(pred_prob), 2)
        }
        df = pd.DataFrame([entry])
        df.to_csv("reports.csv", mode='a', header=False, index=False)
        st.success("✅ Report saved successfully!")
    
st.markdown("---")



