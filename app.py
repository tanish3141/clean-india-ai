import streamlit as st
import numpy as np
import pickle
from PIL import Image

st.title("🧹 AI for Clean India")
st.write("Upload an image to detect if the area is clean or dirty.")

import tensorflow as tf
model = tf.keras.models.load_model("model/clean_model_tf_v2.keras")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    img = img.resize((128,128))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]
    label = "🧼 CLEAN" if pred <0.7 else "🚮 DIRTY"
    st.subheader(f"Prediction: {label} ")
    st.caption(f"Confidence (dirty): {pred:.3f}")

