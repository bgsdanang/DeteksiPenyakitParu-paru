import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = keras.models.load_model('model_output.h5')

# Function to make predictions
def predict_image(uploaded_image):
    # Load and preprocess the uploaded image
    img = tf.keras.preprocessing.image.load_img(uploaded_image, target_size=(224, 224), color_mode='grayscale')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    prediction = model.predict(img)
    kelas_prediksi = np.argmax(prediction)
    persentase_prediksi = prediction[0][kelas_prediksi] * 100

    return kelas_prediksi, persentase_prediksi

st.title("Deteksi Penyakit Paru-paru")

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload Gambar Paru-paru", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Gambar yang Diunggah", use_column_width=True)
    st.write("")
    st.write("Hasil Prediksi:")
    
    kelas_prediksi, persentase_prediksi = predict_image(uploaded_image)
    
    classes = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
    predicted_class = classes[kelas_prediksi]
    st.write(f"Kelas Prediksi: {predicted_class}")
    st.write(f"Persentase Prediksi: {persentase_prediksi:.2f}%")

# Sidebar
st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("Aplikasi ini dapat digunakan untuk mendeteksi penyakit paru-paru pada gambar rontgen.")

# Footer
st.text("Dibuat oleh danang")
