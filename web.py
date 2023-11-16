import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = keras.models.load_model('model_website.h5')

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
    
    classes = ["Bukan Gambar X-ray Paru", "COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
    predicted_class = classes[kelas_prediksi]

    st.subheader("Diagnosa:")
    st.write(f"Persentase Prediksi: {persentase_prediksi:.2f}%")

    # Menampilkan informasi tambahan sesuai dengan kelas prediksi
    if kelas_prediksi == 0:
        st.write("Gambar ini bukan merupakan gambar X-ray paru-paru.")
    elif kelas_prediksi == 1:
        st.write("Gambar ini menunjukkan adanya COVID-19.")
        # Anda dapat menambahkan informasi tambahan untuk kelas ini
    elif kelas_prediksi == 2:
        st.write("Gambar ini normal tanpa tanda-tanda penyakit.")
        # Anda dapat menambahkan informasi tambahan untuk kelas ini
    elif kelas_prediksi == 3:
        st.write("Gambar ini menunjukkan tanda-tanda pneumonia.")
        # Anda dapat menambahkan informasi tambahan untuk kelas ini
    elif kelas_prediksi == 4:
        st.write("Gambar ini menunjukkan tanda-tanda tuberkulosis.")
        # Anda dapat menambahkan informasi tambahan untuk kelas ini


# Sidebar
st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("Aplikasi ini dapat digunakan untuk mendeteksi penyakit paru-paru pada gambar rontgen.")

# Footer
st.text("Dibuat oleh danang")
