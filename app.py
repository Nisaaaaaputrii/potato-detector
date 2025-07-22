import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import scipy.stats
from sklearn.preprocessing import StandardScaler

# Fungsi untuk preprocessing dan ekstraksi fitur GLCM + statistik
def load_and_preprocess_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

def extract_glcm_features(gray_image):
    from skimage.feature import graycomatrix, graycoprops
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
    ]
    return features

def additional_statistical_features(gray_img):
    mean = np.mean(gray_img)
    std = np.std(gray_img)
    var = np.var(gray_img)
    skew = scipy.stats.skew(gray_img.ravel())
    kurt = scipy.stats.kurtosis(gray_img.ravel())
    return [mean, std, var, skew, kurt]

# Load model & scaler (buat scaler manual jika perlu)
model = load_model("model_cnn_glcm.h5")

st.title("🟢 Identifikasi Penyakit Daun Kentang (GLCM + CNN)")

uploaded_file = st.file_uploader("Unggah gambar daun kentang", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Diupload", use_column_width=True)

    gray = load_and_preprocess_image(image)
    glcm_feat = extract_glcm_features(gray)
    stat_feat = additional_statistical_features(gray)
    features = np.array(glcm_feat + stat_feat).reshape(1, -1)

    # Normalisasi fitur
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    prediction = model.predict(features_scaled)
    label = ['Early Blight', 'Late Blight', 'Healthy'][np.argmax(prediction)]

    st.success(f"Hasil Prediksi: **{label}**")
