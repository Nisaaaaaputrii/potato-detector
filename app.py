import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
import scipy.stats

# --- Load Model dan Tools ---
model = load_model("model_cnn_glcm.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --- Fungsi preprocessing gambar ---
def load_and_preprocess_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

def extract_glcm_features(gray_image):
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
    ]
    return features

def extract_statistical_features(gray_img):
    mean = np.mean(gray_img)
    std = np.std(gray_img)
    var = np.var(gray_img)
    skew = scipy.stats.skew(gray_img.ravel())
    kurt = scipy.stats.kurtosis(gray_img.ravel())
    return [mean, std, var, skew, kurt]

# --- Streamlit UI ---
st.title("🟢 Deteksi Penyakit Daun Kentang (CNN + GLCM)")
st.write("Upload gambar daun kentang untuk mengetahui jenis penyakitnya.")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Diupload", use_column_width=True)

    # Preprocessing
    gray = load_and_preprocess_image(image)
    glcm_feat = extract_glcm_features(gray)
    stat_feat = extract_statistical_features(gray)
    all_features = np.array(glcm_feat + stat_feat).reshape(1, -1)

    # Normalisasi dan prediksi
    scaled_features = scaler.transform(all_features)
    prediction = model.predict(scaled_features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    st.subheader("🔍 Hasil Prediksi:")
    st.success(f"Penyakit terdeteksi: **{predicted_label}**")
