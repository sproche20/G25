import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import procesar_imagen_desde_bytes, CLASSES

# Cargar modelo
model = tf.keras.models.load_model('modelo_mobilenet.h5')

st.title("Clasificador de Rayos X de Tórax")

archivo = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
if archivo is not None:
    imagen_bytes = archivo.read()
    try:
        img = procesar_imagen_desde_bytes(imagen_bytes)
        pred = model.predict(img)[0]
        clase_predicha = CLASSES[np.argmax(pred)]
        st.subheader(f"🧠 Predicción: {clase_predicha}")

        # Mostrar gráfico
        df = pd.DataFrame({'Probabilidades': pred}, index=CLASSES)
        st.bar_chart(df)
    except Exception as e:
        st.error(f"❌ Error: {e}")
