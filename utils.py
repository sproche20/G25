# utils.py
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 150
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# Procesa imagen desde ruta (para consola)
def procesar_imagen_desde_ruta(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo cargar la imagen.")
    return _preprocesar(img)

# Procesa imagen desde bytes (para Streamlit)
def procesar_imagen_desde_bytes(file_bytes):
    img = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")
    return _preprocesar(img)

# Función común de preprocesamiento
def _preprocesar(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[..., np.newaxis]  # (150, 150, 1)
    img = np.concatenate([img] * 3, axis=-1)  # (150, 150, 3)
    img = img.astype(np.float32)
    img = img[np.newaxis, ...]  # (1, 150, 150, 3)
    img = preprocess_input(img * 255.0)
    return img
