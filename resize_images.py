import os
import cv2
import numpy as np
import pandas as pd

# Ruta a las carpetas
BASE_DIR = 'ruta/a/COVID-19 Radiography Database'
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']
IMG_SIZE = 150

data = []  # Lista donde guardaremos imágenes y etiquetas

for label in CLASSES:
    folder = os.path.join(BASE_DIR, label)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # leer en blanco y negro
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))        # redimensionar
            data.append([img, label])
        except Exception as e:
            print(f"Error con {filename}: {e}")

# Guardamos en arrays NumPy
X = np.array([item[0] for item in data]) / 255.0  # Normalizar a [0, 1]
y = np.array([item[1] for item in data])

# Guardar como .npz
np.savez_compressed("dataset_resized.npz", X=X, y=y)

print("Dataset procesado y guardado como dataset_resized.npz")