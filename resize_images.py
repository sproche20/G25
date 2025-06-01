import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configuración
BASE_DIR = r'D:/Dataset_COVID/COVID-19_Radiography_Dataset'
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
EXCEL_FILES = {
    'COVID': 'COVID.metadata.xlsx',
    'Lung_Opacity': 'Lung_Opacity.metadata.xlsx',
    'Normal': 'Normal.metadata.xlsx',
    'Viral Pneumonia': 'Viral Pneumonia.metadata.xlsx'
}
IMG_SIZE = 150
BATCH_SIZE = 1024

# Cargar nombres desde Excel
def cargar_nombres_desde_excel(clase):
    excel_path = os.path.join(BASE_DIR, EXCEL_FILES[clase])
    df = pd.read_excel(excel_path)
    nombres = df['FILE NAME'].astype(str).tolist()
    return nombres

# Procesar imagen
def procesar_imagen(path_label_tuple):
    img_path, label_index = path_label_tuple
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return (img, label_index)
    except:
        return None

# Generador de lotes
def cargar_imagenes_por_lotes(batch_size=BATCH_SIZE):
    rutas_y_labels = []
    for i, label in enumerate(CLASSES):
        folder = os.path.join(BASE_DIR, label, 'images')
        if not os.path.exists(folder): continue
        nombres = cargar_nombres_desde_excel(label)
        for nombre in nombres:
            ruta_imagen = os.path.join(folder, nombre + '.png')
            if os.path.exists(ruta_imagen):
                rutas_y_labels.append((ruta_imagen, i))

    total = len(rutas_y_labels)
    print(f"🔢 Total imágenes listadas encontradas: {total}")

    for i in range(0, total, batch_size):
        batch = rutas_y_labels[i:i+batch_size]
        with ThreadPoolExecutor() as executor:
            resultados = list(executor.map(procesar_imagen, batch))
        batch_imgs, batch_labels = zip(*[r for r in resultados if r is not None])
        yield np.array(batch_imgs, dtype=np.float32) / 255.0, np.array(batch_labels)

# Guardar por lotes
X_total, y_total = [], []

print("🚀 Cargando imágenes por lotes desde los archivos Excel...\n")
for X_batch, y_batch in tqdm(cargar_imagenes_por_lotes()):
    X_total.append(X_batch)
    y_total.append(y_batch)

# Concatenar
X_final = np.concatenate(X_total, axis=0)
y_final = np.concatenate(y_total, axis=0)

# Guardar
np.savez_compressed("dataset_resized_from_excel.npz", X=X_final, y=y_final)
print(f"\n✅ Dataset final guardado con {X_final.shape[0]} imágenes.")
