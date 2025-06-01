import sys
import tensorflow as tf
import numpy as np
from utils import procesar_imagen_desde_ruta, CLASSES

# Cargar modelo
model = tf.keras.models.load_model('modelo_mobilenet.h5')
print("✅ Modelo cargado correctamente.")

# Predicción
def predecir(ruta_imagen):
    try:
        img = procesar_imagen_desde_ruta(ruta_imagen)
        pred = model.predict(img)[0]
        clase_predicha = CLASSES[np.argmax(pred)]
        confianza = np.max(pred) * 100

        print(f"🧠 Predicción: {clase_predicha}")
        print("📈 Probabilidades:")
        for i, prob in enumerate(pred):
            print(f"  - {CLASSES[i]}: {prob:.4f}")
    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python predecir_imagen.py <ruta_imagen>")
    else:
        ruta = sys.argv[1]
        predecir(ruta)
