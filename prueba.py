import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 1. Cargar dataset procesado
data = np.load("dataset_resized_from_excel.npz")
X = data['X']
y = data['y']

print("📦 Imágenes:", X.shape)
print("🏷️ Etiquetas:", y.shape)
print("🔍 Ejemplo de etiqueta numérica:", y[0])

# 2. Convertir etiquetas a one-hot (formato para clasificación multicategoría)
y_categorical = to_categorical(y, num_classes=4)  # porque hay 4 clases

# 3. Dividir entre entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

print("✅ Entrenamiento:", X_train.shape, y_train.shape)
print("✅ Prueba:", X_test.shape, y_test.shape)

# Visualizar una imagen con su etiqueta
import matplotlib.pyplot as plt

plt.imshow(X[0], cmap='gray')
from sklearn.preprocessing import LabelEncoder

# Crear el codificador y ajustarlo con las clases reales
label_encoder = LabelEncoder()
label_encoder.fit([ 'COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia' ])
plt.title(f"Etiqueta: {label_encoder.inverse_transform([int(y[0])])[0]}")
plt.axis('off')
plt.show()

