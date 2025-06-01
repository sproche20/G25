import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Cargar el dataset (ajustar nombre)
print("📂 Cargando dataset...")
data = np.load('dataset_resized_from_excel.npz')
X = data['X']  # (N, 150, 150)
y = data['y']  # (N,)

# 2. Expandir dimensiones (1 canal → 3 canales)
X = X[..., np.newaxis]  # (N, 150, 150, 1)
X = np.concatenate([X, X, X], axis=-1)  # (N, 150, 150, 3)

# 3. Preprocesar imágenes para MobileNetV2 (rango esperado [-1, 1])
X = X * 255.0
X = preprocess_input(X)

# 4. Separar datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Cargar MobileNetV2 base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# 6. Construir modelo completo
inputs = tf.keras.Input(shape=(150, 150, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# 7. Compilar con sparse_categorical_crossentropy
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Entrenar
print("🚀 Entrenando modelo...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# 9. Evaluar
loss, acc = model.evaluate(X_val, y_val)
print(f"\n📊 Evaluación final - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# 10. Guardar modelo y convertir a TFLite
model.save('modelo_mobilenet.h5')
print("✅ Modelo guardado como modelo_mobilenet.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('modelo_covid.tflite', 'wb') as f:
    f.write(tflite_model)
print("📦 Modelo convertido a TensorFlow Lite como modelo_covid.tflite")
