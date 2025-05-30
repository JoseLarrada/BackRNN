import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Cargar y preparar datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cambiar forma a (28, 28, 1) y normalizar
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encoding de las etiquetas
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Crear el modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar y guardar el historial
history = model.fit(x_train, y_train_cat, epochs=10, validation_data=(x_test, y_test_cat))

# Guardar el modelo
model.save('mnist_cnn_model.h5')

# =============================
# GRÁFICAS DE ENTRENAMIENTO
# =============================

# Precisión
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.savefig('grafica_precision.png')
plt.close()

# Pérdida
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.savefig('grafica_perdida.png')
plt.close()

# =============================
# MATRIZ DE CONFUSIÓN Y F1-SCORE
# =============================

# Predecir clases (no one-hot)
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Dibujar la matriz
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.savefig("confusion_matrix.png")
plt.close()

# Reporte de clasificación
report = classification_report(y_test, y_pred, digits=4)
print("📊 Reporte de Clasificación:\n")
print(report)

# También lo puedes guardar si deseas
with open("reporte_clasificacion.txt", "w") as f:
    f.write(report)

# Final
print("✅ Modelo guardado como 'mnist_cnn_model.h5'")
print("✅ Gráficas guardadas como:")
print("   - 'grafica_precision.png'")
print("   - 'grafica_perdida.png'")
print("   - 'confusion_matrix.png'")
print("✅ Reporte guardado como 'reporte_clasificacion.txt'")
