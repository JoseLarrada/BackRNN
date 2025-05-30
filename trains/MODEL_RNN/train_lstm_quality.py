import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ===============================
# CONFIG
# ===============================

DATA_DIR = "dataset_agrupado"  # Carpeta donde están los JSON por número
MAX_SEQ_LEN = 100                # Longitud máxima de secuencia (ajustable)
BATCH_SIZE = 32
EPOCHS = 50

# ===============================
# FUNCIONES
# ===============================

def procesar_stroke(stroke):
    """
    Convierte el trazo en una secuencia de [x, y, t], normalizados.
    """
    xs = [p['x'] for p in stroke]
    ys = [p['y'] for p in stroke]
    ts = [p['t'] for p in stroke]

    # Normalizar entre 0 y 1
    def normalize(arr):
        min_val = min(arr)
        max_val = max(arr)
        return [(val - min_val) / (max_val - min_val) if max_val > min_val else 0.0 for val in arr]

    norm_xs = normalize(xs)
    norm_ys = normalize(ys)
    norm_ts = normalize(ts)

    secuencia = [[x, y, t] for x, y, t in zip(norm_xs, norm_ys, norm_ts)]
    return secuencia

def cargar_datos(data_dir):
    X, y = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            path = os.path.join(data_dir, filename)
            with open(path, 'r') as f:
                muestras = json.load(f)
                for muestra in muestras:
                    stroke = procesar_stroke(muestra['stroke'])
                    X.append(stroke)
                    y.append(muestra['quality'])  # Esperamos un float
    return X, y

# ===============================
# CARGA Y PREPROCESAMIENTO
# ===============================

print("Cargando datos...")
X_raw, y_raw = cargar_datos(DATA_DIR)

# Padding a longitud fija
X_padded = pad_sequences(X_raw, maxlen=MAX_SEQ_LEN, dtype='float32', padding='post', truncating='post')

# Convertir etiquetas
y = np.array(y_raw, dtype='float32')

# División de entrenamiento/test
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# ===============================
# MODELO LSTM
# ===============================

print("Construyendo modelo...")
model = Sequential([
    Masking(mask_value=0.0, input_shape=(MAX_SEQ_LEN, 3)),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)  # Salida: calidad del trazo
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# ===============================
# ENTRENAMIENTO
# ===============================

print("Entrenando...")
history = model.fit(X_train, y_train, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE,
                    validation_split=0.2)

# Guardar el modelo
model.save('model\data_rnn_model.h5')

# ===============================
# EVALUACIÓN
# ===============================
# ===============================
# GRAFICAR PÉRDIDAS
# ===============================

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida (MSE)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.savefig('trains\MODEL_RNN\metrics\loss.png')
plt.legend()

# MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Entrenamiento')
plt.plot(history.history['val_mae'], label='Validación')
plt.title('Error Absoluto Medio (MAE)')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.savefig('trains\MODEL_RNN\metrics\mae.png')
plt.legend()

plt.tight_layout()
plt.savefig("trains\MODEL_RNN\metrics\graficas_entrenamiento.png")
plt.close()

# ===============================
# CONVERTIR A CLASES
# ===============================

def to_class(valor):
    if valor < 40:
        return 0  # Malo
    elif valor < 70:
        return 1  # Medio
    else:
        return 2  # Bueno

y_test_clas = np.array([to_class(q) for q in y_test])
y_pred = model.predict(X_test).flatten()
y_pred_clas = np.array([to_class(q) for q in y_pred])

# ===============================
# MATRIZ DE CONFUSIÓN
# ===============================

matriz = confusion_matrix(y_test_clas, y_pred_clas)

plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malo', 'Medio', 'Bueno'],
            yticklabels=['Malo', 'Medio', 'Bueno'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.savefig("matriz_confusion.png")
plt.close()

# ===============================
# REPORTE DE CLASIFICACIÓN
# ===============================

report = classification_report(y_test_clas, y_pred_clas, target_names=['Malo', 'Medio', 'Bueno'])
# También lo puedes guardar si deseas
with open(r"trains\MODEL_RNN\metrics\reporte_clasificacion.txt", "w") as f:
    f.write(report)


loss, mae = model.evaluate(X_test, y_test)
print(f"\nEvaluación final - Loss: {loss:.2f}, MAE: {mae:.2f}")