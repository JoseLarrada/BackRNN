import json
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# --- 1. Carga de datos ---

folder_path = r'agrupados\feed'
all_data = []

for i in range(10):
    file_path = os.path.join(folder_path, f'resultado_total_{i}_feedback.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_data.extend(data)

print(f"Total registros cargados: {len(all_data)}")

# --- 2. Preprocesamiento y etiquetado del feedback ---

def categorize_feedback(fb):
    fb = fb.lower()
    if "dificultades" in fb:
        return 0
    elif "recomendable" in fb:
        return 1
    elif "preciso" in fb:
        return 2
    elif "excelente" in fb:
        return 3
    else:
        return 1  # clase por defecto si no coincide

X = []
y_label = []
y_feedback = []

for d in all_data:
    p = d["promedios"]
    features = [
        p["longitud_total"],
        p["tiempo_total"],
        p["cambios_direccion"],
        p["curvatura_promedio"],
        p["simetria_horizontal"],
        d["quality"],
        int(d["label"])
    ]
    X.append(features)
    y_label.append(int(d["label"]))  # número a predecir
    y_feedback.append(categorize_feedback(d["feedback"]))  # feedback numérico

X = np.array(X)
y_label = np.array(y_label)
y_feedback = np.array(y_feedback)

# --- 3. Normalización ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- GUARDA EL SCALER JUSTO AQUÍ ---
with open(r"trains\MODEL_FEED\metrics\feed_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler guardado como 'feed_scaler.pkl'")

# --- 4. División de datos ---

X_train, X_test, y_label_train, y_label_test, y_feedback_train, y_feedback_test = train_test_split(
    X_scaled, y_label, y_feedback, test_size=0.2, random_state=42
)

# --- 5. Modelo Multi-Output ---

inputs = Input(shape=(7,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)

output_num = Dense(10, activation='softmax', name='num_output')(x)
output_feedback = Dense(4, activation='softmax', name='feedback_output')(x)

model = Model(inputs=inputs, outputs=[output_num, output_feedback])

model.compile(
    optimizer='adam',
    loss={
        'num_output': 'sparse_categorical_crossentropy',
        'feedback_output': 'sparse_categorical_crossentropy',
    },
    metrics={
        'num_output': 'accuracy',
        'feedback_output': 'accuracy'
    }
)

# --- 6. Entrenamiento ---

history = model.fit(
    X_train,
    {'num_output': y_label_train, 'feedback_output': y_feedback_train},
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

# Guardar el modelo
os.makedirs(r'trains\MODEL_FEED\metrics', exist_ok=True)
model.save(r'trains\MODEL_FEED\metrics\feed_multioutput_model.h5')

# =============================
# GRÁFICAS DE ENTRENAMIENTO
# =============================

# Precisión
plt.figure(figsize=(8, 5))
plt.plot(history.history['num_output_accuracy'], label='Precisión número (train)')
plt.plot(history.history['val_num_output_accuracy'], label='Precisión número (val)')
plt.plot(history.history['feedback_output_accuracy'], label='Precisión feedback (train)')
plt.plot(history.history['val_feedback_output_accuracy'], label='Precisión feedback (val)')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.savefig(r'trains\MODEL_FEED\metrics\grafica_precision_multi.png')
plt.close()

# Pérdida
plt.figure(figsize=(8, 5))
plt.plot(history.history['num_output_loss'], label='Pérdida número (train)')
plt.plot(history.history['val_num_output_loss'], label='Pérdida número (val)')
plt.plot(history.history['feedback_output_loss'], label='Pérdida feedback (train)')
plt.plot(history.history['val_feedback_output_loss'], label='Pérdida feedback (val)')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.savefig(r'trains\MODEL_FEED\metrics\grafica_perdida_multi.png')
plt.close()

# =============================
# MATRIZ DE CONFUSIÓN Y REPORTES
# =============================

# Predicciones
y_pred_probs_label, y_pred_probs_feedback = model.predict(X_test)
y_pred_label = np.argmax(y_pred_probs_label, axis=1)
y_pred_feedback = np.argmax(y_pred_probs_feedback, axis=1)

# Matriz de confusión para número
cm_label = confusion_matrix(y_label_test, y_pred_label)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_label, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - NÚMERO")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.savefig(r"trains\MODEL_FEED\metrics\confusion_matrix_num.png")
plt.close()

# Matriz de confusión para feedback
cm_feedback = confusion_matrix(y_feedback_test, y_pred_feedback)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_feedback, annot=True, fmt="d", cmap="Greens")
plt.title("Matriz de Confusión - FEEDBACK")
plt.xlabel("Feedback Predicho")
plt.ylabel("Feedback Verdadero")
plt.savefig(r"trains\MODEL_FEED\metrics\confusion_matrix_feedback.png")
plt.close()

# Reportes de clasificación
report_label = classification_report(y_label_test, y_pred_label, digits=4)
report_feedback = classification_report(y_feedback_test, y_pred_feedback, digits=4)
print("📊 Reporte de Clasificación (NÚMERO):\n")
print(report_label)
print("📊 Reporte de Clasificación (FEEDBACK):\n")
print(report_feedback)

# Guardar reportes
with open(r"trains\MODEL_FEED\metrics\reporte_clasificacion_num.txt", "w") as f:
    f.write(report_label)
with open(r"trains\MODEL_FEED\metrics\reporte_clasificacion_feedback.txt", "w") as f:
    f.write(report_feedback)

# Final
print("✅ Modelo guardado como 'feed_multioutput_model.h5'")
print("✅ Gráficas guardadas como:")
print("   - 'grafica_precision_multi.png'")
print("   - 'grafica_perdida_multi.png'")
print("   - 'confusion_matrix_num.png'")
print("   - 'confusion_matrix_feedback.png'")
print("✅ Reportes guardados como:")
print("   - 'reporte_clasificacion_num.txt'")
print("   - 'reporte_clasificacion_feedback.txt'")

# --- UTILIDAD PARA PREDICCIÓN DE NUEVOS DATOS ---
def predict_new_number_and_feedback(features, scaler, model):
    """features: lista con los 7 features en orden"""
    X_new = np.array([features])
    X_new_scaled = scaler.transform(X_new)
    pred_probs = model.predict(X_new_scaled)
    pred_num = np.argmax(pred_probs[0], axis=1)[0]
    pred_feedback = np.argmax(pred_probs[1], axis=1)[0]
    feedback_dict = {
        0: "Dificultades",
        1: "Recomendable",
        2: "Preciso",
        3: "Excelente"
    }
    return pred_num, feedback_dict[pred_feedback]