from fastapi import APIRouter
from schemas.stroke_input import StrokeInput
from utils.extract_features import extract_features
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os

router = APIRouter()

# Cargar modelo y scaler
MODEL_PATH = r"trains\MODEL_FEED\metrics\feed_multioutput_model.h5"
SCALER_PATH = r"trains\MODEL_FEED\metrics\feed_scaler.pkl"

model_feed = load_model(MODEL_PATH)

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler no encontrado en {SCALER_PATH}; asegúrate de guardarlo al entrenar.")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

feedback_map = {
    0: "El trazo presenta dificultades; enfócate en mantener mejor ritmo y dirección.",
    1: "Trazado aceptable, pero es recomendable trabajar la fluidez y consistencia.",
    2: "Trazado preciso, fluido y bien equilibrado.",
    3: "Excelente nivel de detalle y dominio en trazos complejos."
}

@router.post("/predict/feedforward")
def predict_feedforward(data: StrokeInput):
    """
    Endpoint de predicción multi-output: número y feedback del trazo.
    """
    try:
        # Extraer features del trazo
        stroke_data = [p.dict() for p in data.stroke]
        features = extract_features(stroke_data)

        # Armar vector de entrada (orden: igual que en entrenamiento)
        input_values = np.array([[ 
            features["longitud_total"],
            features["tiempo_total"],
            features["cambios_direccion"],
            features["curvatura_promedio"],
            features["simetria_horizontal"],
            data.quality,
            data.label
        ]])

        # Normalizar los features
        input_scaled = scaler.transform(input_values)

        # Predicción multi-output
        prob_num, prob_feedback = model_feed.predict(input_scaled)
        pred_num = int(np.argmax(prob_num, axis=1)[0])
        pred_feedback = int(np.argmax(prob_feedback, axis=1)[0])
        conf_num = float(np.max(prob_num, axis=1)[0])
        conf_feedback = float(np.max(prob_feedback, axis=1)[0])

        feedback_text = feedback_map.get(pred_feedback, "Sin feedback disponible")

        return {
            "modelo": "feedforward_multioutput",
            "prediccion_numero": pred_num,
            "confianza_numero": round(conf_num * 100, 2),
            "prediccion_feedback": pred_feedback,
            "confianza_feedback": round(conf_feedback * 100, 2),
            "feedback_text": feedback_text
        }

    except Exception as e:
        return {
            "error": str(e),
            "modelo": "feedforward_multioutput"
        }
