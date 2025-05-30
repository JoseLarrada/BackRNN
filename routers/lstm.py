from fastapi import APIRouter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from schemas.stroke_input import StrokeInput

router = APIRouter()
model_lstm = load_model("model\data_rnn_model.h5")
MAX_SEQ_LEN = 100

def normalize(arr):
    min_val = min(arr)
    max_val = max(arr)
    return [(val - min_val) / (max_val - min_val) if max_val > min_val else 0.0 for val in arr]

def procesar_stroke(stroke):
    xs = [p.x for p in stroke]
    ys = [p.y for p in stroke]
    ts = [p.t for p in stroke]

    norm_xs = normalize(xs)
    norm_ys = normalize(ys)
    norm_ts = normalize(ts)

    secuencia = [[x, y, t] for x, y, t in zip(norm_xs, norm_ys, norm_ts)]
    return secuencia

@router.post("/predict/lstm")
def predict_lstm(data: StrokeInput):
    try:
        print(data)
        secuencia = procesar_stroke(data.stroke)
        secuencia_padded = pad_sequences([secuencia], maxlen=MAX_SEQ_LEN, dtype='float32', padding='post', truncating='post')

        prediccion_valor = float(model_lstm.predict(secuencia_padded)[0][0])  # convierte numpy.float32 a float nativo

        return {
            "modelo": "lstm",
            "valor_predicho": round(prediccion_valor, 2)
        }
    except Exception as e:
        return {"error": str(e)}
