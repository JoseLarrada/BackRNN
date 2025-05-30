from fastapi import FastAPI, UploadFile, File, Form
from tensorflow.keras.models import load_model
from utils.prepocess import preprocess_image
import numpy as np
from routers import feedforward
from routers import lstm

app = FastAPI()

# Cargar modelo al iniciar
model = load_model("model/mnist_cnn_model.h5")

@app.get("/")
def root():
    return {"message": "Servidor FastAPI para predicción de números escritos a mano"}

@app.post("/predict")
async def predict_number(
    file: UploadFile = File(...),
    esperado: int = Form(...)
):
    try:
        # Leer imagen y preprocesarla
        image = await file.read()
        processed_image = preprocess_image(image)  # -> debe retornar shape (1, 28, 28, 1)

        # Hacer predicción
        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction))
        confianza_prediccion = float(np.max(prediction))
        similitud_esperado = float(prediction[0][esperado])

        return {
            "prediccion": predicted_class,
            "confianza_prediccion": round(confianza_prediccion * 100, 2),
            "esperado": esperado,
            "similitud_con_esperado": round(similitud_esperado * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}

app.include_router(feedforward.router)

app.include_router(lstm.router)