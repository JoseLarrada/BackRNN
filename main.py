import os
from fastapi import FastAPI, UploadFile, File, Form
import numpy as np

app = FastAPI(title="Predicción MNIST", version="1.0.0")

# Variable global para el modelo CNN
model = None

def load_model_lazy():
    """Carga el modelo CNN solo cuando se necesita"""
    global model
    if model is None:
        try:
            from tensorflow.keras.models import load_model
            print("Cargando modelo mnist_cnn_model.h5...")
            model = load_model("models/mnist_cnn_model.h5")
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            raise e
    return model

@app.get("/")
def root():
    return {"message": "Servidor FastAPI para predicción de números escritos a mano"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict_number(
    file: UploadFile = File(...),
    esperado: int = Form(...)
):
    try:
        current_model = load_model_lazy()
        from utils.prepocess import preprocess_image  # import interno para evitar fallos al iniciar

        image = await file.read()
        processed_image = preprocess_image(image)  # shape (1, 28, 28, 1)

        prediction = current_model.predict(processed_image)
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

# Routers: feedforward y lstm se importan solo si no dan error
try:
    from routers import feedforward
    app.include_router(feedforward.router)
except Exception as e:
    print(f"Feedforward no cargado: {e}")

try:
    from routers import lstm
    app.include_router(lstm.router)
except Exception as e:
    print(f"LSTM no cargado: {e}")

# Para ejecución local o en Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Iniciando servidor en el puerto {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
