import os
from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
from routers import feedforward
from routers import lstm

app = FastAPI(title="Predicción MNIST", version="1.0.0")

# Variable global para el modelo (se carga de manera lazy)
model = None

def load_model_lazy():
    """Carga el modelo solo cuando se necesita"""
    global model
    if model is None:
        try:
            from tensorflow.keras.models import load_model
            print("Cargando modelo...")
            model = load_model("model/mnist_cnn_model.h5")
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
        # Cargar modelo solo cuando se necesite
        current_model = load_model_lazy()
        
        # Importar aquí para evitar problemas de inicio
        from utils.prepocess import preprocess_image
        
        # Leer imagen y preprocesarla
        image = await file.read()
        processed_image = preprocess_image(image)  # -> debe retornar shape (1, 28, 28, 1)

        # Hacer predicción
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

# Incluir routers
app.include_router(feedforward.router)
app.include_router(lstm.router)

# Para ejecución local
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )