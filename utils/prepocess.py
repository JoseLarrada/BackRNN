from PIL import Image, ImageOps
import numpy as np
import io

def preprocess_image(image_bytes):
    # Abrir la imagen desde los bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Escala de grises
    
    # Redimensionar a 28x28 píxeles
    image = image.resize((28, 28))

    # Invertir colores (MNIST espera fondo negro, dígitos blancos)
    image = ImageOps.invert(image)

    # Convertir a array de numpy y normalizar
    image_array = np.array(image) / 255.0

    # Asegurar forma (1, 28, 28, 1) para CNN
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array