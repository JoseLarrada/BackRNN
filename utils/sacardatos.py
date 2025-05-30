import json
import numpy as np
import os

def extract_features(stroke):
    xs = np.array([p["x"] for p in stroke])
    ys = np.array([p["y"] for p in stroke])
    ts = np.array([p["t"] for p in stroke])

    deltas = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    total_length = np.sum(deltas)
    total_time = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    direction_changes = np.sum(np.abs(np.diff(np.sign(np.diff(xs)))) + np.abs(np.diff(np.sign(np.diff(ys)))))

    angles = []
    for i in range(1, len(xs)-1):
        v1 = np.array([xs[i] - xs[i-1], ys[i] - ys[i-1]])
        v2 = np.array([xs[i+1] - xs[i], ys[i+1] - ys[i]])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 != 0 and norm2 != 0:
            cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            angle = np.arccos(cos_theta)
            angles.append(np.degrees(angle))
    avg_curvature = np.mean(angles) if angles else 0.0

    symmetry = np.mean(np.abs(xs - (1 - xs)))

    return {
        "longitud_total": round(total_length, 3),
        "tiempo_total": round(total_time, 3),
        "cambios_direccion": int(direction_changes),
        "curvatura_promedio": round(avg_curvature, 2),
        "simetria_horizontal": round(symmetry, 3)
    }

# Carpeta que contiene los archivos
carpeta = r'DatasetTrazos\9'
resultado_final = []

# Iterar sobre los archivos prueba001.json a prueba030.json
for i in range(1, 31):
    nombre_archivo = f'prueba{i:03d}.json'
    ruta_archivo = os.path.join(carpeta, nombre_archivo)

    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            datos = json.load(f)

        stroke = datos["stroke"]
        caracteristicas = extract_features(stroke)

        resultado_final.append({
            "label": datos["label"],
            "quality": datos.get("quality", None),  # usa None si falta
            "promedios": caracteristicas
        })

    except Exception as e:
        print(f"Error procesando {nombre_archivo}: {e}")

# Guardar todos los resultados en un solo archivo JSON
ruta_salida = os.path.join(r'DatasetTrazos\agrupados', r'resultado_total_9.json')
with open(ruta_salida, 'w', encoding='utf-8') as f:
    json.dump(resultado_final, f, ensure_ascii=False, indent=2)

print(f"Resultado combinado guardado en: {ruta_salida}")