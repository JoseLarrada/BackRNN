import numpy as np

def extract_features(stroke: list[dict]) -> dict:
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
