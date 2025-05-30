import os
import json

# Ruta base donde están las carpetas del 0 al 9
BASE_DIR = "DatasetTrazos"
OUTPUT_DIR = "dataset_agrupado"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Recorre los números del 0 al 9
for number in range(10):
    folder_path = os.path.join(BASE_DIR, str(number))
    merged_data = []

    # Verifica si la carpeta existe
    if not os.path.isdir(folder_path):
        print(f"[!] Carpeta no encontrada: {folder_path}")
        continue

    # Recorre cada archivo JSON dentro de la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                try:
                    data = json.load(file)
                    merged_data.append(data)
                except json.JSONDecodeError:
                    print(f"[!] Error leyendo {filepath}, archivo omitido.")

    # Guarda todos los datos en un solo archivo por número
    output_path = os.path.join(OUTPUT_DIR, f"{number}.json")
    with open(output_path, 'w') as outfile:
        json.dump(merged_data, outfile, indent=2)

    print(f"[✓] Guardado: {output_path} ({len(merged_data)} muestras)")
