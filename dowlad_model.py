import os
import requests
import json

# URL del JSON con la lista de modelos
JSON_URL = "https://gpt4all.io/models/models.json"

# Crear carpeta para almacenar modelos si no existe
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def obtener_lista_modelos():
    """Descarga y retorna la lista de modelos desde la URL oficial."""
    try:
        response = requests.get(JSON_URL)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error al obtener la lista de modelos: {e}")
        return []

def mostrar_modelos_disponibles(modelos):
    """Muestra los modelos disponibles en la lista."""
    print("\nModelos disponibles en GPT4All:\n")
    for idx, modelo in enumerate(modelos):
        print(f"{idx + 1}. {modelo['name']} ({modelo['parameters']} parámetros)")

def seleccionar_modelo(modelos):
    """Pide al usuario seleccionar un modelo de la lista."""
    while True:
        try:
            opcion = int(input("\nIngresa el número del modelo que deseas descargar: "))
            if 1 <= opcion <= len(modelos):
                return modelos[opcion - 1]  # Retorna el modelo seleccionado
            else:
                print("Número fuera de rango. Inténtalo de nuevo.")
        except ValueError:
            print("Entrada inválida. Ingresa un número.")

def descargar_modelo(modelo):
    """Descarga el modelo seleccionado y lo guarda en la carpeta models/."""
    model_url = modelo.get("url")
    model_filename = modelo["filename"]
    model_path = os.path.join(MODELS_DIR, model_filename)

    if not model_url:
        print("Este modelo no tiene una URL de descarga.")
        return

    print(f"\nDescargando {modelo['name']} desde:\n{model_url}")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 8192  # Tamaño del bloque de descarga

        with open(model_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                downloaded += len(chunk)
                progress = (downloaded / total_size) * 100 if total_size else 0
                print(f"\rDescargando... {progress:.2f}%", end="")

        print(f"\nDescarga completada: {model_path}")

    except requests.RequestException as e:
        print(f"Error al descargar el modelo: {e}")

# Ejecutar el proceso
modelos = obtener_lista_modelos()
if modelos:
    mostrar_modelos_disponibles(modelos)
    modelo_seleccionado = seleccionar_modelo(modelos)
    descargar_modelo(modelo_seleccionado)
else:
    print("No se pudo obtener la lista de modelos.")
