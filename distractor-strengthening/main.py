import json
import time
from datetime import datetime
from distractor_strengthening import DistractorStrengthening

def ejecutar_prueba_local():
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    
    ruta_entrada = "../dataset/dataset_prueba_120420261547.json" 
    ruta_salida = f"../dataset/dataset_atacado_{timestamp}.json" 
    
    print(f"Leyendo datos de: {ruta_entrada}...")
    try:
        with open(ruta_entrada, 'r', encoding='utf-8') as f:
            datos_originales = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo. Revisa la ruta.")
        return

    ataque = DistractorStrengthening()
    datos_atacados = []

    print("Iniciando ataque...")
    for i, pregunta in enumerate(datos_originales):
        print(f"  -> Atacando pregunta {pregunta['id']} ({i+1}/{len(datos_originales)})...")
        
        pregunta_modificada = ataque.apply(pregunta)
        datos_atacados.append(pregunta_modificada)
        
        if i < len(datos_originales) - 1:
            time.sleep(4)
            
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        json.dump(datos_atacados, f, indent=4, ensure_ascii=False)
        
    print(f"¡Éxito! Resultados guardados en: {ruta_salida}")

if __name__ == "__main__":
    ejecutar_prueba_local()