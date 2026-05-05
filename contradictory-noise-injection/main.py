import json
import time
import os
from datetime import datetime
from contradictory_noise_injection import ContradictoryNoiseInjection

def ejecutar_prueba_local():
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    
    # Rutas relativas asumiendo que el script se ejecuta desde su propia carpeta
    ruta_entrada = "../dataset/dataset_prueba_120420261547.json" 
    ruta_salida = f"../dataset/dataset_atacado_noise_{timestamp}.json" 
    
    print(f"--- Iniciando Ataque: Contradictory Noise Injection ---")
    print(f"Leyendo datos de: {ruta_entrada}...")
    
    try:
        with open(ruta_entrada, 'r', encoding='utf-8') as f:
            datos_originales = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo en {ruta_entrada}. Asegúrate de estar en el directorio correcto.")
        return

    ataque = ContradictoryNoiseInjection()
    datos_atacados = []

    print(f"Procesando {len(datos_originales)} preguntas...")
    for i, pregunta in enumerate(datos_originales):
        print(f"  -> Atacando pregunta {pregunta['id']} ({i+1}/{len(datos_originales)})...")
        
        pregunta_modificada = ataque.apply(pregunta)
        datos_atacados.append(pregunta_modificada)
        
        # Pequeña pausa para no saturar la cuota de la API
        if i < len(datos_originales) - 1:
            time.sleep(2)
            
    try:
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(datos_atacados, f, indent=4, ensure_ascii=False)
        print(f"--- ¡Éxito! ---")
        print(f"Resultados guardados en: {ruta_salida}")
    except Exception as e:
        print(f"ERROR al guardar los resultados: {e}")

if __name__ == "__main__":
    ejecutar_prueba_local()
