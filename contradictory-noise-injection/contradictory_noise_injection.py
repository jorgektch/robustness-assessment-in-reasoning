import os
import sys
import json
from google import genai
from dotenv import load_dotenv

# Añadir el directorio raíz al path para poder importar base_attack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_attack import Attack

load_dotenv(dotenv_path="../.env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: No se encontró la API Key en contradictory_noise_injection.py. Revisa tu archivo .env")
    client = None
else:
    client = genai.Client(api_key=api_key)

class ContradictoryNoiseInjection(Attack):
    def __init__(self, model_name='gemini-2.5-flash'):
        self.model_name = model_name

    def apply(self, example):
        if not client:
            print(f"  [ERROR] Cliente Gemini no inicializado para la pregunta {example.get('id')}")
            return example

        new_example = example.copy()
        contexto = new_example.get("context", "")
        pregunta = new_example.get("question", "")
        respuesta_correcta = new_example.get("label", "")
        opciones = new_example.get("options", [])

        prompt = f"""
        Actúa como un experto en diseño de pruebas adversariales para modelos de lenguaje.
        
        CONTEXTO: {contexto}
        PREGUNTA: {pregunta}
        OPCIONES: {opciones}
        RESPUESTA CORRECTA: {respuesta_correcta}

        Tu tarea es generar una ÚNICA oración corta para añadir al final del contexto que introduzca una contradicción parcial o "ruido" contradictorio.
        
        REGLAS MANDATORIAS:
        1. La oración debe ser AMBIGUA o basada en RUMORES (ej: "Sin embargo, algunos registros sugieren que...", "Aunque se rumoreaba que...", "Existían dudas sobre si...").
        2. NO DEBE INVALIDAR la respuesta correcta. Los hechos originales deben seguir siendo la fuente de verdad principal.
        3. Debe generar incertidumbre sobre el razonamiento sin contradecir directamente los hechos probados en el texto.
        4. Idioma: Español.
        
        IMPORTANTE: Devuelve ÚNICAMENTE la oración generada. Sin explicaciones, sin comillas adicionales, sin formato markdown.
        """

        try:
            respuesta_ia = client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            ruido = respuesta_ia.text.strip()
            ruido = ruido.replace('"', '').replace('`', '')
            
            new_example["context"] = f"{contexto} {ruido}"
            new_example["metadata"] = {
                "attack": "contradictory_noise_injection",
                "difficulty": "medium",
                "injected_noise": ruido
            }

        except Exception as e:
            print(f"  [ERROR] Pregunta {new_example.get('id', 'N/A')}: error de API — {e}")
            new_example["metadata"] = {
                "attack": "contradictory_noise_injection",
                "error": str(e)
            }

        return new_example
