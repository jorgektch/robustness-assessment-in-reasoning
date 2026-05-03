import os
import re
import json
from google import genai
from dotenv import load_dotenv
from base_attack import Attack

load_dotenv(dotenv_path="../.env")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: No se encontró la API Key. Revisa tu archivo .env")
    exit()

client = genai.Client(api_key=api_key)

class DistractorStrengthening(Attack):
    def apply(self, example):
        new_example = example.copy()
        contexto = new_example.get("context", "")
        pregunta = new_example.get("question", "")
        opciones_originales = new_example.get("options", [])
        respuesta_correcta = new_example.get("label", "")

        prompt = f"""
        Actúa como un creador de exámenes engañosos.
        Contexto: {contexto}
        Pregunta: {pregunta}
        Opciones actuales: {opciones_originales}
        Respuesta correcta: {respuesta_correcta}

        Reescribe las opciones incorrectas para que sean semánticamente cercanas al contexto
        (más plausibles), pero que sigan siendo incorrectas. No cambies la opción correcta.
        
        IMPORTANTE: Devuélveme SOLO un JSON array válido.
        Sin markdown, sin comillas triples, sin explicaciones.
        Ejemplo exacto del formato esperado: ["A: texto", "B: texto", "C: texto", "D: texto"]
        """

        metadata_actual = new_example.get("metadata", {})
        ataques_previos = metadata_actual.get("attacks", [])
        ataques_previos.append("distractor_strengthening")

        try:
            respuesta_ia = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            texto_limpio = respuesta_ia.text.strip()
            
            if "```" in texto_limpio:
                texto_limpio = texto_limpio.split("```")[1]
                if texto_limpio.startswith(("json", "python")):
                    texto_limpio = texto_limpio.split("\n", 1)[1]

            match = re.search(r'\[.*?\]', texto_limpio, re.DOTALL)
            if match:
                texto_limpio = match.group()
            
            nuevas_opciones = json.loads(texto_limpio)
            
            if len(nuevas_opciones) != len(opciones_originales):
                raise ValueError(
                    f"El modelo devolvió {len(nuevas_opciones)} opciones "
                    f"pero se esperaban {len(opciones_originales)}"
                )

            new_example["options"] = nuevas_opciones
            new_example["metadata"] = {
                "attacks": ataques_previos,
                "intensity": "medium"
            }

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [WARN] Pregunta {new_example['id']}: error de parsing — {e}")
            new_example["metadata"] = {
                "attacks": ataques_previos,
                "intensity": "medium",
                "error": str(e)
            }

        except Exception as e:
            print(f"  [ERROR] Pregunta {new_example['id']}: error de API — {e}")
            new_example["metadata"] = {
                "attacks": ataques_previos,
                "intensity": "medium",
                "error": str(e)
            }

        return new_example