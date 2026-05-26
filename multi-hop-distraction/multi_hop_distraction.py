import os
import json
import random
from google import genai
from dotenv import load_dotenv
import sys
sys.path.append('..')
from base_attack import Attack

load_dotenv(dotenv_path="../.env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: No se encontró la API Key")
    exit()

client = genai.Client(api_key=api_key)


class MultiHopDistraction(Attack):
    def __init__(self):
        self.intensity = "medium"
        self.num_distractions = random.randint(2, 3)
    
    def apply(self, example):
        new_example = example.copy()
        contexto = new_example.get("context", "")
        pregunta = new_example.get("question", "")
        respuesta_correcta = new_example.get("label", "")
        
        prompt = f"""Genera {self.num_distractions} oraciones que sean PASOS DE RAZONAMIENTO VÁLIDOS pero IRRELEVANTES para responder la pregunta.

CONTEXTO ORIGINAL:
{contexto}

PREGUNTA:
{pregunta}

RESPUESTA CORRECTA:
{respuesta_correcta}

REQUISITOS:
1. Relacionadas TEMÁTICAMENTE con el contexto
2. FACTUALMENTE COHERENTES (información válida)
3. Parecen INFORMACIÓN ADICIONAL LEGÍTIMA
4. NO ayudan a responder la pregunta
5. Requieren RAZONAMIENTO MULTI-PASO para descartarlas
6. NO contradecir el contexto ni cambiar la respuesta correcta

INTENSIDAD: Moderada - requieren atención para descartarlas

FORMATO: JSON array sin markdown ni explicaciones.
Ejemplo: ["oración 1", "oración 2", "oración 3"]

Genera {self.num_distractions} oraciones:
"""

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
                texto_limpio = texto_limpio.strip()
            
            distracciones = json.loads(texto_limpio)
            
            nuevo_contexto = self._insert_distractions(contexto, distracciones)
            new_example["context"] = nuevo_contexto
            
            new_example["metadata"] = {
                "attack": "multi_hop_distraction",
                "intensity": self.intensity,
                "num_distractions": len(distracciones),
                "distractions_added": distracciones
            }
            
            if not self._validate_example(example, new_example):
                new_example["metadata"]["validation_warning"] = True
        
        except json.JSONDecodeError as e:
            new_example["metadata"] = {
                "attack": "multi_hop_distraction",
                "intensity": self.intensity,
                "error": "json_decode_error",
                "error_details": str(e)
            }
        except Exception as e:
            new_example["metadata"] = {
                "attack": "multi_hop_distraction",
                "intensity": self.intensity,
                "error": "api_error",
                "error_details": str(e)
            }
        
        return new_example
    
    def _insert_distractions(self, contexto, distracciones):
        oraciones = contexto.split('. ')
        
        if not oraciones[-1].endswith('.'):
            oraciones[-1] += '.'
        else:
            oraciones = [o if i == len(oraciones)-1 else o for i, o in enumerate(oraciones)]
        
        max_pos = len(oraciones)
        posiciones = random.sample(range(1, max_pos), min(len(distracciones), max_pos-1))
        posiciones.sort()
        
        for i, distraccion in enumerate(reversed(distracciones)):
            pos = posiciones[len(posiciones) - 1 - i]
            if not distraccion.endswith('.'):
                distraccion += '.'
            oraciones.insert(pos, distraccion)
        
        return ' '.join(oraciones)
    
    def _validate_example(self, original, attacked):
        try:
            if original.get("label") != attacked.get("label"):
                return False
            if original.get("question") != attacked.get("question"):
                return False
            if original.get("options") != attacked.get("options"):
                return False
            if len(attacked.get("context", "")) <= len(original.get("context", "")):
                return False
            return True
        except Exception as e:
            return False
