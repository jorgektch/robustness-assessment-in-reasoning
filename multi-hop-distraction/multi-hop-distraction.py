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
    print("ERROR: No se encontró la API Key. Revisa tu archivo .env")
    print("Para obtener tu API Key:")
    print("1. Ve a: https://aistudio.google.com/app/apikey")
    print("2. Crea una API Key")
    print("3. Agrégala al archivo .env")
    exit()

client = genai.Client(api_key=api_key)


class MultiHopDistraction(Attack):
    """Ataque que añade pasos de razonamiento válidos pero irrelevantes al contexto."""
    
    def __init__(self, intensity="medium"):
        self.intensity = intensity
        self.num_distractions = self._get_num_distractions(intensity)
    
    def _get_num_distractions(self, intensity):
        if intensity == "low":
            return random.randint(1, 2)
        elif intensity == "medium":
            return random.randint(2, 3)
        else:
            return random.randint(3, 4)
    
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

INTENSIDAD: {self.intensity}
- low: Sutiles, relativamente obvias que son irrelevantes
- medium: Moderadas, requieren atención para descartarlas  
- high: Complejas con datos específicos, muy difíciles de identificar como irrelevantes

FORMATO: JSON array sin markdown ni explicaciones.
Ejemplo: ["oración 1", "oración 2", "oración 3"]

Genera {self.num_distractions} oraciones:
"""

        try:
            respuesta_ia = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            texto_limpio = respuesta_ia.text.strip()
            
            if "```" in texto_limpio:
                texto_limpio = texto_limpio.split("```")[1]
                if texto_limpio.startswith(("json", "python")):
                    texto_limpio = texto_limpio.split("\n", 1)[1]
                texto_limpio = texto_limpio.strip()
            
            distracciones = json.loads(texto_limpio)
            
            if len(distracciones) != self.num_distractions:
                print(f"  [WARN] Ejemplo {new_example['id']}: Se esperaban {self.num_distractions} distracciones pero se generaron {len(distracciones)}")
            
            nuevo_contexto = self._insert_distractions(contexto, distracciones)
            new_example["context"] = nuevo_contexto
            
            new_example["metadata"] = {
                "attack": "multi_hop_distraction",
                "intensity": self.intensity,
                "num_distractions": len(distracciones),
                "distractions_added": distracciones
            }
            
            if not self._validate_example(example, new_example):
                print(f"  [WARN] Ejemplo {new_example['id']}: Falló la validación")
                new_example["metadata"]["validation_warning"] = True
        
        except json.JSONDecodeError as e:
            print(f"  [ERROR] Ejemplo {new_example['id']}: Error de parsing JSON - {e}")
            print(f"  Respuesta recibida: {texto_limpio[:200]}...")
            new_example["metadata"] = {
                "attack": "multi_hop_distraction",
                "intensity": self.intensity,
                "error": "json_decode_error",
                "error_details": str(e)
            }
        except Exception as e:
            print(f"  [ERROR] Ejemplo {new_example['id']}: Error de API - {e}")
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
            print(f"  [ERROR] Error en validación: {e}")
            return False


def test_attack():
    print("=" * 70)
    print("PRUEBA DEL ATAQUE MULTI-HOP DISTRACTION")
    print("=" * 70)
    
    test_example = {
        "id": "test_001",
        "task": "mcq",
        "context": "Pedro estudió 5 horas para el examen de matemáticas. Obtuvo una calificación de 95 puntos.",
        "question": "¿Cuántas horas estudió Pedro?",
        "options": ["A: 3 horas", "B: 5 horas", "C: 7 horas", "D: No se menciona"],
        "label": "B"
    }
    
    print("\nEJEMPLO ORIGINAL:")
    print(f"Context: {test_example['context']}")
    print(f"Question: {test_example['question']}")
    print(f"Label: {test_example['label']}")
    
    for intensity in ["low", "medium", "high"]:
        print(f"\n{'-' * 70}")
        print(f"PROBANDO INTENSIDAD: {intensity}")
        print(f"{'-' * 70}")
        
        attack = MultiHopDistraction(intensity=intensity)
        attacked_example = attack.apply(test_example)
        
        print(f"\nContext atacado:")
        print(attacked_example['context'])
        print(f"\nMetadata:")
        print(json.dumps(attacked_example['metadata'], indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("PRUEBA COMPLETADA")
    print("=" * 70)
