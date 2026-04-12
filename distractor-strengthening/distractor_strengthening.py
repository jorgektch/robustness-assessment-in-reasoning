import os
from google import genai
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: No se encontró la API Key. Revisa tu archivo .env")
    exit()

client = genai.Client(api_key=api_key)

class DistractorStrengthening:
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
        
        Reescribe las opciones incorrectas para que sean semánticamente cercanas al contexto (más plausibles), pero que sigan siendo incorrectas. No cambies la opción correcta.
        Devuélveme SOLO una lista de Python válida (ejemplo: ["A: falsa1", "B: correcta", "C: falsa2"]) con las nuevas opciones. No agregues comillas triples de markdown, ni explicaciones extra.
        """
        
        try:
            respuesta_ia = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            texto_limpio = respuesta_ia.text.strip().strip('```python').strip('```json').strip('```').strip()
            new_example["options"] = eval(texto_limpio) 
            new_example["metadata"] = {
                "attack": "distractor_strengthening",
                "intensity": "medium"
            }
        except Exception as e:
            print(f"Error procesando la pregunta {new_example['id']}: {e}")
            pass 
            
        return new_example