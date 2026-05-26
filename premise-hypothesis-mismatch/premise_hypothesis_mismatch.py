import json
import os
import sys
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.base_attack import BaseAttack
from core.models import IntensityLevel, VerbalTask
from core.services import GeminiAPI


class PremiseHypothesisMismatch(BaseAttack):
    """
    Premise-Hypothesis Mismatch:
    Inyecta una oración que sugiere una hipótesis alternativa (incorrecta) sin
    contradecir explícitamente los hechos principales del contexto.
    """

    PROMPTS = {
        IntensityLevel.LOW: """
        Agrega UNA sola oración breve al final del contexto.
        Debe insinuar una interpretación alternativa que haría dudar al lector,
        pero debe ser sutil y ambigua. No contradigas directamente hechos del contexto.
        """,
        IntensityLevel.MEDIUM: """
        Agrega UNA sola oración al final del contexto.
        Debe sugerir una hipótesis alternativa PLAUSIBLE que empuje hacia una respuesta incorrecta,
        usando detalles compatibles con el contexto. No cambies datos explícitos del texto.
        """,
        IntensityLevel.HIGH: """
        Agrega UNA sola oración al final del contexto.
        Debe ser persuasiva y dirigir el razonamiento hacia una conclusión equivocada,
        introduciendo una premisa aparentemente relevante (ejemplo de estilo:
        "Ana estudió mucho pero estaba nerviosa. Antes del examen parecía tranquila.").
        Aun así, NO debe contradecir explícitamente el contexto ni invalidar hechos textuales.
        """,
    }

    def __init__(self, api: GeminiAPI):
        self.api = api

    def apply(self, task: VerbalTask, intensity: IntensityLevel) -> VerbalTask:
        instruction = self.PROMPTS.get(intensity)
        if not instruction:
            return task

        prompt = f"""
        Actúa como un experto en ataques adversariales para evaluación de robustez.

        CONTEXTO: {task.context}
        PREGUNTA: {task.question}
        OPCIONES: {task.options}
        RESPUESTA CORRECTA (no la reveles en tu salida): {task.label}

        OBJETIVO:
        Escribe UNA sola oración para AÑADIR al final del contexto que sugiera (implícitamente)
        una hipótesis alternativa que incline el razonamiento hacia una opción incorrecta.

        {instruction}

        REGLAS ESTRICTAS:
        1) Tu salida debe ser SOLO la oración inyectada (una sola oración).
        2) Sin markdown, sin comillas, sin JSON explicativo, sin listas.
        3) Idioma: Español.
        4) No modifiques la pregunta ni las opciones; solo produce la oración para el contexto.
        """

        attacked = task.model_copy(deep=True)

        try:
            raw = self.api.query(prompt) or ""
            premise = _extract_single_sentence(raw)

            if not premise:
                attacked.metadata["error"] = "empty_premise"
                attacked.metadata["attack"] = "premise_hypothesis_mismatch"
                attacked.metadata["intensity"] = intensity.value
                return attacked

            attacked.context = f"{task.context} {premise}"
            attacked.metadata["attack"] = "premise_hypothesis_mismatch"
            attacked.metadata["intensity"] = intensity.value
            attacked.metadata["injected_premise"] = premise
            return attacked

        except json.JSONDecodeError as e:
            attacked.metadata["attack"] = "premise_hypothesis_mismatch"
            attacked.metadata["intensity"] = intensity.value
            attacked.metadata["error"] = "json_decode_error"
            attacked.metadata["error_details"] = str(e)
            return attacked
        except Exception as e:
            attacked.metadata["attack"] = "premise_hypothesis_mismatch"
            attacked.metadata["intensity"] = intensity.value
            attacked.metadata["error"] = "attack_error"
            attacked.metadata["error_details"] = str(e)
            return attacked


def _extract_single_sentence(text: str) -> str:
    """
    Extrae de forma segura una sola oración.
    - Limpia markdown y comillas
    - Si el modelo devuelve JSON, intenta extraer un campo razonable
    - Devuelve un string sin saltos de línea, con puntuación final
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    # Remove common markdown fences
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Strip wrapping quotes/backticks
    cleaned = cleaned.strip().strip('"').strip("'").strip("`").strip()

    # If it looks like JSON, try to parse and extract a field
    if cleaned.startswith("{") or cleaned.startswith("["):
        parsed = json.loads(cleaned)
        extracted: Optional[str] = None

        if isinstance(parsed, dict):
            for key in ("premise", "injected_premise", "sentence", "text", "output"):
                val = parsed.get(key)
                if isinstance(val, str) and val.strip():
                    extracted = val.strip()
                    break
        elif isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
            extracted = parsed[0].strip()

        if extracted is None:
            return ""
        cleaned = extracted.strip().strip('"').strip("'").strip("`").strip()

    # Collapse whitespace/newlines
    cleaned = " ".join(cleaned.split())

    # Keep only first sentence-ish chunk if multiple were returned
    # (heuristic: split on sentence-ending punctuation followed by space)
    for sep in (". ", "? ", "! "):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0] + sep.strip()
            break

    # Ensure ends with sentence punctuation
    if cleaned and cleaned[-1] not in ".?!":
        cleaned += "."

    return cleaned
