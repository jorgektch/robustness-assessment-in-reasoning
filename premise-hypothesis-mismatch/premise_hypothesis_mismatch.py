import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.base_attack import BaseAttack
from core.models import IntensityLevel, VerbalTask
from core.llm_client import LLMClient
from core.utils import extract_single_sentence


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

    def __init__(self, api: LLMClient):
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
            premise = extract_single_sentence(raw)

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
