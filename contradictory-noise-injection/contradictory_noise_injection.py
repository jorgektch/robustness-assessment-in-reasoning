import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.base_attack import BaseAttack
from core.models import VerbalTask, IntensityLevel
from core.llm_client import LLMClient


class ContradictoryNoiseInjection(BaseAttack):
    PROMPTS = {
        IntensityLevel.LOW: """
        Añade UNA oración muy breve al final del contexto que insinúe una duda mínima,
        sin contradecir los hechos principales. Tono neutro y sutil.
        """,
        IntensityLevel.MEDIUM: """
        Añade UNA oración corta al final del contexto que introduzca una contradicción parcial
        o "ruido" contradictorio AMBIGUO o basado en RUMORES
        (ej: "Sin embargo, algunos registros sugieren que...", "Aunque se rumoreaba que...").
        NO debe invalidar la respuesta correcta.
        """,
        IntensityLevel.HIGH: """
        Añade UNA oración al final del contexto que genere mayor incertidumbre mediante
        referencias a fuentes conflictivas o interpretaciones alternativas,
        sin contradecir directamente los hechos probados del texto original.
        """,
    }

    def __init__(self, api: LLMClient):
        self.api = api

    def apply(self, task: VerbalTask, intensity: IntensityLevel) -> VerbalTask:
        instruction = self.PROMPTS.get(intensity)
        if not instruction:
            return task

        prompt = f"""
        Actúa como un experto en diseño de pruebas adversariales para modelos de lenguaje.

        CONTEXTO: {task.context}
        PREGUNTA: {task.question}
        OPCIONES: {task.options}
        RESPUESTA CORRECTA: {task.label}

        {instruction}

        REGLAS MANDATORIAS:
        1. Los hechos originales deben seguir siendo la fuente de verdad principal.
        2. Idioma: Español.
        3. Devuelve ÚNICAMENTE la oración generada, sin explicaciones ni markdown.
        """

        noise = self.api.query(prompt).strip().replace('"', '').replace("`", "")
        if not noise:
            return task

        attacked = task.model_copy(deep=True)
        attacked.context = f"{task.context} {noise}"
        attacked.metadata["attack"] = "contradictory_noise_injection"
        attacked.metadata["intensity"] = intensity.value
        attacked.metadata["injected_noise"] = noise
        return attacked
