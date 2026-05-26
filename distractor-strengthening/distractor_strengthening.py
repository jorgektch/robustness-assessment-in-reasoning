import json
import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.base_attack import BaseAttack
from core.models import VerbalTask, IntensityLevel
from core.services import GeminiAPI


class DistractorStrengthening(BaseAttack):
    PROMPTS = {
        IntensityLevel.LOW: """
        Reescribe ligeramente las opciones incorrectas para que sean un poco más plausibles,
        manteniéndolas claramente incorrectas.
        """,
        IntensityLevel.MEDIUM: """
        Reescribe las opciones incorrectas para que sean semánticamente cercanas al contexto
        (más plausibles), pero que sigan siendo incorrectas.
        """,
        IntensityLevel.HIGH: """
        Reescribe las opciones incorrectas para que sean muy convincentes y difíciles de descartar,
        usando terminología del contexto, pero sin volverlas correctas.
        """,
    }

    def __init__(self, api: GeminiAPI):
        self.api = api

    def apply(self, task: VerbalTask, intensity: IntensityLevel) -> VerbalTask:
        instruction = self.PROMPTS.get(intensity)
        if not instruction:
            return task

        prompt = f"""
        Actúa como un creador de exámenes engañosos.
        Contexto: {task.context}
        Pregunta: {task.question}
        Opciones actuales: {task.options}
        Respuesta correcta: {task.label}

        {instruction}
        No cambies la opción correcta.

        IMPORTANTE: Devuélveme SOLO un JSON array válido.
        Sin markdown, sin comillas triples, sin explicaciones.
        Ejemplo: ["A: texto", "B: texto", "C: texto", "D: texto"]
        """

        attacked = task.model_copy(deep=True)
        attacks = list(attacked.metadata.get("attacks", []))
        attacks.append("distractor_strengthening")

        try:
            response = self.api.query(prompt).strip()
            if "```" in response:
                response = response.split("```")[1]
                if response.startswith(("json", "python")):
                    response = response.split("\n", 1)[1]

            match = re.search(r"\[.*?\]", response, re.DOTALL)
            if match:
                response = match.group()

            new_options = json.loads(response)
            if len(new_options) != len(task.options):
                raise ValueError(
                    f"El modelo devolvió {len(new_options)} opciones; "
                    f"se esperaban {len(task.options)}"
                )

            attacked.options = new_options
            attacked.metadata["attacks"] = attacks
            attacked.metadata["intensity"] = intensity.value

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [WARN] Tarea {task.id}: error de parsing — {e}")
            attacked.metadata["attacks"] = attacks
            attacked.metadata["intensity"] = intensity.value
            attacked.metadata["error"] = str(e)

        return attacked
