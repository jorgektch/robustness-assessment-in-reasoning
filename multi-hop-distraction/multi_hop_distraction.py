import json
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.base_attack import BaseAttack
from core.models import VerbalTask, IntensityLevel
from core.services import GeminiAPI


class MultiHopDistraction(BaseAttack):
    DISTRACTION_COUNTS = {
        IntensityLevel.LOW: 1,
        IntensityLevel.MEDIUM: 2,
        IntensityLevel.HIGH: 4,
    }

    def __init__(self, api: GeminiAPI):
        self.api = api

    def apply(self, task: VerbalTask, intensity: IntensityLevel) -> VerbalTask:
        num_distractions = self.DISTRACTION_COUNTS.get(intensity, 2)

        prompt = f"""Genera {num_distractions} oraciones que sean PASOS DE RAZONAMIENTO VÁLIDOS
pero IRRELEVANTES para responder la pregunta.

CONTEXTO ORIGINAL:
{task.context}

PREGUNTA:
{task.question}

RESPUESTA CORRECTA:
{task.label}

REQUISITOS:
1. Relacionadas temáticamente con el contexto
2. Factualmente coherentes
3. No ayudan a responder la pregunta
4. No contradicen el contexto ni cambian la respuesta correcta

FORMATO: JSON array sin markdown ni explicaciones.
Ejemplo: ["oración 1", "oración 2"]

Genera exactamente {num_distractions} oraciones:
"""

        attacked = task.model_copy(deep=True)

        try:
            response = self.api.query(prompt).strip()
            if "```" in response:
                response = response.split("```")[1]
                if response.startswith(("json", "python")):
                    response = response.split("\n", 1)[1]
                response = response.strip()

            distractions = json.loads(response)
            attacked.context = self._insert_distractions(task.context, distractions)
            attacked.metadata["attack"] = "multi_hop_distraction"
            attacked.metadata["intensity"] = intensity.value
            attacked.metadata["num_distractions"] = len(distractions)
            attacked.metadata["distractions_added"] = distractions

            if not self._validate_task(task, attacked):
                attacked.metadata["validation_warning"] = True

        except json.JSONDecodeError as e:
            attacked.metadata["attack"] = "multi_hop_distraction"
            attacked.metadata["intensity"] = intensity.value
            attacked.metadata["error"] = "json_decode_error"
            attacked.metadata["error_details"] = str(e)

        return attacked

    def _insert_distractions(self, context: str, distractions: list) -> str:
        sentences = context.split(". ")
        if not sentences[-1].endswith("."):
            sentences[-1] += "."

        max_pos = len(sentences)
        num_to_insert = min(len(distractions), max(1, max_pos - 1))
        distractions = distractions[:num_to_insert]

        positions = random.sample(range(1, max_pos), num_to_insert)
        positions.sort()

        for i, distraction in enumerate(reversed(distractions)):
            pos = positions[len(positions) - 1 - i]
            if not distraction.endswith("."):
                distraction += "."
            sentences.insert(pos, distraction)

        return " ".join(sentences)

    def _validate_task(self, original: VerbalTask, attacked: VerbalTask) -> bool:
        return (
            original.label == attacked.label
            and original.question == attacked.question
            and original.options == attacked.options
            and len(attacked.context) > len(original.context)
        )