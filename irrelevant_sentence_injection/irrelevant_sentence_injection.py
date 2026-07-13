import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.base_attack import BaseAttack
from core.models import VerbalTask, IntensityLevel
from core.llm_client import LLMClient
from core.utils import extract_sentence_list


class IrrelevantSentenceInjection(BaseAttack):
    """
    Irrelevant Sentence Injection:
    Añade al contexto oraciones temáticamente afines pero causalmente
    irrelevantes para responder la pregunta. El contexto original queda
    íntegro; las oraciones se concatenan programáticamente al final.
    """

    SENTENCE_COUNTS = {
        IntensityLevel.LOW: 1,
        IntensityLevel.MEDIUM: 2,
        IntensityLevel.HIGH: 3,
    }

    def __init__(self, api: LLMClient):
        self.api = api

    def apply(self, task: VerbalTask, intensity: IntensityLevel) -> VerbalTask:
        num_sentences = self.SENTENCE_COUNTS.get(intensity)
        if not num_sentences:
            print(
                f"No sentence count defined for intensity level {intensity.name}. Returning original task."
            )
            return task

        prompt = f"""
        Actúa como un experto en diseño de pruebas adversariales para modelos de lenguaje.

        CONTEXTO: {task.context}
        PREGUNTA: {task.question}
        OPCIONES: {task.options}
        RESPUESTA CORRECTA (no la reveles, no la refuerces y no la contradigas): {task.label}

        OBJETIVO:
        Escribe exactamente {num_sentences} oración(es) NUEVA(s) para añadir al final del contexto.
        Cada oración debe cumplir:
        1. Temáticamente afín al contexto (mismo tema, personajes o escenario).
        2. Factualmente inocua: no aporta información necesaria para responder la pregunta.
        3. Causalmente irrelevante: no ayuda a responder, no contradice el contexto
           y no vuelve ambigua la respuesta correcta.
        4. Autocontenida, en español y con puntuación final.

        FORMATO DE SALIDA: un JSON array con exactamente {num_sentences} string(s),
        sin markdown, sin explicaciones ni texto adicional.
        Ejemplo: ["Oración uno.", "Oración dos."]
        """

        attacked = task.model_copy(deep=True)
        attacked.metadata["attack"] = "irrelevant_sentence_injection"
        attacked.metadata["intensity"] = intensity.value

        raw = self.api.query(prompt) or ""
        sentences = extract_sentence_list(raw)[:num_sentences]
        attacked.metadata["injected_sentences"] = sentences

        if len(sentences) < num_sentences:
            print(
                f"  [WARN] Tarea {task.id}: se esperaban {num_sentences} oraciones "
                f"y se obtuvieron {len(sentences)}."
            )
            attacked.metadata["error"] = "insufficient_sentences"
            return attacked

        attacked.context = " ".join([task.context] + sentences)
        return attacked
