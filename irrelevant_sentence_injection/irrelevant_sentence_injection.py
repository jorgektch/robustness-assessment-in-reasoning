import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.base_atack import BaseAttack
from core.models import VerbalTask, IntensityLevel
from core.services import GeminiAPI


class IrrelevantSentenceInjection(BaseAttack):
    PROMPTS = {
        IntensityLevel.LOW: """
        Inject a single, semantically parasitic sentence into the following text.
        This sentence must latch onto the specific terminology and logical structure of the preceding sentence, appearing as a direct continuation or clarification.
        However, it must introduce a subtle, unverifiable presupposition or a shift in definition that is logically void but stylistically seamless.
        The goal is to create a "micro-fallacy" that is nearly impossible to spot without deep, critical analysis of the sentence's logical relationship to its neighbors.
        Return ONLY the new, modified text.

        Original Text:
        "{context}"
        """,
        IntensityLevel.MEDIUM: """
        Inject a short paragraph (two to three sentences) that introduces an abstract, philosophical, or meta-level consideration that appears profound but is a sophisticated red herring.
        This injection should use academic or technical jargon appropriate to the text's domain, forcing the reader to grapple with a complex, orthogonal concept.
        The goal is to trigger cognitive tunneling, where the reader invests significant effort in deciphering the red herring's meaning and relevance, thereby neglecting the primary task's core components.
        Return ONLY the new, modified text.

        Original Text:
        "{context}"
        """,
        IntensityLevel.HIGH: """
        Inject a series of sentences (three to five) that construct a "false framework" or a simulacrum of a solution.
        This injection must mimic the structure of a logical proof, a scientific explanation, or a methodical problem-solving approach, but be built upon a foundation of subtle misinterpretations of the original text's premises or a flawed axiom.
        It should present a coherent, compelling, but ultimately invalid path to a conclusion, designed to capture the reasoning process and lead it astray entirely. The deception must be structural, not just topical.
        Return ONLY the new, modified text.

        Original Text:
        "{context}"
        """,
    }

    def __init__(self, api: GeminiAPI):
        self.api = api

    def apply(self, task: VerbalTask, intensity: IntensityLevel) -> VerbalTask:
        prompt_template = self.PROMPTS.get(intensity)

        if not prompt_template:
            print(
                f"No prompt defined for intensity level {intensity.name}. Returning original task."
            )
            return task

        prompt = prompt_template.format(context=task.context)
        new_context = self.api.query(prompt)

        if new_context:
            attacked_task = task.copy(deep=True)
            attacked_task.context = new_context
            attacked_task.metadata["attack"] = "IrrelevantSentenceInjection"
            attacked_task.metadata["intensity"] = intensity.name
            return attacked_task
        else:
            print("Failed to generate new context. Returning original task.")
            return task
