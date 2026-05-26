import random
import re
import sys
import os
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.base_attack import BaseAttack
from core.models import VerbalTask, IntensityLevel
from core.services import GeminiAPI


class SentenceShuffling(BaseAttack):

    def __init__(self, api: Optional[GeminiAPI] = None):
        self.api = api

    def apply(self, task: VerbalTask, intensity: IntensityLevel) -> VerbalTask:
        attacked = task.model_copy(deep=True)
        sentences = re.split(r"(?<=[.!?])\s+", task.context.strip())

        if len(sentences) <= 1:
            return attacked

        if intensity == IntensityLevel.LOW:
            i = random.randint(0, len(sentences) - 2)
            sentences[i], sentences[i + 1] = sentences[i + 1], sentences[i]
        elif intensity == IntensityLevel.MEDIUM:
            shuffled = sentences.copy()
            while shuffled == sentences:
                random.shuffle(shuffled)
            sentences = shuffled
        else:
            shuffled = sentences.copy()
            while shuffled == sentences:
                random.shuffle(shuffled)
            mid = len(shuffled) // 2
            shuffled[mid:] = reversed(shuffled[mid:])
            sentences = shuffled

        attacked.context = " ".join(sentences)
        attacks = list(attacked.metadata.get("attacks", []))
        attacks.append("sentence_shuffling")
        attacked.metadata["attacks"] = attacks
        attacked.metadata["intensity"] = intensity.value
        return attacked
