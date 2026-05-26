import re
import random
from base_attack import Attack


class SentenceShuffling(Attack):
    def apply(self, example):
        new_example = example.copy()
        contexto = new_example.get("context", "")

        oraciones = re.split(r'(?<=[.!?])\s+', contexto.strip())

        if len(oraciones) > 1:
            shuffled = oraciones.copy()
            while shuffled == oraciones:
                random.shuffle(shuffled)
            nuevo_contexto = " ".join(shuffled)
            new_example["context"] = nuevo_contexto

        metadata_actual = new_example.get("metadata", {})
        ataques_previos = metadata_actual.get("attacks", [])
        ataques_previos.append("sentence_shuffling")

        new_example["metadata"] = {
            "attacks": ataques_previos,
            "intensity": "medium"
        }

        return new_example