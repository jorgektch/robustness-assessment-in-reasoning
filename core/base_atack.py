from abc import ABC, abstractmethod
from .models import VerbalTask, IntensityLevel


class BaseAttack(ABC):
    @abstractmethod
    def apply(self, task: VerbalTask, intensity: IntensityLevel) -> VerbalTask:
        """
        Applies an attack to a verbal task.

        Args:
            task: The verbal task to attack.
            intensity: The intensity of the attack.

        Returns:
            The attacked verbal task.
        """
        pass
