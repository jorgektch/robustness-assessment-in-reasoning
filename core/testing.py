from typing import Callable, List, Optional, Union

from .llm_client import LLMClient

MockResponse = Union[str, Callable[[str], str]]


class MockClient(LLMClient):
    """Cliente de prueba que devuelve respuestas predefinidas sin tocar red.

    - `responses`: lista de respuestas consumidas en orden (FIFO). Cada elemento
      puede ser un string o un callable(prompt) -> str.
    - `default_response`: respuesta cuando la lista se agota (o no se proporciona).
    - Registra cada prompt recibido en `self.calls`.
    """

    def __init__(
        self,
        responses: Optional[List[MockResponse]] = None,
        default_response: str = "",
        model_name: str = "mock-model",
    ):
        self.responses = list(responses) if responses else []
        self.default_response = default_response
        self._model_name = model_name
        self.calls: List[str] = []

    @property
    def model_id(self) -> str:
        return f"mock/{self._model_name}"

    def query(self, prompt: str, retries: int = 5) -> str:
        self.calls.append(prompt)
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = self.default_response
        if callable(response):
            return response(prompt)
        return response
