import os
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

# Construct the path to the .env file in the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)


class LLMClient(ABC):
    """Abstracción de un cliente LLM. Contrato: query() devuelve "" ante fallo definitivo."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identificador proveedor+modelo, ej. "groq/llama-3.3-70b-versatile"."""

    @abstractmethod
    def query(self, prompt: str, retries: int = 5) -> str:
        pass


class GeminiClient(LLMClient):
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        import google.generativeai as genai

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    @property
    def model_id(self) -> str:
        return f"gemini/{self.model_name}"

    def query(self, prompt: str, retries: int = 5) -> str:
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg:
                    match = re.search(r'seconds:\s*(\d+)', error_msg)
                    wait_seconds = int(match.group(1)) + 3 if match else 65
                    print(f"  [Rate limit] Esperando {wait_seconds}s... (intento {attempt + 1}/{retries})")
                    time.sleep(wait_seconds)
                else:
                    print(f"  [Error] {e}")
                    return ""
        print("  [Error] Se agotaron los reintentos.")
        return ""


def _provider_label(base_url: str) -> str:
    host = urlparse(base_url).hostname or ""
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 2:
        return parts[-2]
    if parts:
        return parts[0]
    return "openai_compatible"


class OpenAICompatibleClient(LLMClient):
    """Cliente genérico para APIs compatibles con OpenAI (Groq, OpenRouter, DeepSeek, Mistral)."""

    def __init__(self, base_url: str, api_key: str, model_name: str, temperature: float = 0.0):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "El paquete 'openai' es requerido para OpenAICompatibleClient. "
                "Instálalo con: pip install openai"
            ) from exc
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @property
    def model_id(self) -> str:
        return f"{_provider_label(self.base_url)}/{self.model_name}"

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        status = getattr(error, "status_code", None)
        if status is not None:
            return status == 429 or status >= 500
        error_msg = str(error)
        return any(code in error_msg for code in ("429", "500", "502", "503", "504"))

    def query(self, prompt: str, retries: int = 5) -> str:
        wait_seconds = 2.0
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if self._is_retryable(e) and attempt < retries - 1:
                    print(
                        f"  [Rate limit/Servidor] Esperando {wait_seconds:.0f}s... "
                        f"(intento {attempt + 1}/{retries})"
                    )
                    time.sleep(wait_seconds)
                    wait_seconds *= 2
                else:
                    print(f"  [Error] {e}")
                    return ""
        print("  [Error] Se agotaron los reintentos.")
        return ""


def build_client(role: str) -> LLMClient:
    """Construye el cliente LLM de un rol ("attacker", "solver", "judge") desde variables de entorno."""
    prefix = role.upper()
    provider = os.getenv(f"{prefix}_PROVIDER", "gemini").strip().lower()
    model_name = os.getenv(f"{prefix}_MODEL")
    api_key = os.getenv(f"{prefix}_API_KEY")

    if provider == "gemini":
        return GeminiClient(
            model_name=model_name or "gemini-2.5-flash",
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
        )

    if provider == "openai_compatible":
        base_url = os.getenv(f"{prefix}_BASE_URL")
        missing = [
            var
            for var, value in (
                (f"{prefix}_BASE_URL", base_url),
                (f"{prefix}_MODEL", model_name),
                (f"{prefix}_API_KEY", api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Faltan variables de entorno para el rol '{role}' "
                f"(proveedor openai_compatible): {', '.join(missing)}"
            )
        return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model_name=model_name)

    raise ValueError(f"Proveedor no soportado en {prefix}_PROVIDER: '{provider}'")


_MODEL_FAMILY_KEYWORDS = (
    "gemini",
    "gemma",
    "llama",
    "deepseek",
    "mistral",
    "mixtral",
    "qwen",
    "gpt",
    "claude",
    "phi",
    "command",
)


def _shared_family(model_id_a: str, model_id_b: str) -> Optional[str]:
    a, b = model_id_a.lower(), model_id_b.lower()
    if a == b:
        return a
    for keyword in _MODEL_FAMILY_KEYWORDS:
        if keyword in a and keyword in b:
            return keyword
    return None


def detect_shared_families(clients: Dict[str, LLMClient]) -> List[str]:
    """Devuelve mensajes de advertencia por cada par de roles que comparten familia de modelo."""
    messages = []
    roles = list(clients.items())
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            role_a, client_a = roles[i]
            role_b, client_b = roles[j]
            family = _shared_family(client_a.model_id, client_b.model_id)
            if family:
                messages.append(
                    f"Los roles '{role_a}' ({client_a.model_id}) y '{role_b}' ({client_b.model_id}) "
                    f"comparten la misma familia de modelo ('{family}'). "
                    "Esto reintroduce el riesgo de circularidad."
                )
    return messages


def warn_on_shared_families(clients: Dict[str, LLMClient]) -> None:
    for message in detect_shared_families(clients):
        print(f"[ADVERTENCIA] {message}")
