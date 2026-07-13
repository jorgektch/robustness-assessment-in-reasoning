import json
import random
import re
from typing import List, Optional


def extract_single_sentence(text: str) -> str:
    """
    Extrae de forma segura una sola oración.
    - Limpia markdown y comillas
    - Si el modelo devuelve JSON, intenta extraer un campo razonable
    - Devuelve un string sin saltos de línea, con puntuación final
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    # Remove common markdown fences
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Strip wrapping quotes/backticks
    cleaned = cleaned.strip().strip('"').strip("'").strip("`").strip()

    # If it looks like JSON, try to parse and extract a field
    if cleaned.startswith("{") or cleaned.startswith("["):
        parsed = json.loads(cleaned)
        extracted: Optional[str] = None

        if isinstance(parsed, dict):
            for key in ("premise", "injected_premise", "sentence", "text", "output"):
                val = parsed.get(key)
                if isinstance(val, str) and val.strip():
                    extracted = val.strip()
                    break
        elif isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
            extracted = parsed[0].strip()

        if extracted is None:
            return ""
        cleaned = extracted.strip().strip('"').strip("'").strip("`").strip()

    # Collapse whitespace/newlines
    cleaned = " ".join(cleaned.split())

    # Keep only first sentence-ish chunk if multiple were returned
    # (heuristic: split on sentence-ending punctuation followed by space)
    for sep in (". ", "? ", "! "):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0] + sep.strip()
            break

    # Ensure ends with sentence punctuation
    if cleaned and cleaned[-1] not in ".?!":
        cleaned += "."

    return cleaned


def extract_sentence_list(text: str) -> List[str]:
    """
    Extrae una lista de oraciones de la salida de un modelo.
    Acepta JSON array (con o sin fences de markdown), JSON dict con una clave
    razonable, o texto plano con una oración por línea (viñetas/numeración).
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    candidates: List[str] = []

    # Try JSON array (possibly embedded in surrounding text)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            if isinstance(parsed, list):
                candidates = [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            candidates = []

    # Try JSON dict with a reasonable key
    if not candidates and cleaned.startswith("{"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                for key in ("sentences", "injected_sentences", "oraciones", "output"):
                    val = parsed.get(key)
                    if isinstance(val, list):
                        candidates = [str(item).strip() for item in val if str(item).strip()]
                        break
        except json.JSONDecodeError:
            candidates = []

    # Fallback: one sentence per non-empty line, stripping bullets/numbering
    if not candidates:
        for line in cleaned.splitlines():
            line = line.strip().lstrip("-*•").strip()
            line = re.sub(r"^\d+[.)]\s*", "", line)
            line = line.strip().strip('"').strip("'").strip("`").strip()
            if line:
                candidates.append(line)

    result: List[str] = []
    for sentence in candidates:
        sentence = " ".join(sentence.split()).strip().strip('"').strip("'").strip()
        if not sentence:
            continue
        if sentence[-1] not in ".?!":
            sentence += "."
        result.append(sentence)
    return result


def insert_distractions(context: str, distractions: List[str]) -> str:
    """
    Inserta oraciones en posiciones aleatorias del contexto (nunca al inicio),
    preservando todas las oraciones originales y su orden relativo.
    """
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
