import json
import re
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .models import IntensityLevel, VerbalTask

# Ataques que modifican el contexto añadiendo oraciones
CONTEXT_ATTACKS = {"isi", "cni", "mhd", "phm"}

# Oraciones añadidas esperadas por ataque e intensidad
EXPECTED_SENTENCE_COUNTS: Dict[str, Dict[IntensityLevel, int]] = {
    "isi": {IntensityLevel.LOW: 1, IntensityLevel.MEDIUM: 2, IntensityLevel.HIGH: 3},
    "cni": {IntensityLevel.LOW: 1, IntensityLevel.MEDIUM: 1, IntensityLevel.HIGH: 1},
    "phm": {IntensityLevel.LOW: 1, IntensityLevel.MEDIUM: 1, IntensityLevel.HIGH: 1},
    "mhd": {IntensityLevel.LOW: 1, IntensityLevel.MEDIUM: 2, IntensityLevel.HIGH: 4},
}

# Clave de metadata donde cada ataque registra lo que inyectó
METADATA_SENTENCE_KEYS = {
    "isi": "injected_sentences",
    "cni": "injected_noise",
    "phm": "injected_premise",
    "mhd": "distractions_added",
}

# Rubric por ataque para el juez LLM: qué debe y qué no debe hacer cada ataque
ATTACK_RUBRICS = {
    "isi": (
        "Irrelevant Sentence Injection: la(s) oración(es) añadida(s) deben ser "
        "temáticamente afines al contexto pero causalmente irrelevantes para responder "
        "la pregunta. NO deben aportar información necesaria para responder, ni "
        "contradecir el contexto, ni volver ambigua la respuesta correcta."
    ),
    "cni": (
        "Contradictory Noise Injection: la oración añadida introduce una duda ambigua "
        "o basada en rumores (ej. 'algunos registros sugieren...'). NO debe negar hechos "
        "explícitos del contexto ni invalidar la respuesta correcta."
    ),
    "mhd": (
        "Multi-Hop Distraction: las oraciones añadidas son pasos de razonamiento válidos "
        "y temáticamente relacionados, pero irrelevantes para responder la pregunta. "
        "NO deben contradecir el contexto ni cambiar la respuesta correcta."
    ),
    "phm": (
        "Premise-Hypothesis Mismatch: la oración añadida sugiere una hipótesis alternativa "
        "plausible que empuja hacia una opción incorrecta, SIN contradecir explícitamente "
        "los hechos del contexto ni invalidar la respuesta correcta."
    ),
    "ds": (
        "Distractor Strengthening: las opciones incorrectas fueron reescritas para ser más "
        "plausibles y cercanas al contexto, pero deben seguir siendo claramente incorrectas. "
        "La opción correcta debe permanecer intacta y seguir siendo la única respuesta correcta."
    ),
}


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def _intensity_from_metadata(attacked: VerbalTask) -> Optional[IntensityLevel]:
    raw = attacked.metadata.get("intensity")
    if raw is None:
        return None
    try:
        # Acepta "low" (nuevo) y "LOW" (datasets antiguos)
        return IntensityLevel(str(raw).lower())
    except ValueError:
        return None


def _find_option_index(options: List[str], label: str) -> Optional[int]:
    for i, option in enumerate(options):
        prefix = option.split(":", 1)[0].strip()
        if prefix == label or option.strip() == label:
            return i
    return None


def _registered_sentences(attacked: VerbalTask, attack_key: str) -> List[str]:
    value = attacked.metadata.get(METADATA_SENTENCE_KEYS[attack_key])
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _is_subsequence(needles: List[str], haystack: List[str]) -> bool:
    iterator = iter(haystack)
    return all(needle in iterator for needle in needles)


def structural_checks(original: VerbalTask, attacked: VerbalTask, attack_key: str) -> List[str]:
    """Checks deterministas (sin API). Devuelve lista de violaciones; vacía = OK."""
    violations: List[str] = []

    if attacked.id != original.id:
        violations.append("el id fue modificado")
    if attacked.label != original.label:
        violations.append("el label fue modificado")
    if attacked.question != original.question:
        violations.append("la question fue modificada")
    if "error" in (attacked.metadata or {}):
        violations.append(f"metadata contiene 'error': {attacked.metadata['error']}")

    intensity = _intensity_from_metadata(attacked)

    if attack_key in CONTEXT_ATTACKS:
        if attacked.options != original.options:
            violations.append("las options fueron modificadas en un ataque de contexto")

        original_sentences = _split_sentences(original.context)
        attacked_sentences = _split_sentences(attacked.context)

        if attack_key == "mhd":
            if not _is_subsequence(original_sentences, attacked_sentences):
                violations.append(
                    "el contexto original no está íntegro (oraciones originales faltantes "
                    "o fuera de su orden relativo)"
                )
        else:
            if original.context not in attacked.context:
                violations.append("el contexto original no es substring exacto del contexto atacado")

        if intensity is None:
            violations.append("intensidad ausente o desconocida en metadata")
        else:
            expected = EXPECTED_SENTENCE_COUNTS[attack_key][intensity]
            added = len(attacked_sentences) - len(original_sentences)
            if added != expected:
                violations.append(f"se añadieron {added} oraciones; se esperaban {expected}")

        registered = _registered_sentences(attacked, attack_key)
        if not registered:
            violations.append(
                f"no hay oraciones registradas en metadata['{METADATA_SENTENCE_KEYS[attack_key]}']"
            )
        for sentence in registered:
            if sentence not in attacked.context:
                violations.append(
                    f"oración registrada en metadata ausente del contexto: \"{sentence[:80]}\""
                )

    elif attack_key == "ds":
        if attacked.context != original.context:
            violations.append("el context fue modificado en un ataque de opciones")
        if len(attacked.options) != len(original.options):
            violations.append(
                f"cantidad de opciones distinta: {len(attacked.options)} vs {len(original.options)}"
            )
        else:
            correct_idx = _find_option_index(original.options, original.label)
            if correct_idx is None:
                violations.append("no se pudo ubicar la opción correcta según el label")
            else:
                if attacked.options[correct_idx] != original.options[correct_idx]:
                    violations.append("la opción correcta fue modificada")
                if intensity in (IntensityLevel.MEDIUM, IntensityLevel.HIGH):
                    incorrect_changed = any(
                        attacked.options[i] != original.options[i]
                        for i in range(len(original.options))
                        if i != correct_idx
                    )
                    if not incorrect_changed:
                        violations.append(
                            "ninguna opción incorrecta fue modificada (requerido en MEDIUM/HIGH)"
                        )

    elif attack_key == "ss":
        original_sentences = _split_sentences(original.context)
        attacked_sentences = _split_sentences(attacked.context)
        if sorted(original_sentences) != sorted(attacked_sentences):
            violations.append("el multiset de oraciones no coincide con el original")
        elif attacked_sentences == original_sentences:
            violations.append("el orden de las oraciones no cambió")

    else:
        violations.append(f"ataque desconocido: '{attack_key}'")

    return violations


def _parse_judge_json(raw: str) -> Optional[Dict[str, Any]]:
    cleaned = (raw or "").strip().replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return None
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if "label_preserved" not in parsed or "spec_compliant" not in parsed:
        return None
    return parsed


class AttackValidator:
    """Valida que un ataque generado cumpla su especificación.

    Combina checks estructurales deterministas con un juez LLM (cliente 'judge',
    de familia distinta al atacante). Para 'ss' (determinístico) solo aplica
    los checks estructurales.
    """

    def __init__(self, judge: Optional[LLMClient] = None):
        self.judge = judge

    def semantic_validation(
        self, original: VerbalTask, attacked: VerbalTask, attack_key: str
    ) -> Dict[str, Any]:
        rubric = ATTACK_RUBRICS.get(attack_key, "")
        prompt = f"""
        Eres un juez experto en ataques adversariales para evaluación de robustez de LLMs.
        Un modelo atacante modificó una tarea de opción múltiple. Evalúa si el ataque
        fue generado correctamente.

        ESPECIFICACIÓN DEL ATAQUE ({attack_key}):
        {rubric}

        CONTEXTO ORIGINAL:
        "{original.context}"

        CONTEXTO ATACADO:
        "{attacked.context}"

        PREGUNTA: {original.question}
        OPCIONES ORIGINALES: {original.options}
        OPCIONES ATACADAS: {attacked.options}
        RESPUESTA CORRECTA: {original.label}

        Responde ÚNICAMENTE con un objeto JSON con estas claves exactas:
        - "label_preserved" (bool): dado el contexto ATACADO junto con la pregunta y las
          opciones atacadas, ¿la respuesta correcta sigue siendo inequívocamente
          "{original.label}"? false si el ataque volvió el ítem ambiguo o cambió la
          respuesta correcta.
        - "spec_compliant" (bool): ¿la modificación cumple la especificación del ataque
          descrita arriba?
        - "explanation" (str): 1-2 oraciones justificando tu evaluación.

        Sin markdown, sin texto adicional antes o después del JSON.
        """

        judge_model = self.judge.model_id if self.judge else None
        # Ante JSON inválido, reintentar la consulta al juez 1 vez
        for _ in range(2):
            raw = self.judge.query(prompt)
            parsed = _parse_judge_json(raw)
            if parsed is not None:
                return {
                    "label_preserved": bool(parsed.get("label_preserved")),
                    "spec_compliant": bool(parsed.get("spec_compliant")),
                    "explanation": str(parsed.get("explanation", "")),
                    "judge_error": False,
                    "judge_model": judge_model,
                }
        return {
            "label_preserved": False,
            "spec_compliant": False,
            "explanation": "El juez no devolvió un JSON válido.",
            "judge_error": True,
            "judge_model": judge_model,
        }

    def validate(
        self, original: VerbalTask, attacked: VerbalTask, attack_key: str
    ) -> Dict[str, Any]:
        """Devuelve {"valid": bool, "failures": [...], "structural_violations": [...], "semantic": dict|None}."""
        violations = structural_checks(original, attacked, attack_key)
        report: Dict[str, Any] = {
            "structural_violations": violations,
            "semantic": None,
        }

        if violations:
            report["valid"] = False
            report["failures"] = list(violations)
            return report

        # 'ss' es determinístico: no requiere juez LLM
        if attack_key == "ss" or self.judge is None:
            report["valid"] = True
            report["failures"] = []
            return report

        semantic = self.semantic_validation(original, attacked, attack_key)
        report["semantic"] = semantic

        failures: List[str] = []
        if semantic["judge_error"]:
            failures.append("juez LLM: respuesta no parseable (judge_error)")
        else:
            if not semantic["label_preserved"]:
                failures.append(f"juez LLM: label no preservado — {semantic['explanation']}")
            if not semantic["spec_compliant"]:
                failures.append(f"juez LLM: incumple la especificación — {semantic['explanation']}")

        report["valid"] = not failures
        report["failures"] = failures
        return report
