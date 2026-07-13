"""
Verificación Fase 3 (parte 1): AttackValidator — checks estructurales,
juez LLM con reintento de parseo y retry loop de generación.
Ejecutar: python tests/test_phase3_attack_validator.py  (también compatible con pytest)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.attack_validator import AttackValidator, structural_checks
from core.models import IntensityLevel, VerbalTask
from core.testing import MockClient
from main import MAX_ATTEMPTS, apply_validated_attack, load_attack_class


JUDGE_OK = json.dumps(
    {"label_preserved": True, "spec_compliant": True, "explanation": "Ataque correcto."}
)
JUDGE_BAD_LABEL = json.dumps(
    {"label_preserved": False, "spec_compliant": True, "explanation": "El ítem quedó ambiguo."}
)


def make_task() -> VerbalTask:
    return VerbalTask(
        id="001",
        task="mcq",
        context=(
            "Ana estudió toda la semana para el examen de historia. "
            "El examen cubría la revolución industrial. "
            "Su profesor le recomendó repasar las fechas clave. "
            "Ana obtuvo la nota más alta de su clase."
        ),
        question="¿Por qué Ana obtuvo la nota más alta?",
        options=[
            "A: Porque estudió toda la semana.",
            "B: Porque copió en el examen.",
            "C: Porque tuvo suerte.",
        ],
        label="A",
        rationale="El contexto indica que estudió toda la semana.",
    )


GOOD_SENTENCE = "La biblioteca del colegio permanece abierta hasta las ocho."


def test_attack_valid_on_first_attempt():
    task = make_task()
    attack = load_attack_class("isi")(MockClient([json.dumps([GOOD_SENTENCE], ensure_ascii=False)]))
    validator = AttackValidator(MockClient([JUDGE_OK]))

    attacked = apply_validated_attack(attack, validator, task, IntensityLevel.LOW, "isi", "mock/attacker")

    assert attacked.metadata["attack_valid"] is True
    assert attacked.metadata["generation_attempts"] == 1
    assert attacked.metadata["attacker_model"] == "mock/attacker"
    assert attacked.metadata["attack_validation"]["valid"] is True
    assert "validation_failures" not in attacked.metadata


def test_attack_regenerated_after_structural_failure():
    task = make_task()
    # Primer intento: respuesta vacía (falla checks); segundo: correcto
    attack = load_attack_class("isi")(
        MockClient(["", json.dumps([GOOD_SENTENCE], ensure_ascii=False)])
    )
    judge = MockClient([JUDGE_OK])
    validator = AttackValidator(judge)

    attacked = apply_validated_attack(attack, validator, task, IntensityLevel.LOW, "isi", "mock/attacker")

    assert attacked.metadata["attack_valid"] is True
    assert attacked.metadata["generation_attempts"] == 2
    # El juez solo se consulta cuando los checks estructurales pasan
    assert len(judge.calls) == 1
    # La tarea original no fue mutada por los intentos fallidos
    assert "attack" not in task.metadata


def test_attack_invalid_after_exhausting_attempts():
    task = make_task()
    attack = load_attack_class("isi")(MockClient(default_response=""))
    judge = MockClient([JUDGE_OK])
    validator = AttackValidator(judge)

    attacked = apply_validated_attack(attack, validator, task, IntensityLevel.HIGH, "isi", "mock/attacker")

    assert attacked.metadata["attack_valid"] is False
    assert attacked.metadata["generation_attempts"] == MAX_ATTEMPTS
    assert attacked.metadata["validation_failures"]
    assert len(judge.calls) == 0


def test_attack_invalid_when_judge_rejects_label():
    task = make_task()
    attack = load_attack_class("isi")(
        MockClient(default_response=json.dumps([GOOD_SENTENCE], ensure_ascii=False))
    )
    validator = AttackValidator(MockClient(default_response=JUDGE_BAD_LABEL))

    attacked = apply_validated_attack(attack, validator, task, IntensityLevel.LOW, "isi", "mock/attacker")

    assert attacked.metadata["attack_valid"] is False
    assert any("label no preservado" in f for f in attacked.metadata["validation_failures"])


def test_judge_retries_once_on_invalid_json():
    task = make_task()
    attacked = task.model_copy(deep=True)
    attacked.context = f"{task.context} {GOOD_SENTENCE}"
    judge = MockClient(["esto no es JSON", "tampoco esto"])
    validator = AttackValidator(judge)

    semantic = validator.semantic_validation(task, attacked, "isi")

    assert len(judge.calls) == 2
    assert semantic["judge_error"] is True
    assert semantic["label_preserved"] is False and semantic["spec_compliant"] is False


def test_structural_checks_context_attacks():
    task = make_task()

    # OK: inyección al final registrada en metadata
    attacked = task.model_copy(deep=True)
    attacked.context = f"{task.context} {GOOD_SENTENCE}"
    attacked.metadata.update(
        {"intensity": "low", "injected_sentences": [GOOD_SENTENCE]}
    )
    assert structural_checks(task, attacked, "isi") == []

    # Violación: contexto original reescrito
    rewritten = attacked.model_copy(deep=True)
    rewritten.context = "Un contexto completamente distinto. " + GOOD_SENTENCE
    assert any("substring" in v for v in structural_checks(task, rewritten, "isi"))

    # Violación: label alterado y clave error presente
    broken = attacked.model_copy(deep=True)
    broken.label = "B"
    broken.metadata["error"] = "algo"
    violations = structural_checks(task, broken, "isi")
    assert any("label" in v for v in violations)
    assert any("error" in v for v in violations)

    # Violación: se añadieron 2 oraciones cuando LOW espera 1
    extra = attacked.model_copy(deep=True)
    extra.context = f"{task.context} {GOOD_SENTENCE} Otra oración añadida de más."
    assert any("se esperaban 1" in v for v in structural_checks(task, extra, "isi"))


def test_structural_checks_mhd_interleaved():
    task = make_task()
    distractions = ["Los mapas antiguos decoraban el aula.", "El colegio celebraba su aniversario."]
    attack = load_attack_class("mhd")(MockClient([json.dumps(distractions, ensure_ascii=False)]))
    attacked = attack.apply(task, IntensityLevel.MEDIUM)
    assert structural_checks(task, attacked, "mhd") == []


def test_structural_checks_ds():
    task = make_task()

    # OK: opciones incorrectas reforzadas, la correcta intacta
    attacked = task.model_copy(deep=True)
    attacked.options = [
        task.options[0],
        "B: Porque memorizó los apuntes del profesor.",
        "C: Porque las preguntas coincidieron con su repaso.",
    ]
    attacked.metadata["intensity"] = "medium"
    assert structural_checks(task, attacked, "ds") == []

    # Violación: la opción correcta fue modificada
    bad = attacked.model_copy(deep=True)
    bad.options[0] = "A: Porque estudió muchísimo toda la semana."
    assert any("opción correcta" in v for v in structural_checks(task, bad, "ds"))

    # Violación: en HIGH ninguna incorrecta cambió
    unchanged = task.model_copy(deep=True)
    unchanged.metadata["intensity"] = "high"
    assert any("incorrecta" in v for v in structural_checks(task, unchanged, "ds"))

    # Violación: el contexto no debe cambiar en ds
    ctx = attacked.model_copy(deep=True)
    ctx.context = task.context + " Oración extra."
    assert any("context" in v for v in structural_checks(task, ctx, "ds"))


def test_structural_checks_ss():
    task = make_task()
    attack = load_attack_class("ss")()
    attacked = attack.apply(task, IntensityLevel.MEDIUM)
    assert structural_checks(task, attacked, "ss") == []
    # 'ss' válido sin juez
    assert AttackValidator(judge=None).validate(task, attacked, "ss")["valid"] is True

    # Violación: se perdió una oración
    dropped = attacked.model_copy(deep=True)
    dropped.context = "Ana estudió toda la semana para el examen de historia."
    assert any("multiset" in v for v in structural_checks(task, dropped, "ss"))

    # Violación: el orden no cambió
    same = task.model_copy(deep=True)
    same.metadata["intensity"] = "medium"
    assert any("orden" in v for v in structural_checks(task, same, "ss"))


if __name__ == "__main__":
    tests = [
        test_attack_valid_on_first_attempt,
        test_attack_regenerated_after_structural_failure,
        test_attack_invalid_after_exhausting_attempts,
        test_attack_invalid_when_judge_rejects_label,
        test_judge_retries_once_on_invalid_json,
        test_structural_checks_context_attacks,
        test_structural_checks_mhd_interleaved,
        test_structural_checks_ds,
        test_structural_checks_ss,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print("\nVerificación Fase 3 (AttackValidator): todas las pruebas pasaron.")
