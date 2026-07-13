"""
Verificación Fase 3 (parte 2): métrica AVR y métricas condicionadas a
ataques válidos en el executor; RQS con el cliente judge; alias deprecado.
Ejecutar: python tests/test_phase3_metrics.py  (también compatible con pytest)
"""

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.executor import AttackedVerbalTasksExecutor
from core.models import VerbalTask
from core.services import ResponseEvaluator, TaskValidator
from core.testing import MockClient


def make_task(task_id: str, is_correct: bool, attack_valid=None) -> VerbalTask:
    task = VerbalTask(
        id=task_id,
        task="mcq",
        context="Contexto de prueba con una oración.",
        question="¿Pregunta?",
        options=["A: sí.", "B: no."],
        label="A",
        rationale="r",
        validation={"is_correct": is_correct},
    )
    if attack_valid is not None:
        task.metadata["attack_valid"] = attack_valid
    return task


def test_avr_and_conditioned_metrics():
    # og: 4 tareas, 3 correctas (001, 002, 003) → accuracy original 0.75
    og = [
        make_task("001", True),
        make_task("002", True),
        make_task("003", True),
        make_task("004", False),
    ]
    # Ataque: 4 generados, 3 válidos (001 incorrecta, 002 correcta, 004 correcta);
    # 003 inválido debe quedar FUERA de todas las métricas.
    attacked = [
        make_task("001", False, attack_valid=True),
        make_task("002", True, attack_valid=True),
        make_task("003", False, attack_valid=False),
        make_task("004", True, attack_valid=True),
    ]

    executor = AttackedVerbalTasksExecutor(MockClient())
    analysis = executor.generate_analysis({"og": og, "isi_low": attacked})

    assert analysis["original_accuracy"] == 0.75
    # AVR = 3 válidos / 4 generados
    assert analysis["isi_low_avr"] == 0.75
    assert analysis["isi_low_n_valid"] == 3
    # Accuracy condicionada: 2 correctas de 3 válidas
    assert analysis["isi_low_accuracy"] == 2 / 3
    assert abs(analysis["isi_low_delta_accuracy"] - (2 / 3 - 0.75)) < 1e-9
    # Flip rate: válidas con og correcta = {001, 002}; flip solo 001 → 1/2
    assert analysis["isi_low_flip_rate"] == 0.5


def test_metrics_zero_when_no_valid_attacks():
    og = [make_task("001", True)]
    attacked = [make_task("001", False, attack_valid=False)]
    executor = AttackedVerbalTasksExecutor(MockClient())
    analysis = executor.generate_analysis({"og": og, "cni_high": attacked})
    assert analysis["cni_high_avr"] == 0
    assert analysis["cni_high_n_valid"] == 0
    assert analysis["cni_high_accuracy"] == 0
    assert analysis["cni_high_flip_rate"] == 0


def test_rqs_uses_judge_client():
    solver = MockClient(model_name="solver-model")
    judge = MockClient(
        default_response=json.dumps(
            {"reasoning_quality_score": 4, "explanation": "Correct answer."}
        ),
        model_name="judge-model",
    )
    executor = AttackedVerbalTasksExecutor(solver, judge)

    task = make_task("001", True)
    task.results = {"label": "A", "rationale": "r"}
    executor.validate_attack_results({"og": [task]})

    # El RQS lo calculó el juez, no el solver
    assert len(judge.calls) == 1
    assert len(solver.calls) == 0
    assert task.validation["judge_model"] == "mock/judge-model"
    assert task.validation["reasoning_quality_score"] == 4


def test_task_validator_alias_is_deprecated():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validator = TaskValidator()
    assert isinstance(validator, ResponseEvaluator)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


if __name__ == "__main__":
    tests = [
        test_avr_and_conditioned_metrics,
        test_metrics_zero_when_no_valid_attacks,
        test_rqs_uses_judge_client,
        test_task_validator_alias_is_deprecated,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print("\nVerificación Fase 3 (métricas AVR): todas las pruebas pasaron.")
