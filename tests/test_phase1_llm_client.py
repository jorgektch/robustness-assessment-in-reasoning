"""
Verificación Fase 1: abstracción LLMClient, factory por variables de entorno,
MockClient y ejecución de los 6 ataques sin tocar red.
Ejecutar: python tests/test_phase1_llm_client.py  (también compatible con pytest)
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.llm_client import (
    GeminiClient,
    OpenAICompatibleClient,
    build_client,
    detect_shared_families,
)
from core.models import IntensityLevel, VerbalTask
from core.services import TaskResolver, TaskValidator
from core.testing import MockClient
from main import load_attack_class


def make_task() -> VerbalTask:
    return VerbalTask(
        id="t1",
        task="reading_comprehension",
        context=(
            "Ana estudió toda la semana para el examen de historia. "
            "El examen cubría la revolución industrial. "
            "Su profesor le recomendó repasar las fechas clave. "
            "Ana obtuvo la nota más alta de su clase."
        ),
        question="¿Por qué Ana obtuvo la nota más alta?",
        options=["A: Porque estudió toda la semana", "B: Porque copió", "C: Por suerte"],
        label="A",
        rationale="El contexto indica que estudió toda la semana.",
    )


def test_factory_respects_env():
    env = {
        "ATTACKER_PROVIDER": "openai_compatible",
        "ATTACKER_BASE_URL": "https://api.groq.com/openai/v1",
        "ATTACKER_MODEL": "llama-3.3-70b-versatile",
        "ATTACKER_API_KEY": "clave-falsa-groq",
        "SOLVER_PROVIDER": "gemini",
        "SOLVER_MODEL": "gemini-2.5-flash",
        "SOLVER_API_KEY": "clave-falsa-gemini",
        "JUDGE_PROVIDER": "openai_compatible",
        "JUDGE_BASE_URL": "https://openrouter.ai/api/v1",
        "JUDGE_MODEL": "deepseek/deepseek-chat",
        "JUDGE_API_KEY": "clave-falsa-openrouter",
    }
    os.environ.update(env)

    attacker = build_client("attacker")
    solver = build_client("solver")
    judge = build_client("judge")

    assert isinstance(attacker, OpenAICompatibleClient)
    assert attacker.model_id == "groq/llama-3.3-70b-versatile"
    assert isinstance(solver, GeminiClient)
    assert solver.model_id == "gemini/gemini-2.5-flash"
    assert isinstance(judge, OpenAICompatibleClient)
    assert judge.model_id == "openrouter/deepseek/deepseek-chat"

    warnings_found = detect_shared_families(
        {"attacker": attacker, "solver": solver, "judge": judge}
    )
    assert warnings_found == [], f"No debería haber advertencias: {warnings_found}"


def test_shared_family_warning():
    clients = {
        "attacker": MockClient(model_name="llama-3.3-70b-versatile"),
        "solver": MockClient(model_name="llama-4-maverick"),
        "judge": MockClient(model_name="deepseek-chat"),
    }
    warnings_found = detect_shared_families(clients)
    assert len(warnings_found) == 1, f"Se esperaba 1 advertencia: {warnings_found}"
    assert "llama" in warnings_found[0]


def test_attacks_run_with_mock_client():
    task = make_task()

    # ISI (implementación actual, pre-Fase 2: devuelve el texto completo modificado)
    isi_response = task.context + " Oración inyectada de prueba."
    isi = load_attack_class("isi")(MockClient([isi_response]))
    attacked = isi.apply(task, IntensityLevel.LOW)
    assert attacked.context == isi_response
    assert attacked.metadata.get("attack") == "IrrelevantSentenceInjection"

    # CNI
    noise = "Sin embargo, algunos registros sugieren que hubo dudas."
    cni = load_attack_class("cni")(MockClient([noise]))
    attacked = cni.apply(task, IntensityLevel.MEDIUM)
    assert attacked.context == f"{task.context} {noise}"
    assert attacked.metadata.get("injected_noise") == noise
    assert "error" not in attacked.metadata

    # DS
    new_options = ["A: Porque estudió toda la semana", "B: Porque memorizó apuntes", "C: Por repasar fechas"]
    ds = load_attack_class("ds")(MockClient([json.dumps(new_options, ensure_ascii=False)]))
    attacked = ds.apply(task, IntensityLevel.HIGH)
    assert attacked.options == new_options
    assert "error" not in attacked.metadata

    # MHD
    distractions = ["La revolución industrial comenzó en Inglaterra.", "Los exámenes escritos datan de la China imperial."]
    mhd = load_attack_class("mhd")(MockClient([json.dumps(distractions, ensure_ascii=False)]))
    attacked = mhd.apply(task, IntensityLevel.MEDIUM)
    assert attacked.metadata.get("num_distractions") == 2
    for sentence in distractions:
        assert sentence in attacked.context
    assert "error" not in attacked.metadata

    # SS (determinístico, sin API)
    ss = load_attack_class("ss")()
    attacked = ss.apply(task, IntensityLevel.MEDIUM)
    assert sorted(attacked.context.split()) == sorted(task.context.split())
    assert attacked.context != task.context

    # PHM
    premise = "Algunos compañeros comentaron que Ana parecía nerviosa antes del examen."
    phm = load_attack_class("phm")(MockClient([premise]))
    attacked = phm.apply(task, IntensityLevel.LOW)
    assert attacked.context.endswith(premise)
    assert task.context in attacked.context
    assert "error" not in attacked.metadata


def test_provenance_recorded_by_resolver_and_validator():
    task = make_task()

    solver = MockClient(
        [json.dumps({"label": "A", "rationale": "Estudió toda la semana."})],
        model_name="solver-model",
    )
    TaskResolver().resolve(task, solver)
    assert task.results["label"] == "A"
    assert task.results["solver_model"] == "mock/solver-model"

    judge = MockClient(
        [json.dumps({"reasoning_quality_score": 5, "explanation": "Correct answer."})],
        model_name="judge-model",
    )
    TaskValidator().validate(task, task, judge)
    assert task.validation["is_correct"] is True
    assert task.validation["reasoning_quality_score"] == 5
    assert task.validation["judge_model"] == "mock/judge-model"


if __name__ == "__main__":
    tests = [
        test_factory_respects_env,
        test_shared_family_warning,
        test_attacks_run_with_mock_client,
        test_provenance_recorded_by_resolver_and_validator,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print("\nVerificación Fase 1: todas las pruebas pasaron.")
