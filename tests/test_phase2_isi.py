"""
Verificación Fase 2: el ataque ISI inyecta oraciones programáticamente,
preserva el contexto original íntegro y respeta el conteo por intensidad.
Ejecutar: python tests/test_phase2_isi.py  (también compatible con pytest)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.models import IntensityLevel, VerbalTask
from core.testing import MockClient
from core.utils import extract_sentence_list
from main import load_attack_class


def make_task() -> VerbalTask:
    return VerbalTask(
        id="t1",
        task="reading_comprehension",
        context=(
            "Ana estudió toda la semana para el examen de historia. "
            "El examen cubría la revolución industrial. "
            "Ana obtuvo la nota más alta de su clase."
        ),
        question="¿Por qué Ana obtuvo la nota más alta?",
        options=["A: Porque estudió toda la semana", "B: Porque copió", "C: Por suerte"],
        label="A",
        rationale="El contexto indica que estudió toda la semana.",
    )


SENTENCES = [
    "La biblioteca del colegio permanece abierta hasta las ocho.",
    "El aula de historia tiene mapas antiguos en las paredes.",
    "Varios estudiantes usan tarjetas de memoria para repasar.",
]


def test_isi_counts_match_intensity():
    expected = {
        IntensityLevel.LOW: 1,
        IntensityLevel.MEDIUM: 2,
        IntensityLevel.HIGH: 3,
    }
    for intensity, count in expected.items():
        task = make_task()
        response = json.dumps(SENTENCES[:count], ensure_ascii=False)
        attack = load_attack_class("isi")(MockClient([response]))
        attacked = attack.apply(task, intensity)

        # El contexto original debe ser substring exacto del atacado
        assert task.context in attacked.context, f"{intensity.name}: contexto original alterado"
        # El número de oraciones añadidas coincide con la intensidad
        injected = attacked.metadata["injected_sentences"]
        assert len(injected) == count, f"{intensity.name}: se esperaban {count} oraciones"
        for sentence in injected:
            assert sentence in attacked.context
        # Nada más cambió
        assert attacked.id == task.id
        assert attacked.question == task.question
        assert attacked.options == task.options
        assert attacked.label == task.label
        assert attacked.metadata["attack"] == "irrelevant_sentence_injection"
        assert attacked.metadata["intensity"] == intensity.value
        assert "error" not in attacked.metadata


def test_isi_prompt_includes_task_information():
    task = make_task()
    client = MockClient([json.dumps([SENTENCES[0]], ensure_ascii=False)])
    load_attack_class("isi")(client).apply(task, IntensityLevel.LOW)
    prompt = client.calls[0]
    assert task.context in prompt
    assert task.question in prompt
    assert task.label in prompt
    assert task.options[0] in prompt


def test_isi_handles_markdown_and_lists():
    task = make_task()
    # JSON con fences de markdown
    fenced = f"```json\n{json.dumps(SENTENCES[:2], ensure_ascii=False)}\n```"
    attacked = load_attack_class("isi")(MockClient([fenced])).apply(task, IntensityLevel.MEDIUM)
    assert len(attacked.metadata["injected_sentences"]) == 2
    assert "error" not in attacked.metadata

    # Lista numerada en texto plano (fallback)
    numbered = f"1. {SENTENCES[0]}\n2. {SENTENCES[1]}"
    attacked = load_attack_class("isi")(MockClient([numbered])).apply(task, IntensityLevel.MEDIUM)
    assert attacked.metadata["injected_sentences"] == SENTENCES[:2]

    # Exceso de oraciones: se trunca al conteo esperado
    excess = json.dumps(SENTENCES, ensure_ascii=False)
    attacked = load_attack_class("isi")(MockClient([excess])).apply(task, IntensityLevel.LOW)
    assert len(attacked.metadata["injected_sentences"]) == 1


def test_isi_failure_keeps_original_context():
    task = make_task()
    attacked = load_attack_class("isi")(MockClient([""])).apply(task, IntensityLevel.HIGH)
    assert attacked.context == task.context
    assert attacked.metadata["error"] == "insufficient_sentences"


def test_extract_sentence_list_utility():
    assert extract_sentence_list('["Una oración."]') == ["Una oración."]
    assert extract_sentence_list('{"sentences": ["A.", "B."]}') == ["A.", "B."]
    assert extract_sentence_list("- Primera\n- Segunda") == ["Primera.", "Segunda."]
    assert extract_sentence_list("") == []


if __name__ == "__main__":
    tests = [
        test_isi_counts_match_intensity,
        test_isi_prompt_includes_task_information,
        test_isi_handles_markdown_and_lists,
        test_isi_failure_keeps_original_context,
        test_extract_sentence_list_utility,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print("\nVerificación Fase 2: todas las pruebas pasaron.")
