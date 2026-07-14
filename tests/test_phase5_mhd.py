"""
Verificación Fase 5: el ataque MHD inserta TODAS las distracciones generadas
(1/2/4 según intensidad), sin truncarlas ni perderlas aunque el contexto tenga
pocas oraciones, y metadata['distractions_added'] coincide cadena a cadena con
lo insertado en el contexto.
Ejecutar: python tests/test_phase5_mhd.py  (también compatible con pytest)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.attack_validator import structural_checks
from core.models import IntensityLevel, VerbalTask
from core.testing import MockClient
from core.utils import split_sentences
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


DISTRACTIONS = [
    "La biblioteca del colegio permanece abierta hasta las ocho de la noche.",
    "El aula de historia tiene mapas antiguos colgados en las paredes.",
    "Varios estudiantes usan tarjetas de memoria para repasar fechas.",
    "El profesor de historia lleva quince años enseñando en el colegio.",
]

EXPECTED_COUNTS = {
    IntensityLevel.LOW: 1,
    IntensityLevel.MEDIUM: 2,
    IntensityLevel.HIGH: 4,
}


def _apply_mhd(task: VerbalTask, intensity: IntensityLevel, sentences: list) -> VerbalTask:
    response = json.dumps(sentences, ensure_ascii=False)
    attack = load_attack_class("mhd")(MockClient([response]))
    return attack.apply(task, intensity)


def test_mhd_inserts_all_distractions_per_intensity():
    for intensity, count in EXPECTED_COUNTS.items():
        task = make_task()
        attacked = _apply_mhd(task, intensity, DISTRACTIONS[:count])

        assert "error" not in attacked.metadata, f"{intensity.name}: {attacked.metadata}"

        # El número de oraciones añadidas coincide con la intensidad,
        # aunque el contexto tenga menos oraciones que distracciones (HIGH: 4 > 3)
        added = len(split_sentences(attacked.context)) - len(split_sentences(task.context))
        assert added == count, f"{intensity.name}: se añadieron {added}, se esperaban {count}"

        # Metadata registra exactamente las distracciones insertadas
        registered = attacked.metadata["distractions_added"]
        assert len(registered) == count, f"{intensity.name}: metadata registra {len(registered)}"

        # Cada distracción de metadata aparece ÍNTEGRA (substring exacto) en el contexto
        for sentence in registered:
            assert sentence in attacked.context, (
                f"{intensity.name}: distracción ausente del contexto: {sentence!r}"
            )


def test_mhd_preserves_original_sentences():
    for intensity, count in EXPECTED_COUNTS.items():
        task = make_task()
        attacked = _apply_mhd(task, intensity, DISTRACTIONS[:count])

        original_sentences = split_sentences(task.context)
        attacked_sentences = split_sentences(attacked.context)

        # Todas las oraciones originales siguen presentes y en su orden relativo
        iterator = iter(attacked_sentences)
        assert all(sentence in iterator for sentence in original_sentences), (
            f"{intensity.name}: contexto original no íntegro"
        )

        # Nada más cambió
        assert attacked.id == task.id
        assert attacked.question == task.question
        assert attacked.options == task.options
        assert attacked.label == task.label


def test_mhd_normalizes_metadata_to_match_context():
    # Distracciones sin punto final y con espacios raros: metadata debe guardar
    # la misma cadena normalizada que se inserta en el contexto
    task = make_task()
    messy = [
        "La biblioteca  permanece abierta\nhasta las ocho",
        "¿Sabías que el aula tiene mapas antiguos?",
    ]
    attacked = _apply_mhd(task, IntensityLevel.MEDIUM, messy)

    registered = attacked.metadata["distractions_added"]
    assert registered == [
        "La biblioteca permanece abierta hasta las ocho.",
        "¿Sabías que el aula tiene mapas antiguos?",
    ]
    for sentence in registered:
        assert sentence in attacked.context


def test_mhd_passes_structural_checks():
    for intensity, count in EXPECTED_COUNTS.items():
        task = make_task()
        attacked = _apply_mhd(task, intensity, DISTRACTIONS[:count])
        violations = structural_checks(task, attacked, "mhd")
        assert violations == [], f"{intensity.name}: {violations}"


if __name__ == "__main__":
    tests = [
        test_mhd_inserts_all_distractions_per_intensity,
        test_mhd_preserves_original_sentences,
        test_mhd_normalizes_metadata_to_match_context,
        test_mhd_passes_structural_checks,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print("\nVerificación Fase 5: todas las pruebas pasaron.")
