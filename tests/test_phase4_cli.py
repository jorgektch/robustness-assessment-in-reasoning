"""
Verificación Fase 4: ORIGINAL_DATASET_PATH por entorno y validación de
datasets atacados ya existentes (opción 6 del menú).
Ejecutar: python tests/test_phase4_cli.py  (también compatible con pytest)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main
from core.attack_validator import AttackValidator
from core.models import VerbalTask
from core.testing import MockClient

JUDGE_OK = json.dumps(
    {"label_preserved": True, "spec_compliant": True, "explanation": "Ataque correcto."}
)

ORIGINAL = {
    "id": "001",
    "task": "mcq",
    "context": "Ana estudió toda la semana. El examen cubría historia. Ana obtuvo la nota más alta.",
    "question": "¿Por qué Ana obtuvo la nota más alta?",
    "options": ["A: Porque estudió.", "B: Porque copió."],
    "label": "A",
    "rationale": "r",
    "metadata": {},
    "results": {},
    "validation": {},
}


def test_original_dataset_path_respects_env():
    previous = os.environ.pop("ORIGINAL_DATASET_PATH", None)
    try:
        assert main.original_dataset_path() == main.DEFAULT_ORIGINAL_DATASET
        os.environ["ORIGINAL_DATASET_PATH"] = "otra/ruta/dataset.json"
        assert main.original_dataset_path() == Path("otra/ruta/dataset.json")
    finally:
        os.environ.pop("ORIGINAL_DATASET_PATH", None)
        if previous is not None:
            os.environ["ORIGINAL_DATASET_PATH"] = previous


def test_validate_existing_datasets_persists_flags():
    injected = "La biblioteca del colegio permanece abierta hasta las ocho."
    valid_attacked = dict(
        ORIGINAL,
        context=f"{ORIGINAL['context']} {injected}",
        metadata={"attack": "irrelevant_sentence_injection", "intensity": "low",
                  "injected_sentences": [injected]},
    )
    invalid_attacked = dict(
        ORIGINAL,
        id="002",
        metadata={"attack": "irrelevant_sentence_injection", "intensity": "low",
                  "injected_sentences": [], "error": "insufficient_sentences"},
    )
    original_002 = dict(ORIGINAL, id="002")

    previous_env = os.environ.pop("ORIGINAL_DATASET_PATH", None)
    previous_dataset_dir = main.DATASET_DIR
    previous_sleep = main.API_SLEEP_SECONDS
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        og_path = tmp_path / "og_dataset.json"
        og_path.write_text(
            json.dumps([ORIGINAL, original_002], ensure_ascii=False), encoding="utf-8"
        )
        attacked_path = tmp_path / "attacked" / "isi" / "isi_low_dataset.json"
        attacked_path.parent.mkdir(parents=True)
        attacked_path.write_text(
            json.dumps([valid_attacked, invalid_attacked], ensure_ascii=False),
            encoding="utf-8",
        )

        try:
            os.environ["ORIGINAL_DATASET_PATH"] = str(og_path)
            main.DATASET_DIR = tmp_path / "attacked"
            main.API_SLEEP_SECONDS = 0

            main.validate_existing_datasets(
                ["isi"], validator=AttackValidator(MockClient(default_response=JUDGE_OK))
            )

            saved = [
                VerbalTask(**task)
                for task in json.loads(attacked_path.read_text(encoding="utf-8"))
            ]
            assert saved[0].metadata["attack_valid"] is True
            assert saved[1].metadata["attack_valid"] is False
            assert saved[1].metadata["validation_failures"]
            # El contenido de las tareas no se altera
            assert saved[0].context == valid_attacked["context"]
            assert saved[0].options == ORIGINAL["options"]
        finally:
            os.environ.pop("ORIGINAL_DATASET_PATH", None)
            if previous_env is not None:
                os.environ["ORIGINAL_DATASET_PATH"] = previous_env
            main.DATASET_DIR = previous_dataset_dir
            main.API_SLEEP_SECONDS = previous_sleep


if __name__ == "__main__":
    tests = [
        test_original_dataset_path_respects_env,
        test_validate_existing_datasets_persists_flags,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print("\nVerificación Fase 4: todas las pruebas pasaron.")
