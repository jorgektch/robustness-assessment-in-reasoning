"""
Punto de entrada unificado: generación de datasets atacados y pipeline de evaluación.
Ejecutar desde la raíz del proyecto: python main.py
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Type

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.executor import AttackedVerbalTasksExecutor
from core.models import IntensityLevel, VerbalTask
from core.llm_client import build_client
from core.base_attack import BaseAttack

ORIGINAL_DATASET = ROOT / "dataset" / "og_dataset.json"
DATASET_DIR = ROOT / "dataset" / "attacked"
API_SLEEP_SECONDS = 4

ATTACK_REGISTRY: Dict[str, Dict] = {
    "isi": {
        "folder": "irrelevant_sentence_injection",
        "module": "irrelevant_sentence_injection",
        "class": "IrrelevantSentenceInjection",
        "label": "Irrelevant Sentence Injection",
    },
    "cni": {
        "folder": "contradictory-noise-injection",
        "module": "contradictory_noise_injection",
        "class": "ContradictoryNoiseInjection",
        "label": "Contradictory Noise Injection",
    },
    "ds": {
        "folder": "distractor-strengthening",
        "module": "distractor_strengthening",
        "class": "DistractorStrengthening",
        "label": "Distractor Strengthening",
    },
    "mhd": {
        "folder": "multi-hop-distraction",
        "module": "multi_hop_distraction",
        "class": "MultiHopDistraction",
        "label": "Multi-Hop Distraction",
    },
    "ss": {
        "folder": "sentence-shuffling",
        "module": "sentence-shuffling",
        "class": "SentenceShuffling",
        "label": "Sentence Shuffling",
    },
    "phm": {
        "folder": "premise-hypothesis-mismatch",
        "module": "premise_hypothesis_mismatch",
        "class": "PremiseHypothesisMismatch",
        "label": "Premise-Hypothesis Mismatch",
    },
}


def load_attack_class(attack_key: str) -> Type[BaseAttack]:
    spec_entry = ATTACK_REGISTRY[attack_key]
    module_path = ROOT / spec_entry["folder"] / f"{spec_entry['module']}.py"
    spec = importlib.util.spec_from_file_location(spec_entry["module"], module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el módulo: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, spec_entry["class"])


def attacked_dataset_path(attack_key: str, intensity: IntensityLevel) -> Path:
    return DATASET_DIR / attack_key / f"{attack_key}_{intensity.value}_dataset.json"


def load_original_dataset() -> List[VerbalTask]:
    with open(ORIGINAL_DATASET, "r", encoding="utf-8") as f:
        return [VerbalTask(**task) for task in json.load(f)]


def save_dataset(tasks: List[VerbalTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [task.model_dump() for task in tasks],
            f,
            ensure_ascii=False,
            indent=4,
        )


def generate_attack_datasets(
    attack_key: str,
    intensity: Optional[IntensityLevel] = None,
) -> None:
    attacker = build_client("attacker")
    attack_cls = load_attack_class(attack_key)
    attack = attack_cls(attacker)
    original_dataset = load_original_dataset()
    intensities = [intensity] if intensity else list(IntensityLevel)
    label = ATTACK_REGISTRY[attack_key]["label"]

    print(f"\n=== Generando datasets: {label} ===")
    for level in intensities:
        output_path = attacked_dataset_path(attack_key, level)

        if output_path.exists():
            print(f"\n--- Intensidad {level.name} --- [YA EXISTE, saltando]")
            continue

        attacked_tasks = []
        print(f"\n--- Intensidad {level.name} ---")
        for task in original_dataset:
            print(f"  Atacando tarea {task.id}...")
            attacked_task = attack.apply(task, level)
            attacked_task.metadata["attacker_model"] = attacker.model_id
            attacked_tasks.append(attacked_task)
            time.sleep(API_SLEEP_SECONDS)

        save_dataset(attacked_tasks, output_path)
        print(f"Guardado en {output_path}")


def generate_all_attacks(intensity: Optional[IntensityLevel] = None) -> None:
    for attack_key in ATTACK_REGISTRY:
        generate_attack_datasets(attack_key, intensity)


def execute_evaluation_pipeline(attack_keys: Optional[List[str]] = None) -> None:
    keys = attack_keys or list(ATTACK_REGISTRY.keys())
    solver = build_client("solver")
    executor = AttackedVerbalTasksExecutor(solver)

    attacked_paths: Dict[str, str] = {}
    for attack_key in keys:
        for intensity in IntensityLevel:
            path = attacked_dataset_path(attack_key, intensity)
            if path.exists():
                attacked_paths[f"{attack_key}_{intensity.value}"] = str(path)
            else:
                print(f"[AVISO] No existe dataset: {path}")

    if not attacked_paths:
        print("No hay datasets atacados. Genera datasets primero (opción del menú).")
        return

    executor.execute_attacks_pipeline(str(ORIGINAL_DATASET), attacked_paths)


def print_attack_menu() -> None:
    print("\n--- Ataques disponibles ---")
    for key, entry in ATTACK_REGISTRY.items():
        print(f"  {key}: {entry['label']}")


def main() -> None:
    while True:
        print("\n========== Robustness Assessment — Menú principal ==========")
        print("1. Generar datasets de UN ataque (todas las intensidades)")
        print("2. Generar datasets de UN ataque (una intensidad)")
        print("3. Generar TODOS los ataques (todas las intensidades)")
        print("4. Ejecutar pipeline de evaluación (todos los datasets existentes)")
        print("5. Ejecutar evaluación de un ataque concreto")
        print("6. Salir")
        print_attack_menu()

        choice = input("\nOpción: ").strip()

        if choice == "1":
            attack_key = input("Clave del ataque (isi, cni, ds, mhd, ss): ").strip().lower()
            if attack_key in ATTACK_REGISTRY:
                generate_attack_datasets(attack_key)
            else:
                print("Clave de ataque no válida.")

        elif choice == "2":
            attack_key = input("Clave del ataque: ").strip().lower()
            level_name = input("Intensidad (low / medium / high): ").strip().lower()
            if attack_key not in ATTACK_REGISTRY:
                print("Clave de ataque no válida.")
                continue
            try:
                level = IntensityLevel(level_name)
            except ValueError:
                print("Intensidad no válida.")
                continue
            generate_attack_datasets(attack_key, level)

        elif choice == "3":
            generate_all_attacks()

        elif choice == "4":
            execute_evaluation_pipeline()

        elif choice == "5":
            attack_key = input("Clave del ataque: ").strip().lower()
            if attack_key in ATTACK_REGISTRY:
                execute_evaluation_pipeline([attack_key])
            else:
                print("Clave de ataque no válida.")

        elif choice == "6":
            break

        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()
