import sys
import os
import json
import time

# Adjust path to import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.services import GeminiAPI
from core.models import IntensityLevel, VerbalTask
from core.executor import AttackedVerbalTasksExecutor
from irrelevant_sentence_injection import IrrelevantSentenceInjection


def generate_attack_datasets(intensity_level: IntensityLevel = None):
    """
    Generates attacked datasets. If intensity_level is specified, only that
    dataset is generated. Otherwise, all datasets are generated.
    """
    print("Generating attack datasets...")
    try:
        api = GeminiAPI()
        attack = IrrelevantSentenceInjection(api)

        with open("../dataset/og_dataset.json", "r", encoding="utf-8") as f:
            original_dataset = [VerbalTask(**task) for task in json.load(f)]

        intensities_to_generate = (
            [intensity_level] if intensity_level else list(IntensityLevel)
        )

        for intensity in intensities_to_generate:
            attacked_tasks = []
            print(f"\n--- Generating {intensity.name} dataset ---")
            for task in original_dataset:
                print(f"Applying {intensity.name} attack to task {task.id}...")
                attacked_task = attack.apply(task, intensity)
                attacked_tasks.append(attacked_task)
                time.sleep(4)  # To avoid hitting API rate limits

            output_path = f"../dataset/isi/isi_{intensity.value}_dataset.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    [task.dict() for task in attacked_tasks],
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
            print(f"Saved {intensity.name} attack dataset to {output_path}")

        print("\nFinished generating attack datasets.")

    except Exception as e:
        print(f"An error occurred during dataset generation: {e}")


def execute_evaluation_pipeline():
    print("Executing evaluation pipeline...")
    try:
        api = GeminiAPI()
        executor = AttackedVerbalTasksExecutor(api)

        original_dataset_path = "../dataset/og_dataset.json"
        attacked_dataset_paths = {
            intensity.value: f"../dataset/isi/isi_{intensity.value}_dataset.json"
            for intensity in IntensityLevel
        }

        executor.execute_attacks_pipeline(original_dataset_path, attacked_dataset_paths)
        print("Evaluation pipeline finished.")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")


def main():
    while True:
        print("\n--- Irrelevant Sentence Injection Attack Menu ---")
        print("1. Generate ALL Attack Datasets")
        print("2. Generate LOW Intensity Dataset")
        print("3. Generate MEDIUM Intensity Dataset")
        print("4. Generate HIGH Intensity Dataset")
        print("5. Execute Evaluation Pipeline")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            generate_attack_datasets()
        elif choice == "2":
            generate_attack_datasets(IntensityLevel.LOW)
        elif choice == "3":
            generate_attack_datasets(IntensityLevel.MEDIUM)
        elif choice == "4":
            generate_attack_datasets(IntensityLevel.HIGH)
        elif choice == "5":
            execute_evaluation_pipeline()
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
