import json
import os
from typing import Dict, List, Optional
from .llm_client import LLMClient
from .services import TaskResolver, ResponseEvaluator
from .models import VerbalTask, IntensityLevel


class AttackedVerbalTasksExecutor:
    def __init__(self, api: LLMClient, judge_api: Optional[LLMClient] = None):
        # api = solver (modelo evaluado); judge_api = juez para el RQS (familia distinta)
        self.api = api
        self.judge_api = judge_api or api
        self.resolver = TaskResolver()
        self.evaluator = ResponseEvaluator()

    def resolve_verbal_tasks(self, datasets: Dict[str, List[VerbalTask]]):
        for name, tasks in datasets.items():
            print(f"Resolving tasks for dataset: {name}")
            for task in tasks:
                self.resolver.resolve(task, self.api)

    def validate_attack_results(self, datasets: Dict[str, List[VerbalTask]]):
        original_tasks_map = {task.id: task for task in datasets.get("og", [])}

        for name, attacked_tasks in datasets.items():
            if (
                name == "og"
            ):  # Original dataset doesn't need validation against itself in the same way
                for task in attacked_tasks:
                    self.evaluator.validate(task, task, self.judge_api)
                continue

            print(f"Validating tasks for dataset: {name}")
            for attacked_task in attacked_tasks:
                original_task = original_tasks_map.get(attacked_task.id)
                if original_task:
                    self.evaluator.validate(attacked_task, original_task, self.judge_api)

    def generate_analysis(self, datasets: Dict[str, List[VerbalTask]]) -> Dict:
        analysis = {}

        og_tasks = datasets.get("og", [])
        og_correct = sum(1 for task in og_tasks if task.validation.get("is_correct"))
        og_accuracy = og_correct / len(og_tasks) if og_tasks else 0
        analysis["original_accuracy"] = og_accuracy

        og_correct_ids = {
            task.id for task in og_tasks if task.validation.get("is_correct")
        }

        for name, tasks in datasets.items():
            if name == "og":
                continue

            # AVR = ataques válidos / ataques generados
            valid_tasks = [
                task for task in tasks if task.metadata.get("attack_valid") is True
            ]
            n_valid = len(valid_tasks)
            avr = n_valid / len(tasks) if tasks else 0

            # Accuracy, ΔAccuracy y Flip Rate SOLO sobre instancias con attack_valid == True
            correct_count = sum(
                1 for task in valid_tasks if task.validation.get("is_correct")
            )
            accuracy = correct_count / n_valid if n_valid else 0
            delta_accuracy = accuracy - og_accuracy

            valid_og_correct = [
                task for task in valid_tasks if task.id in og_correct_ids
            ]
            flipped_count = sum(
                1
                for task in valid_og_correct
                if not task.validation.get("is_correct")
            )
            flip_rate = flipped_count / len(valid_og_correct) if valid_og_correct else 0

            analysis[f"{name}_avr"] = avr
            analysis[f"{name}_n_valid"] = n_valid
            analysis[f"{name}_accuracy"] = accuracy
            analysis[f"{name}_delta_accuracy"] = delta_accuracy
            analysis[f"{name}_flip_rate"] = flip_rate

        return analysis

    def execute_attacks_pipeline(
        self, original_dataset_path: str, attacked_dataset_paths: Dict[str, str]
    ):
        # Load all datasets into memory
        all_datasets: Dict[str, List[VerbalTask]] = {}

        with open(original_dataset_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
            all_datasets["og"] = [VerbalTask(**task) for task in original_data]

        for name, path in attacked_dataset_paths.items():
            with open(path, "r", encoding="utf-8") as f:
                attacked_data = json.load(f)
                all_datasets[name] = [VerbalTask(**task) for task in attacked_data]

        # 1. Resolve tasks (adds 'model_answer' to each task object)
        self.resolve_verbal_tasks(all_datasets)

        # 2. Validate results (adds 'validation' to each task object)
        self.validate_attack_results(all_datasets)

        # 3. Generate analysis from the in-memory data
        analysis_results = self.generate_analysis(all_datasets)

        # 4. Prepare final output structure
        final_output = {
            "analysis": analysis_results,
            "datasets": {
                name: [task.model_dump() for task in tasks]
                for name, tasks in all_datasets.items()
            },
        }

        # 5. Save everything to a single file
        output_path = "isi_full_evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        print(f"Full evaluation pipeline completed. Results saved to {output_path}")
