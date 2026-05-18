import os
import json
import sys
from datetime import datetime
from multi_hop_distraction import MultiHopDistraction


def validar_archivo(filepath):
    if not os.path.exists(filepath):
        return False, f"El archivo no existe: {filepath}"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if not isinstance(dataset, list) or len(dataset) == 0:
            return False, "El dataset debe ser una lista no vacía"
        
        campos_requeridos = ["id", "task", "context", "question", "options", "label"]
        for campo in campos_requeridos:
            if campo not in dataset[0]:
                return False, f"Falta el campo requerido: '{campo}'"
        
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"Error al parsear JSON: {e}"
    except Exception as e:
        return False, f"Error inesperado: {e}"


def aplicar_ataque(dataset, intensity):
    attack = MultiHopDistraction(intensity=intensity)
    dataset_atacado = []
    
    for ejemplo in dataset:
        try:
            ejemplo_atacado = attack.apply(ejemplo)
            dataset_atacado.append(ejemplo_atacado)
        except Exception as e:
            ejemplo["metadata"] = {"attack": "multi_hop_distraction", "intensity": intensity, 
                                   "error": "critical_error", "error_details": str(e)}
            dataset_atacado.append(ejemplo)
    
    return dataset_atacado


def guardar_dataset(dataset, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("Uso: python main.py [dataset.json]")
            sys.exit(0)
        input_dataset = sys.argv[1]
    else:
        input_dataset = "../dataset/dataset_base.json"
    
    valido, mensaje = validar_archivo(input_dataset)
    if not valido:
        print(f"ERROR: {mensaje}")
        sys.exit(1)
    
    with open(input_dataset, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    dataset_name = os.path.splitext(os.path.basename(input_dataset))[0]
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    output_dir = "../dataset/attacked"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for nivel in ["low", "medium", "high"]:
        dataset_atacado = aplicar_ataque(dataset, nivel)
        output_filename = f"{dataset_name}_multihop_{nivel}_{timestamp}.json"
        output_path = os.path.join(output_dir, output_filename)
        guardar_dataset(dataset_atacado, output_path)


if __name__ == "__main__":
    main()
