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
    print(f"\n{'='*70}")
    print(f"APLICANDO ATAQUE: MULTI-HOP DISTRACTION - Nivel: {intensity.upper()}")
    print(f"{'='*70}")
    
    attack = MultiHopDistraction(intensity=intensity)
    dataset_atacado = []
    stats = {"total": len(dataset), "exitosos": 0, "errores": 0, "warnings": 0}
    
    for i, ejemplo in enumerate(dataset, 1):
        print(f"\n[{i}/{len(dataset)}] Procesando ejemplo {ejemplo.get('id', '???')}...")
        
        try:
            ejemplo_atacado = attack.apply(ejemplo)
            dataset_atacado.append(ejemplo_atacado)
            
            metadata = ejemplo_atacado.get("metadata", {})
            if "error" in metadata:
                stats["errores"] += 1
                print(f"  [ERROR] Error: {metadata.get('error_details', 'Unknown')}")
            elif metadata.get("validation_warning"):
                stats["warnings"] += 1
                print(f"  [WARNING] Warning: Validación falló")
            else:
                stats["exitosos"] += 1
                print(f"  [OK] Éxito: {metadata.get('num_distractions', 0)} distracciones añadidas")
        except Exception as e:
            print(f"  [ERROR] Error crítico: {e}")
            stats["errores"] += 1
            ejemplo["metadata"] = {"attack": "multi_hop_distraction", "intensity": intensity, 
                                   "error": "critical_error", "error_details": str(e)}
            dataset_atacado.append(ejemplo)
    
    return dataset_atacado, stats


def guardar_dataset(dataset, output_path):
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\nDataset guardado en: {output_path}")
        return True
    except Exception as e:
        print(f"\nError al guardar dataset: {e}")
        return False


def generar_reporte(stats_por_nivel, output_dir, timestamp):
    reporte_path = os.path.join(output_dir, f"reporte_multihop_{timestamp}.txt")
    
    try:
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("REPORTE DE ATAQUE: MULTI-HOP DISTRACTION\n")
            f.write("="*70 + "\n\n")
            f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            for nivel, stats in stats_por_nivel.items():
                f.write(f"\n{'='*70}\n")
                f.write(f"NIVEL: {nivel.upper()}\n")
                f.write(f"{'='*70}\n")
                f.write(f"Total de ejemplos:        {stats['total']}\n")
                f.write(f"Procesados exitosamente:  {stats['exitosos']} ({stats['exitosos']/stats['total']*100:.1f}%)\n")
                f.write(f"Con warnings:             {stats['warnings']} ({stats['warnings']/stats['total']*100:.1f}%)\n")
                f.write(f"Con errores:              {stats['errores']} ({stats['errores']/stats['total']*100:.1f}%)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("RESUMEN GENERAL\n")
            f.write("="*70 + "\n")
            total_general = sum(s['total'] for s in stats_por_nivel.values())
            exitosos_general = sum(s['exitosos'] for s in stats_por_nivel.values())
            f.write(f"Total de ejemplos procesados: {total_general}\n")
            f.write(f"Tasa de éxito general:        {exitosos_general/total_general*100:.1f}%\n")
        
        print(f"\nReporte guardado en: {reporte_path}")
    except Exception as e:
        print(f"\nNo se pudo generar el reporte: {e}")


def main():
    print("="*70)
    print("MULTI-HOP DISTRACTION ATTACK")
    print("="*70)
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("Uso: python main.py [dataset.json]")
            print("Sin argumentos usa: ../dataset/dataset_base.json")
            sys.exit(0)
        input_dataset = sys.argv[1]
        print(f"\nUsando dataset especificado: {input_dataset}")
    else:
        input_dataset = "../dataset/dataset_base.json"
        print(f"\nUsando dataset por defecto: {input_dataset}")
    
    print("\nValidando dataset...")
    valido, mensaje = validar_archivo(input_dataset)
    
    if not valido:
        print(f"ERROR: {mensaje}")
        print("\nUso correcto:")
        print("  python main.py [ruta_al_dataset.json]")
        sys.exit(1)
    
    print("Dataset válido")
    
    with open(input_dataset, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Dataset cargado: {len(dataset)} ejemplos")
    
    dataset_name = os.path.splitext(os.path.basename(input_dataset))[0]
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    output_dir = "../dataset/attacked"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio creado: {output_dir}")
    
    niveles = ["low", "medium", "high"]
    stats_por_nivel = {}
    
    for nivel in niveles:
        dataset_atacado, stats = aplicar_ataque(dataset, nivel)
        stats_por_nivel[nivel] = stats
        
        output_filename = f"{dataset_name}_multihop_{nivel}_{timestamp}.json"
        output_path = os.path.join(output_dir, output_filename)
        guardar_dataset(dataset_atacado, output_path)
        
        print(f"\n{'='*70}")
        print(f"RESUMEN - Nivel {nivel.upper()}")
        print(f"{'='*70}")
        print(f"Total:     {stats['total']}")
        print(f"Exitosos:  {stats['exitosos']}")
        print(f"Warnings:  {stats['warnings']}")
        print(f"Errores:   {stats['errores']}")
        print(f"Tasa de éxito: {stats['exitosos']/stats['total']*100:.1f}%")
    
    generar_reporte(stats_por_nivel, output_dir, timestamp)
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)
    print(f"\nSe generaron 3 datasets atacados en: {output_dir}")
    print(f"   - {dataset_name}_multihop_low_{timestamp}.json")
    print(f"   - {dataset_name}_multihop_medium_{timestamp}.json")
    print(f"   - {dataset_name}_multihop_high_{timestamp}.json")
    print("\nReporte detallado: reporte_multihop_{}.txt".format(timestamp))
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
