# robustness-assessment-in-reasoning

Evaluación de robustez en razonamiento mediante ataques contextuales y adversariales en modelos de lenguaje en español.

## Estructura unificada

```
robustness-assessment-in-reasoning/
├── main.py                          # Punto de entrada único
├── core/
│   ├── base_attack.py               # Clase abstracta BaseAttack
│   ├── models.py                    # VerbalTask, IntensityLevel
│   ├── services.py                  # GeminiAPI, TaskResolver, TaskValidator
│   └── executor.py                  # AttackedVerbalTasksExecutor
├── irrelevant_sentence_injection/
├── contradictory-noise-injection/
├── distractor-strengthening/
├── multi-hop-distraction/
├── sentence-shuffling/
└── dataset/
    ├── og_dataset.json
    └── attacked/{ataque}/{ataque}_{low|medium|high}_dataset.json
```

## Configuración

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Crear `.env` en la raíz con `GEMINI_API_KEY=tu_clave`.

## Uso

Desde la raíz del proyecto:

```bash
python main.py
```

Claves de ataque: `isi`, `cni`, `ds`, `mhd`, `ss`.

Los resultados del pipeline de evaluación se guardan en `isi_full_evaluation_results.json`.
