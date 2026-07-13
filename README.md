# robustness-assessment-in-reasoning

Librería de **ataques adversariales contextuales** para evaluar la robustez de razonamiento de LLMs en tareas verbales de opción múltiple (MCQ) en español. Incluye la generación de datasets adversariales y un **sistema de validación de ataques** con roles separados (atacante / solver / juez) para evitar circularidad.

## Pipeline

```
                 ┌─────────────────────┐
 dataset         │ 1. GENERACIÓN       │   modelo ATACANTE (ej. Groq/Llama)
 original ──────►│ attack.apply()      │   genera la perturbación
 (docente)       └─────────┬───────────┘
                           ▼
                 ┌─────────────────────┐   checks deterministas (sin API)
                 │ 2. VALIDACIÓN       │   + juez LLM (ej. OpenRouter/DeepSeek)
                 │ AttackValidator     │   si falla → REGENERAR (máx. 3 intentos)
                 └─────────┬───────────┘   marca metadata.attack_valid
                           ▼
                 dataset/attacked/{ataque}/{ataque}_{intensidad}_dataset.json
                           ▼
                 ┌─────────────────────┐   modelo SOLVER (ej. Gemini) resuelve;
                 │ 3. EVALUACIÓN       │   el JUEZ puntúa el razonamiento (RQS);
                 │ Executor            │   métricas SOLO sobre ataques válidos
                 └─────────────────────┘   → AVR, N válido, Accuracy, ΔAcc, Flip Rate
```

Cada instancia registra la procedencia de los tres roles: `metadata.attacker_model`, `results.solver_model` y `validation.judge_model`.

## Estructura

```
robustness-assessment-in-reasoning/
├── main.py                          # CLI: generación, validación y evaluación
├── core/
│   ├── base_attack.py               # Clase abstracta BaseAttack
│   ├── models.py                    # VerbalTask, IntensityLevel
│   ├── llm_client.py                # LLMClient, GeminiClient, OpenAICompatibleClient, build_client
│   ├── services.py                  # TaskResolver, ResponseEvaluator (RQS)
│   ├── attack_validator.py          # AttackValidator: checks estructurales + juez LLM
│   ├── executor.py                  # AttackedVerbalTasksExecutor: pipeline y métricas
│   ├── utils.py                     # Utilidades compartidas de parsing/inserción
│   └── testing.py                   # MockClient para pruebas sin red
├── irrelevant_sentence_injection/   # isi
├── contradictory-noise-injection/   # cni
├── distractor-strengthening/        # ds
├── multi-hop-distraction/           # mhd
├── sentence-shuffling/              # ss (determinístico, sin API)
├── premise-hypothesis-mismatch/     # phm
├── tests/                           # pytest (usa MockClient, no toca red)
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

### Roles en `.env` (separación anti-circularidad)

El mismo modelo **no** debe generar los ataques, resolver las tareas y juzgar la calidad: eso sesga las conclusiones (auto-preferencia, ataques triviales para sí mismo). Los tres roles se configuran por variables de entorno en `.env` (ver `.env.example`), cada uno con una **familia de modelo distinta**:

```
ATTACKER_PROVIDER=openai_compatible      # genera los ataques
ATTACKER_BASE_URL=https://api.groq.com/openai/v1
ATTACKER_MODEL=llama-3.3-70b-versatile
ATTACKER_API_KEY=...

SOLVER_PROVIDER=gemini                   # modelo evaluado
SOLVER_MODEL=gemini-2.5-flash
GEMINI_API_KEY=...

JUDGE_PROVIDER=openai_compatible         # valida ataques y puntúa razonamiento
JUDGE_BASE_URL=https://openrouter.ai/api/v1
JUDGE_MODEL=deepseek/deepseek-chat
JUDGE_API_KEY=...
```

`ORIGINAL_DATASET_PATH` (opcional) apunta al dataset original provisto por el docente; por defecto `dataset/og_dataset.json`. Al iniciar, la CLI muestra qué modelo ocupa cada rol y **advierte si dos roles comparten familia**.

## Uso

Desde la raíz del proyecto:

```bash
python main.py
```

Menú: generar datasets atacados (por ataque y/o intensidad), ejecutar el pipeline de evaluación, y **validar datasets atacados existentes** (corre solo el `AttackValidator` sobre datasets ya generados y persiste `attack_valid` en su metadata — útil para datasets antiguos).

Claves de ataque: `isi`, `cni`, `ds`, `mhd`, `ss`, `phm`. Los resultados del pipeline de evaluación se guardan en `isi_full_evaluation_results.json`.

### Reproducir el experimento completo

1. Configurar `.env` con los tres roles (familias distintas).
2. `python main.py` → opción 3 (generar todos los ataques). Cada ataque se valida al generarse; si falla, se regenera hasta 3 veces.
3. Opción 4 (pipeline de evaluación): el solver resuelve el dataset original y los atacados; el juez calcula el Reasoning Quality Score; se emite el JSON de análisis con las métricas.

### Tests

```bash
pytest tests/
```

Usan `MockClient` (sin tocar red ni consumir API).

## Validación de ataques y métrica AVR

**¿Cómo sabemos que un ataque fue generado correctamente?** Cada instancia atacada pasa por `AttackValidator`:

1. **Checks deterministas (sin API):** `id`/`label`/`question` intactos; el contexto original íntegro dentro del atacado (substring exacto, u orden relativo preservado en MHD); número de oraciones añadidas según la intensidad; en DS la opción correcta intacta y las incorrectas modificadas; en SS mismo multiset de oraciones en orden distinto.
2. **Juez LLM** (familia distinta al atacante): evalúa `label_preserved` (¿la respuesta correcta sigue siendo inequívoca?) y `spec_compliant` (¿cumple el rubric de su ataque?).
3. **Retry:** si la validación falla, el ataque se regenera (máx. 3 intentos). Si se agotan, la instancia queda marcada `attack_valid=false` con sus `validation_failures`.

**AVR (Attack Validity Rate)** = ataques válidos / ataques generados, por ataque e intensidad (`{name}_avr`, `{name}_n_valid` en el análisis).

**¿Por qué condicionar las métricas a ataques válidos?** Un "ataque" que rompe sus propias reglas (cambia la respuesta correcta, contradice hechos, reescribe el contexto) no mide robustez: mide otra cosa. Si el solver falla ante un ítem cuya respuesta correcta ya no es la etiquetada, ese fallo no es evidencia de fragilidad del razonamiento. Por eso Accuracy, ΔAccuracy y Flip Rate se calculan **solo sobre instancias con `attack_valid == true`**, y se reporta el N válido junto al AVR para dimensionar la evidencia.
