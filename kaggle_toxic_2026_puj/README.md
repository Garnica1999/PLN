# KAGGLE_TOXIC_2026_PUJ

Solucion para la competencia academica de Kaggle `tweet-toxic-2026-puj`
(curso de Procesamiento de Lenguaje Natural, Maestria en Inteligencia
Artificial, Pontificia Universidad Javeriana).

## Tarea

Clasificacion multi-etiqueta de textos en espanol en tres etiquetas binarias:

- `HS`: discurso de odio (hate speech).
- `TR`: dirigido contra una persona o grupo (targeted).
- `AG`: agresividad.

Metrica oficial: F1 Score (leaderboard privado). Deadline: 2026-10-19.

## Restricciones

- Prohibido usar transformers/LLM y embeddings preentrenados.
- Toda representacion (TF-IDF, embeddings) se entrena desde cero con los datos
  de `toxic_train.csv`.
- El conjunto de test nunca se usa para entrenar, ajustar vocabularios,
  vectorizadores, umbrales ni seleccion de features.
- Prohibido usar etiquetas o corpus externos (ver `DECISIONS.md` D-002).

## Estructura

```
kaggle_toxic_2026_puj/
  data/{raw,interim,processed,external}   # datos oficiales y derivados
  notebooks/                              # 01_data_audit ... 10_final_pipeline
  src/                                    # modulos reutilizables
  features/                               # matrices de features cacheadas
  models/                                 # modelos entrenados
  experiments/                            # un script completo por experimento
  reports/{eda,figuras}                   # tablas, figuras y auditorias
  submissions/                            # envios versionados
  docs/competition_rules.md               # reglas verbatim del organizador
  PROJECT_STATE.md  RUNLOG.md  DECISIONS.md  experiments.csv
```

## Ejecucion

Entorno obligatorio: `/home/carlos/venv`.

```bash
# Auditoria de datos (Fase 0)
/home/carlos/venv/bin/python experiments/00_auditoria_datos.py
```

## Memoria del proyecto

Antes de continuar tras una sesion nueva, leer en orden:
`PROJECT_STATE.md`, `RUNLOG.md`, `DECISIONS.md`, `experiments.csv`.

## Estado

- Fase 0 (setup, datos, auditoria): completada 2026-09-22.
- Siguiente: Fase 1 (estado del arte).
