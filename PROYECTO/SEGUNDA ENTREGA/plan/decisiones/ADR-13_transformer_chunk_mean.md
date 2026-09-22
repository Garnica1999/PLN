# ADR-13: Transformer chunk-mean BETO (revisa ADR-10)

- **Status:** Aceptado
- **Date:** 2026-09-11
- **Context:** ADR-10 (v1) determinó que BETO con truncamiento cabeza+cola 256+256
  perdía contra LogReg (0.2080/0.2042 vs 0.2870) porque los documentos (~11k
  palabras) exceden la ventana 512 tokens (~95 % del contenido descartado). El
  ciclo v2 implementó el remedio de la literatura (Chalkidis 2022, arXiv:2211.00974):
  chunk-mean con sliding window (256 tokens, stride 128, máx 64 chunks con
  submuestreo uniforme np.linspace), agregación por media de embeddings de chunks
  y AMP fp16. Un deadlock del DataLoader multiproceso (Python 3.14 + fork) obligó
  a pre-tokenizar los chunks a disco (script 14, cache `data/cache_chunks_beto/`,
  1,3 GB) y entrenar con `num_workers=0`; el fix del entrenamiento incluyó un
  `scaler.update()` faltante en el cierre de época (crash reproducido y corregido).
- **Decision:** Mantener LogReg+TF-IDF (v1, con calibración C de Fase U) como
  ClasificadorArea del producto. El transformer chunk-mean se documenta como
  mejora metodológica pero NO se adopta.
- **Evidencia (resultados/m4_transformer_chunks.json, test 16 clases):**
  - completo: **0.2433** (acc 0.5634; val 0.2313), enmascarado: **0.2363** (acc
    0.5403; val 0.2256); gap leakage 0.0070 (sin leakage).
  - vs BETO v1 truncado (0.2080/0.2042): **+17 % / +16 %** → el método chunk-mean
    reduce la brecha causada por el truncamiento (confirma el diagnóstico ADR-10).
  - meta del plan >= 0.26: NO cumplida (0.2433).
  - vs LogReg v1 0.2870 / Fase U C 0.2980: BETO sigue por debajo; 5,15 h GPU,
    52,7 chunks/doc promedio.
- **Alternatives Considered:**
  - BETO truncado (v1): descartado (ya en ADR-10).
  - legal-longformer-4096-es: no evaluado (29 descargas, riesgo de calidad;
    prioridad baja tras confirmar la mejora de chunk-mean).
  - Fine-tuning más largo/épocas extra: no planificado (comparabilidad v1).
- **Consequences:**
  - (+) ADR-10 queda confirmado con la corrección metodológica aplicada: la
    derrota no era del transformer per se sino del truncamiento.
  - (+) Cache de chunks reutilizable para futuros modelos (1,3 GB, 3,6 min de
    cómputo) y pipeline sin workers (sin deadlocks).
  - (-) BETO chunk-mean no cierra la brecha con LogReg: el producto mantiene
    features TF-IDF de documento completo.
  - (-) El incidente de Python 3.14 + DataLoader queda documentado (evidencia en
    `experimentos/*faulthandler_evidencia.log` y crash `*crash_epoch2*`).
