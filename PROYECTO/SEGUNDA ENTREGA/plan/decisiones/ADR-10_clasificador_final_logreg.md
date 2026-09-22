# ADR-10: Clasificador de área final — LogReg+TF-IDF lemas sobre BETO fine-tuned

## Status
ACEPTADO (2026-09-19) — benchmark 2×2 completado, LogReg gana.

## Date
2026-09-09

## Context
La Fase L3 fine-tuneó BETO (dccuchile/bert-base-spanish-wwm-cased) para
clasificación de área temática (16 clases) con los splits oficiales
(data/splits_m4.parquet, 16.557/3.549/3.548), truncamiento cabeza+cola
256+256 tokens (ADR-3), 3 épocas, batch efectivo 32, lr 2e-5, seed 42,
en doble escenario anti-leakage (ADR-2: texto crudo completo vs citas
enmascaradas con regex v2 sobre texto crudo). El baseline clásico M4
(LogReg sobre TF-IDF de lemas, 100k features, ngram 1-2) ya estaba
auditorado sin leakage (L2).

## Decision
Conservar **LogReg+TF-IDF lemas (macro-F1 test 0.2870, enmascarado)** como
clasificador de área del producto. BETO fine-tuned queda como comparativa
documentada (modelos guardados en modelos/m4_transformer/).

## Alternatives Considered
1. **BETO fine-tuned** (elegido el experimento, no el despliegue):
   completo 0.2080 / enmascarado 0.2042 (acc ~51 %). Gap leakage 0.0039
   → sin leakage, resultado legítimo.
2. **Chunk-mean pooling** (ADR-3 variante b): descartado por presupuesto
   de GPU (requeriría ~20x el tiempo de cómputo para cubrir los ~11k
   palabras/doc promedio; la GPU es compartida y el guard de VRAM ya
   esperó 3 h solo para arrancar).

## Consequences
- **Por qué gana LogReg**: las sentencias promedian ~11k palabras; la
  ventana de 512 tokens descarta ~95 % del contenido (incluidas las
  secciones de fondo/consideraciones donde vive la señal temática),
  mientras que TF-IDF sobre lemas procesa el documento completo. El
  resultado es consistente con la literatura de LJP con documentos
  largos (Chalkidis et al. 2022: los modelos de bolsa de palabras son
  competitivos o superiores a transformers en textos legales largos).
- El gap leakage de BETO (0.0039) replica el hallazgo L2: el modelo
  aprende contenido temático, no atajos de cita.
- Curvas de entrenamiento sanas (val 0.13→0.19→0.21, sin saturar): con
  más épocas/chunk-mean BETO podría cerrar la brecha, pero el costo
  GPU no lo justifica frente a la mejora marginal esperada.
- El producto (10_evaluar_sentencia.py) NO requiere cambios: ya usa
  ClasificadorArea con LogReg.

## Evidence
- resultados/m4_transformer.json (métricas completas, config reproducible).
- experimentos/m4_transformer.log (curvas por época, ambos escenarios).
- modelos/m4_transformer/beto_{completo,enmascarado}/ (checkpoints).
- Benchmark 2×2: LogReg 0.2852/0.2870 vs BETO 0.2080/0.2042
  (completo/enmascarado, macro-F1 test).
