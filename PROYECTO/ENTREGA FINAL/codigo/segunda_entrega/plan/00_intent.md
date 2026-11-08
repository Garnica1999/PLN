# Intención confirmada E2 (2026-09-08)

## Producto norte ("Juris-PLN")
Sistema que, dada una sentencia de la Corte Constitucional: (1) valida que sea
realmente una providencia de la Corte, (2) determina su área temática
constitucional, (3) identifica los artículos de la Constitución Política que la
sustentan (recuperación robusta al cambio de redacción en 30 años), y (4) la
sitúa en su contexto jurisprudencial (precedentes citados y evolución del área,
1992–2021).

## Alcance acordado
- M0 validación de entrada, M1 base lingüística, M2 NER jurídico, M3 puente
  Constitución (etiquetas de área), M4 clasificación (clásicos + transformers,
  anti-leakage), M5 red de citas, M6 diacronía, M7 benchmark vínculo CP
  (3 estrategias × 30 años) + validación de entrada.
- EDA extendido en notebook E1 (sección 9, código previo intacto): n-gramas,
  colocaciones, lematización vs stemmización, lexemas/ontologías,
  morfológico/lexicográfico, granularidad. Figuras + logs en texto.
- Motor de datos: PySpark + Parquet (adopción directa, ADR-7).
- Todo con "por qué" justificado por experimentación (ADRs 1-8).

## Out of scope (por ahora)
Informe final, diapositivas E2, demo streamlit, zip de entrega, commits/PR
(solo cuando Carlos lo pida).
