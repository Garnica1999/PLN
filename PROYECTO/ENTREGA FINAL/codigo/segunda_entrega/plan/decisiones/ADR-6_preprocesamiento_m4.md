# ADR-6 — Preprocesamiento para M4: crudo vs lemas vs stems

- **Status:** Aceptado (2026-09-08)
- **Date:** 2026-09-08
- **Context:** El experimento 9.3 (corpus completo, 23.742 docs) midió compresión,
  tiempo y calidad de colapso de spaCy-lemas vs Snowball-stems; el SOTA E2
  (BETO/Cañete 2020, E5/Wang 2022) usa tokenización subpalabra sobre texto crudo.
- **Decision:**
  - Modelos con tokenizador subpalabra (BETO/LEGAL-BERT, E5): **texto crudo**,
    sin lematizar ni stemizar (normalizar rompería la alineación con el
    vocabulario WordPiece/SentencePiece pre-entrenado).
  - Modelos clásicos (TF-IDF+SVM/LightGBM) y textometría: **lemas**; stems
    descartados como representación principal (truncan: derech, deb).
  - Los contadores de lemas/stems/POS/MTLD quedan como features auxiliares.
- **Alternatives Considered:**
  - Stems para todo: descartado por experimento (54.2 % colapsa formas
    distintas ilegibles; muestra 50 términos: stem trunca, lema conserva
    derechos→derecho, debe→deber).
  - Lemas para transformers: descartado (desalinea subpalabras; SOTA usa crudo).
- **Consequences:** `06_baseline_clasico.py` usa lemas; `07_transformer.py` usa
  crudo (ADR-3 para el manejo de longitud). Coste medido: spaCy 5.227 s vs
  stems 1.171 s (pago único, background con checkpoints).
- **Evidencia:** `resultados/eda_extendido/lemas_stems_pos/RESUMEN.json`
  (vocabs 224.468 → 195.959 lemas 87.3 % → 121.696 stems 54.2 %),
  `muestra_calidad.csv`, `fig_lemma_vs_stem.png`.
