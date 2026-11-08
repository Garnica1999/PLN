# ADR-3 — Documentos largos (~11.000 palabras) vs límite 512 tokens

- **Status:** Aceptado (2026-09-08)
- **Date:** 2026-09-08
- **Context:** Promedio 10.530 palabras limpias/sentencia; los transformers
  (BETO, LEGAL-BERT) aceptan ~512 tokens. Truncar al inicio pierde la parte
  resolutiva y los considerandos finales.
- **Decision:** Estrategia en dos niveles: (a) baseline con truncamiento
  cabeza+cola (primeros 256 + últimos 256 tokens, preserva encabezado y
  resuelve); (b) experimento chunk-mean: dividir en chunks de 512 con solape,
  promediar embeddings [CLS] de chunks. Se comparan ambas en validación
  (macro-F1) y se adopta la ganadora. Modelos clásicos (TF-IDF) usan texto
  completo como referencia.
- **Alternatives Considered:**
  - Solo primeros 512: descartado (sesgo hacia encabezado).
  - Modelos long-context (Longformer): descartados en primera iteración (sin
    checkpoint legal español disponible; coste GPU alto).
  - Resumen extractivo previo: descartado por ahora (añade un componente no
    evaluado a la cadena).
- **Consequences:** `07_transformer.py` implementa ambas variantes con el
  mismo split; la decisión queda registrada en `metricas.json`.
