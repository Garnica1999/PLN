# ADR-5 — Evaluación del vínculo CP con weak labels + estratificación temporal

- **Status:** Aceptado (2026-09-08)
- **Date:** 2026-09-08
- **Context:** M7 debe medir si el retrieval sentencia→artículos CP es robusto
  al cambio de redacción en 30 años. No hay ground truth manual de
  "artículos que sustentan cada sentencia".
- **Decision:** (1) Weak labels: las citas reales a la CP extraídas por el NER
  (M2) con verificación muestral manual. (2) Protocolo temporal: benchmark con
  1 sentencia por año (30 casos, estratificados por tipo) + métricas globales,
  por año y por periodo (1992-2001/2002-2011/2012-2021). Métricas: P@k, R@k,
  F1@k, MRR. (3) Tres estrategias comparadas: cita exacta (regex), BM25/TF-IDF,
  embeddings (GPU). La ganadora se integra al producto con score de confianza.
- **Alternatives Considered:**
  - Anotación experta completa: descartada por coste.
  - Muestra aleatoria sin estratificar por año: descartada (no mide deriva
    temporal, que es la pregunta central de M7).
  - Solo precisión global: descartada (oculta degradación en años antiguos).
- **Consequences:** `11_evaluar_vinculo_cp.py` implementa el protocolo; la
  figura de deriva temporal (métrica vs año) es entregable obligatorio.
