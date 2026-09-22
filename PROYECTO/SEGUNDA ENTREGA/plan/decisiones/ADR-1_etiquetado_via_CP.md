# ADR-1 — Etiquetado de "área" derivado de la Constitución

- **Status:** Aceptado (2026-09-08)
- **Date:** 2026-09-08
- **Context:** El proyecto necesita etiquetas de área temática para clasificar
  sentencias (M4), pero el corpus no trae metadatos de área. Anotación manual de
  23.742 sentencias es inviable.
- **Decision:** Derivar el área de cada sentencia desde la Constitución Política
  de 1991 parseada: artículo citado → título/capítulo → bloque temático =
  etiqueta de área (M3). Las citas reales a la CP se extraen con el NER (M2).
  Para sentencias sin cita CP explícita (p. ej. tutelas): fallback "derecho a X"
  → derechos fundamentales; se reporta % de cobertura de cada vía.
- **Alternatives Considered:**
  - Anotación manual: descartada por volumen.
  - Clustering no supervisado como etiquetas: descartado (etiquetas no
    interpretables jurídicamente, no alineadas con la CP).
  - Etiquetas por tipo T/C/SU: insuficientes (pregunta real es temática, no procesal).
- **Consequences:** Etiquetas 100 % reproducibles y jurídicamente
  interpretables; acopla M2→M3→M4. Riesgo: leakage (el área deriva de citas
  presentes en el texto) → mitigado por ADR-2.
