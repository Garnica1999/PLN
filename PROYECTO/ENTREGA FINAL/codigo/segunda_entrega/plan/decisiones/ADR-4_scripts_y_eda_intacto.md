# ADR-4 — Arquitectura: scripts SOLID + notebook EDA intacto

- **Status:** Aceptado (2026-09-08)
- **Date:** 2026-09-08
- **Context:** El notebook E1 (`primera_entrega.ipynb`) es entregable validado
  (PR #10 fusionado, validación cruzada OK). E2 añade análisis pesados y un
  pipeline de modelado que no caben en un notebook.
- **Decision:** (1) El notebook E1 solo recibe la sección 9 "EDA extendido"
  como celdas NUEVAS al final; el código existente queda byte-idéntico
  (verificado por diff en Fase E). (2) Todo el pipeline E2 son scripts
  `scripts/NN_nombre.py` + `lib/` compartida (limpieza E1 portada, carga,
  constantes), estilo clean-code: docstrings, ruff, sin emojis, seeds fijas,
  1 experimento = 1 script (no se edita ni borra código previo).
- **Alternatives Considered:**
  - Todo en notebooks: descartado (no reproducible en background, mezcla
    análisis con pipeline).
  - Monolito único: descartado (inmantenible, sin trazabilidad por módulo).
- **Consequences:** Cada módulo M0-M7 es ejecutable y auditable por separado;
  `ml-reviewer` valida cada script antes de entrenar.
