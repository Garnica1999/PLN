# ADR-2 — Anti-leakage: doble escenario de evaluación

- **Status:** Aceptado (2026-09-08)
- **Date:** 2026-09-08
- **Context:** Las etiquetas de área (ADR-1) derivan de citas a la CP presentes
  en el texto; un clasificador podría "aprender la etiqueta" memorizando las
  citas en vez del contenido temático (señal de alarma del flujo: métricas
  sospechosamente perfectas).
- **Decision:** Todo modelo de M4 se evalúa en DOS escenarios: (a) texto
  completo y (b) texto con citas normativas enmascaradas (patrones M2
  reemplazados por token `[CITA]`). Se reporta macro-F1 de ambos + gap. Gap
  grande = el modelo depende de las citas → se itera (más regularización o
  features de contenido).
- **Alternatives Considered:**
  - Solo texto completo: descartado (oculta el leakage).
  - Solo texto enmascarado: descartado (pierde señal legítima de contexto).
  - Eliminar citas del train: descartado (distorsiona la distribución real).
- **Consequences:** Gate crítico de `ml-reviewer` antes de reportar M4 como
  éxito; el benchmark final es 2×2 (modelo × escenario). Métricas perfectas
  (~1.0) detienen el flujo hasta descartar leakage.
