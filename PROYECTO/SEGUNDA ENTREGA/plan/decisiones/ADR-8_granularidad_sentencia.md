# ADR-8 — Granularidad de análisis: sentencia completa

- **Status:** Aceptado (2026-09-08)
- **Date:** 2026-09-08
- **Context:** El experimento 9.6 comparó sentencia vs párrafo vs subsentencia
  (muestra estratificada seed 42, 600 docs, marcadores = citas CP + "derecho a X").
  Criterio pre-registrado: adoptar unidad fina solo si pureza +15 % sin perder
  contexto.
- **Decision:** Unidad primaria = **sentencia completa** para M4 (y base de M2/M7).
  Las subsentencias quedan como herramienta auxiliar de localización de
  evidencia (p. ej. extraer la ventana donde aparece una cita), no como unidad
  de clasificación.
- **Alternatives Considered:**
  - Párrafo: descartado por experimento (el corpus no trae `\n\n`: párrafo =
    sentencia, 600/600 unidades, 97.17 % con marcador, idéntico).
  - Subsentencia como unidad: descartada por experimento (219.649 unidades,
    solo 4.64 % con marcador, media 0.05/unidad: fragmenta el contexto y
    diluye la señal; no supera el criterio +15 %).
- **Consequences:** M4 entrena sobre sentencia completa (ADR-2/ADR-3 aplican);
  M2 puede usar ventanas de subsentencia para reportar evidencia. Si a futuro
  se detectan secciones estructurales (encabezado/considerandos/resuelve),
  reabrir este ADR con experimento de marcadores de sección.
- **Evidencia:** `resultados/eda_extendido/granularidad.csv`,
  `ejemplos_granularidad.md`, `fig_granularidad.png`.
