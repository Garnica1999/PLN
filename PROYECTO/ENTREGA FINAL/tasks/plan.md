# Plan de ejecución — Entrega Final (E3) · Juris-PLN

Fecha límite: 2026-11-09 23:59
Plan de referencia: `PLAN_ENTREGA_FINAL.md` (en `/home/carlos/.opencode/plan/`)
Tablero: `tasks/todo.md` (se actualiza fase a fase)

## F0 — Fundaciones (31 oct – 1 nov)
| Tarea | Criterio de aceptación |
|---|---|
| Deps y entorno | langchain + langchain-ollama + ollama + python-pptx instalados en el venv; `import` OK |
| Estructura `ENTREGA FINAL/` | carpetas app/backend, app/frontend, informe, presentacion, codigo, evaluacion, datos, tasks |
| Dirección de diseño | `DESIGN.md` con paleta, tipografía y dials derivados del deck del EDA |
| ADRs | ADR-22 (arquitectura final) y ADR-23 (guardrails) escritos y enlazados |
| GPU | chequeo previo documentado (sección 11 del plan) |
| Backend mínimo | FastAPI: `/health`, `/v1/models`, `/v1/validate`, `/v1/link`; OpenAPI; tests pytest |
| Agente v1 | LangChain tool-calling + router de intención + grounding; `/v1/ask` |
| Frontend mínimo | Streamlit separado (httpx): P1 + P3 + selector de modelo + panel M4 exploratorio |

## F1 — Núcleo (2 – 4 nov)
- P1 completa: evidencia por artículo (fragmentos resaltados), contexto de citas, badge temporal.
- P2: artículo → jurisprudencia (filtros año/tipo, PageRank, evolución).
- Guardrails v1 completos + gold set del agente (~100 prompts) y métricas de ruteo.
- Aceptación: P1/P2 usables por HTTP contra el backend; ruteo correcto en el gold set.

## F2 — Adicionales (5 – 6 nov)
- P3 evaluación/transparencia, P4 ficha PDF, P5 similares (si el tiempo lo permite).
- Comparador de rutas (v2.2 vs baseline) y panel M4 exploratorio afinado.
- Aceptación: las 4 pantallas funcionan y el PDF se descarga.

## F3 — Validación y consolidación (7 – 8 nov)
- Evaluación del pipeline (ruteo, rechazo, saludos, grounding, latencia) + pytest de API.
- Consolidación E1+E2 en `ENTREGA FINAL/` (sin duplicar 47 GB) y documentación actualizada.
- Informe PDF completo (EDA + E2 + E3) y deck storytelling (10-12 slides, diseño del EDA).
- Aceptación: informe y deck revisados; Delivery Gate de antislop en PASS.

## F4 — Entrega (9 nov)
- Integración final, pruebas E2E, zip/repo, ensayo de demo y entrega antes de 23:59.
- Aceptación: demo local reproducible y paquete completo.

## Reglas transversales
- GPU: verificar disponibilidad antes de cada uso (sección 11 del plan).
- Antislop: modo during (session override); Delivery Gate antes de entregar.
- Un experimento por script nuevo; no sobrescribir artefactos congelados.
- Documentación al día en cada fase (sección 10 del plan).
