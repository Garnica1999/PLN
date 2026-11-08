# ADR-22: Arquitectura final del producto (API, agente y frontend separados)

- **Status:** Aceptado (2026-10-31)
- **Date:** 2026-10-31
- **Context:** La Entrega III exige un modelo final funcional, evaluación, informe PDF y
  presentación storytelling (9 nov). El profesor pidió delimitar la utilidad del proyecto. El
  producto validado es el vínculo sentencia → artículos de la CP (v2.2: B2 + B1) con contexto de
  citas (M5); M4 quedó fuera por la falta de etiquetas fiables (ADR-19). Se necesita además un
  backend con API, un agente con guardrails y una interfaz, con frontend y backend separados.
- **Decision:** Producto = "dada una sentencia, artículos de la CP que la fundamentan con evidencia
  textual y contexto jurisprudencial". Arquitectura:
  1. Núcleo reutilizado de `scripts/lib/componentes_producto.py` (M0, M4 exploratorio, M7, M5).
  2. Backend FastAPI (`ENTREGA FINAL/app/backend/`) con endpoints `/v1/validate`, `/v1/link`,
     `/v1/article/{n}`, `/v1/models`, `/v1/ask`, `/v1/similar`, `/v1/eval/summary`.
  3. Agente LangChain con tool-calling (no RAG estricto) y guardrails de 3 capas (ADR-23).
  4. Frontend Streamlit (`ENTREGA FINAL/app/frontend/`) separado, que consume la API por HTTP.
  5. LLM local vía ollama, por defecto `qwen3.8:diario-8k-q3_k_s`, seleccionable en la UI.
  6. Entrega consolidada E1+E2+E3 en `ENTREGA FINAL/` sin duplicar datos pesados.
- **Alternatives Considered:**
  - Streamlit monolítico (rechazado: sin API ni separación para despliegue).
  - RAG estricto (rechazado: rechaza saludos y preguntas conversacionales; ver ADR-23).
  - M4 como feature principal (rechazado: etiquetas no fiables, ADR-19).
- **Consequences:**
  - (+) API testeable y reutilizable; frontend y backend desplegables por separado; el núcleo
    validado no se reescribe.
  - (-) Dos procesos y latencia del LLM; requiere chequeo de GPU antes de usar (sección 11 del plan).
  - Artefactos: `ENTREGA FINAL/app/`, `DESIGN.md`, ADR-23, `tasks/`.
