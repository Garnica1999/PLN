# ADR-23: Guardrails del agente (tool-calling, no RAG estricto)

- **Status:** Aceptado (2026-10-31)
- **Date:** 2026-10-31
- **Context:** El asistente debe responder preguntas jurídicas usando la base (sentencias y CP),
  no inventar, y no responder fuera de contexto. Pero tampoco debe rechazar un saludo o una
  pregunta conversacional ("Hola, ¿cómo estás?") con "esa pregunta no está en la base". Se pide
  implementarlo como tool calling, no como RAG estricto.
- **Decision:** Tres capas:
  1. **Router de intención** (antes del modelo): `saludo/small_talk` → respuesta natural breve;
     `fuera_de_alcance` → mensaje de alcance (sin decir "no está en la base"); `en_alcance` →
     agente; `ambiguo` → aclaración.
  2. **Agente con herramientas** (LangChain): `validar_sentencia`, `vincular_cp`, `contexto_citas`,
     `explorar_articulo`, `sentencias_similares`. El modelo no responde de memoria: debe llamar
     herramientas; si no hay evidencia, responde "no encontré evidencia suficiente".
  3. **Verificación de grounding** (post-proceso): cada cita (id o artículo) de la respuesta se
     valida contra la salida de las herramientas; si no existe, se elimina o se responde sin
     evidencia. Se registra la traza de tool calls.
  - Si el modelo elegido no soporta tool calling, el backend usa el router determinista y el LLM
    solo redacta.
- **Evidence / evaluación:** gold set de ~100 prompts etiquetados por ruta (en alcance, saludo,
  small talk, fuera de alcance, adversario). Métricas: exactitud de ruteo, tasa de rechazo
  correcto, tasa de saludos respondidos, groundedness, tool-call success, latencia p50/p95.
  Baseline: RAG estricto (falla saludos) contra tool-calling + guardrails.
- **Alternatives Considered:**
  - RAG estricto con umbral de similitud (rechaza saludos y preguntas conversacionales).
  - Solo LLM sin herramientas (riesgo de alucinación).
- **Consequences:**
  - (+) Respuestas conversacionales correctas y respuestas jurídicas ancladas a evidencia.
  - (-) Más piezas que probar; la latencia del agente depende del modelo y de la GPU.
  - Artefactos: `ENTREGA FINAL/app/backend/agent.py`, `tools.py`, tests del agente.
