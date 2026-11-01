# TODO — Entrega Final (actualizado 2026-10-31)

Leyenda: [x] hecho, [~] en curso, [ ] pendiente.

## F0 — Fundaciones
- [x] Deps: langchain 1.4.7, langchain-ollama 1.1.0, langgraph 1.2.19, ollama 0.6.3, python-pptx 1.0.2
- [x] Estructura `ENTREGA FINAL/` (app/backend, app/frontend, informe, presentacion, codigo, evaluacion, datos, tasks)
- [x] GPU verificada (13,9 GB libres, sin procesos KFD, ollama sin modelos cargados)
- [x] `DESIGN.md` (dirección derivada del deck del EDA)
- [x] ADR-22 (arquitectura final) y ADR-23 (guardrails)
- [x] Backend FastAPI (/health, /v1/models, /v1/validate, /v1/link, /v1/ask) + tests (29 passed, 1 skipped con la extensión)
- [x] Agente LangChain con tools + guardrails (smoke: saludo small_talk, fuera de alcance, in-scope con vincular_cp y grounding verificado; 21,7 s)
- [x] Frontend Streamlit separado (4 pantallas, selector de modelo, panel M4 exploratorio; 20 tests, AppTest contra backend real)
- [x] Smoke E2E backend + frontend (AppTest con T-293-1997 y métricas reales; HTTP 200 en 8502)
- [x] Extensión de API: /v1/link con área/citadas/resumen; /v1/article/{n}/jurisprudencia (9 ms cacheado); /v1/evidencia (144 ms)

## F1 — Núcleo
- [x] P1 completa: evidencia por artículo (15 fragmentos en el caso 2195), contexto de citas, área exploratoria
- [x] P2: artículo → jurisprudencia (total 16.405 para el art. 86, 30 años, top con PageRank, filtro T/C/SU)
- [x] Gold set del agente (100 prompts) evaluado; v1 en `evaluacion/agente_eval_v1.*`
- [x] Router corregido (saludos, identidad, referencial vs fuera de alcance) y plan determinista cuando el modelo no recupera evidencia; 100/100 en el router puro
- [x] Gold set v3 ejecutado y `metricas.json` actualizado (ruteo 90 %, saludos 100 %, rechazo 100 %, tool-call 100 %, grounding 100 %)

## F2 — Adicionales
- [x] P3 evaluación/transparencia + comparador de rutas y guardrails (desde `/v1/eval/summary`)
- [x] P4 ficha PDF de la sentencia (reportlab, verificada con pypdf)
- [x] P5 sentencias similares (`/v1/similar/{id}`, TF-IDF cacheado, 95 ms en caliente)

## F3 — Validación y consolidación
- [x] Evaluación del pipeline (ruteo, rechazo, saludos, grounding, latencia)
- [x] pytest de API (45 backend + 30 frontend) y ruff limpio
- [x] Consolidación E1+E2 en `ENTREGA FINAL/` con enlaces a datos pesados
- [x] Informe PDF completo (33 páginas, 14 tablas, 13 figuras) en `informe/`
- [x] Deck storytelling de 12 diapositivas en `presentacion/` (PPTX y PDF, 12 páginas)
- [x] Documentación actualizada y Delivery Gate de antislop aplicado (sin guiones largos, sin emojis, sin relleno)

## F4 — Entrega
- [x] Ensayo de demo completo, incluido el chat con el LLM: 16,4 s el primero (carga del modelo) y 9,3 s el segundo, con grounding verificado
- [x] Empaquetado: `Entrega_3_Juris-PLN.zip` (53 MB, 379 archivos, sin caché ni enlaces)
- [x] Zip verificado (`unzip -t` sin errores) y commit local del trabajo
- [ ] Subida a la plataforma antes de 2026-11-09 23:59
