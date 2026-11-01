# Guion de la demo (Entrega 3)

Ensayo ejecutado el 2026-11-01 00:17 contra `http://127.0.0.1:8123` con el modelo `qwen3.8:diario-8k-q3_k_s`.

## Antes de empezar

1. Verificar la GPU: `rocm-smi --showmeminfo vram` (se necesitan 14 GB libres).
2. Levantar el backend: `uvicorn main:app --host 127.0.0.1 --port 8123`.
3. Levantar la interfaz: `streamlit run streamlit_app.py --server.port 8502`.
4. Abrir `http://127.0.0.1:8502` y dejar la barra lateral visible.

La primera consulta en alcance carga el modelo (unos 20 s); la segunda
responde en unos 9 s. Conviene hacer una consulta de calentamiento antes
de la demo.

## Recorrido medido

| Paso | Latencia | Resultado |
|---|---:|---|
| Salud del servicio | 76 ms |  |
| Validación M0 | 595 ms | es_sentencia=True canon=T-293-1997 |
| Enlace sentencia 985 | 4015 ms | canon=T-225-1993 extraídos=12 recuperados=10 citadas=3 |
| Enlace sentencia 2195 | 241 ms | canon=T-293-1997 extraídos=6 recuperados=10 citadas=0 |
| Evidencia del artículo 86 | 151 ms | 86: 3 fragmentos, 13: 3 fragmentos |
| Jurisprudencia del artículo 86 | 168 ms | total=16405 top=C-543-1992 |
| Sentencias similares a 2195 | 896 ms | vecinas=T-826-1999, T-451-1998, T-580-1997 |
| Resumen de evaluación | 4 ms | comparativa y guardrails disponibles |
| Chat: saludo | 4 ms | modo=small_talk intención=saludo grounding=ok |
| Chat: fuera de alcance | 7 ms | modo=fuera_alcance intención=fuera_alcance grounding=ok |
| Chat: consulta en alcance | 19318 ms | modo=agente intención=en_alcance tools=vincular_cp grounding=ok |
| Chat: segunda consulta en alcance | 9173 ms | modo=agente intención=en_alcance tools=contexto_citas grounding=ok |

## Respuesta real del agente (consulta en alcance)

> La sentencia T-293/97 (id 2195) cita explícitamente los artículos 11, 13, 25, 49, 86 y 241 de la CP. - **11 y 13**: bloque de derechos fundamentales (vida e igualdad). - **25**: derechos fundamentales (mínimo vital). - **49**: bloque de derechos sociales, económicos y culturales. - **86**: protección y aplicación de derechos (acción de tutela). - **241**: rama judicial (guarda de la integridad constitucional). Además, el sistema recuperó por similitud los artículos 1, 9, 2 y 53, aunque no aparecen citados directamente en el texto. ¿Quieres que explore el texto de alguno de estos artículos?

Traza: vincular_cp (ok, 231 ms). Grounding: verificado.

## Cifras de guardrails para decir en voz alta

- Ruteo exacto de intención: 90,0 %
- Saludos no rechazados: 100,0 %
- Rechazo correcto fuera de alcance: 100,0 %
- Adversarios contenidos: 100,0 %
- Tool call en consultas de alcance: 100,0 %
- Grounding verificado (global): 100,0 %
- Latencia del agente: p50 13,4 s, p95 60,8 s

## Orden de la presentación

1. Portada y problema (2 min): 23.742 sentencias, 261 M palabras, la tarea manual.
2. Datos y método (2 min): corpus, MTLD, benchmark de 90 casos y gold humano.
3. Resultados (2 min): 0,6013 de la línea base a 0,7852 con gold humano.
4. Producto (3 min): P1 con el id 985, P2 con el artículo 86, P3 y chat.
5. Cierre (1 min): límites, trabajo futuro y la frase de cierre.

## Contingencia

- Si el LLM tarda o la GPU está ocupada, el agente responde con el plan
  determinista y evidencia verificable; el modo aparece en la interfaz.
- Si el backend no responde, revisar `/health` y reiniciar uvicorn.
- Usar el id 985 para la demo con citas: el 2195 solo tiene autocitas.

