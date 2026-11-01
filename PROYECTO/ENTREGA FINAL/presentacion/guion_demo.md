# Guion de la demo (Entrega 3)

Ensayo ejecutado el 2026-10-31 23:54 contra `http://127.0.0.1:8123` con el modelo `qwen3.8:diario-8k-q3_k_s`.

## Antes de empezar

1. Verificar la GPU: `rocm-smi --showmeminfo vram` (se necesitan 14 GB libres).
2. Levantar el backend: `uvicorn main:app --host 127.0.0.1 --port 8123`.
3. Levantar la interfaz: `streamlit run streamlit_app.py --server.port 8502`.
4. Abrir `http://127.0.0.1:8502` y dejar la barra lateral visible.

## Recorrido medido

| Paso | Latencia | Resultado |
|---|---:|---|
| Salud del servicio | 87 ms |  |
| Validación M0 | 4 ms | es_sentencia=True canon=T-293-1997 |
| Enlace sentencia 985 | 1189 ms | canon=T-225-1993 extraídos=12 recuperados=10 citadas=3 |
| Enlace sentencia 2195 | 229 ms | canon=T-293-1997 extraídos=6 recuperados=10 citadas=0 |
| Evidencia del artículo 86 | 153 ms | 86: 3 fragmentos, 13: 3 fragmentos |
| Jurisprudencia del artículo 86 | 16 ms | total=16405 top=C-543-1992 |
| Sentencias similares a 2195 | 130 ms | vecinas=T-826-1999, T-451-1998, T-580-1997 |
| Resumen de evaluación | 4 ms | comparativa y guardrails disponibles |
| Chat: saludo | 4 ms | modo=small_talk intención=saludo |
| Chat: fuera de alcance | 7 ms | modo=fuera_alcance intención=fuera_alcance |

## Cifras de guardrails para decir en voz alta

- Ruteo exacto de intencion: 90,0 %
- Saludos no rechazados: 100,0 %
- Rechazo correcto fuera de alcance: 100,0 %
- Adversarios contenidos: 100,0 %
- Tool call en consultas de alcance: 100,0 %
- Grounding verificado (global): 100,0 %
- Latencia del agente: p50 13,4 s, p95 60,8 s

## Pendiente

El chat en alcance no se midió en este ensayo: la GPU estaba ocupada
por otro proyecto. Repetir `python ensayo_demo.py` con 14 GB libres
antes de la demo.

## Orden de la presentacion

1. Portada y problema (2 min): 23.742 sentencias, 261 M palabras, la tarea manual.
2. Datos y método (2 min): corpus, MTLD, benchmark de 90 casos y gold humano.
3. Resultados (2 min): 0,6013 de la línea base a 0,7852 con gold humano.
4. Producto (3 min): P1 con el id 985, P2 con el articulo 86, P3 y chat.
5. Cierre (1 min): límites, trabajo futuro y la frase de cierre.

## Contingencia

- Si el LLM tarda o la GPU está ocupada, el agente responde con el plan
  determinista y evidencia verificable; el modo aparece en la interfaz.
- Si el backend no responde, revisar `/health` y reiniciar uvicorn.
- Usar el id 985 para la demo con citas: el 2195 solo tiene autocitas.

