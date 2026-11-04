# Conjunto reservado

Creado el 2 de noviembre de 2026 para una evaluacion reservada. En la practica se uso despues para elegir la recuperacion hibrida y para evaluar la compuerta v2, asi que sus resultados son de desarrollo y no una prueba independiente.

## Composicion

| Corte | Total | Directas | Sin respuesta |
|---|---|---|---|
| Reservado | 25 | 20 | 5 |

- 15 preguntas en espanol y 10 en ingles.
- 14 de prensa, 6 de Bluesky y 5 sin respuesta.
- 20 documentos de referencia, todos verificados en la coleccion `rag_full_bge_m3_d1024_v1`.

## Construccion

1. Se muestrearon 45 documentos del parquet, excluyendo todos los `reference_doc_ids` del silver original: 30 de prensa (20 es, 10 en) y 15 de Bluesky (10 en, 5 es).
2. Para cada documento se genero una pregunta con `gemma4:e4b-mtp7` (temperatura 0, sin razonamiento) y se exigio una cita literal.
3. Se validaron automaticamente dos condiciones: la cita es subcadena del documento y la respuesta aparece dentro de la cita.
4. Se descartaron preguntas meta, ambiguas o con respuestas de mas de 8 palabras. Las que quedaron se reescribieron al idioma del documento y se acortaron las respuestas.
5. Las 5 preguntas sin respuesta usan entidades o eventos ausentes del corpus: `Puerto Inexistente`, `Nuevo Bogota`, el Mundial de 2038 y `zorblaxium`. Se comprobo que no aparecen en el parquet.

## Congelacion

- SHA256 del silver: `87fda373a4d40691b8ee766f71cbc41bfdef73683732d27a9456896a6dc3433d`
- SHA256 de la configuracion: `b6d6375ab000cff1853ff96d206ea6054d110c33b0b6f04eb5daa070a1a1b608`
- Configuracion congelada al crear el conjunto (v1): {"collection": "rag_full_bge_m3_d1024_v1", "gate": "logistic", "gate_threshold": 0.5, "k_answer": 5, "k_retrieval": 10, "model": "gemma4:e4b-mtp7", "prompt": "v2", "rerank_top": 5, "reranker": "BAAI/bge-reranker-v2-m3"}. La compuerta v2 se evaluo despues con umbral 0,4 y la recuperacion hibrida se eligio con este mismo conjunto.
- Regla inicial: ejecutarlo una sola vez. No se cumplio: el conjunto se uso despues para elegir la recuperacion hibrida y para evaluar la compuerta v2. Se conserva como registro y sus cifras se reportan como desarrollo.

## Uso

```bash
python -m src.scripts.eval_configuracion_final \
  --silver eval/silver/silver_reservado.jsonl \
  --raw reports/corridas/reservado_raw.jsonl \
  --summary reports/tablas/reservado_resumen.csv \
  --by-lang reports/tablas/reservado_por_idioma.csv \
  --figures-dir /tmp/reservado_figs
```
