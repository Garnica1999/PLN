# Backend Juris-PLN (Entrega Final)

API FastAPI + agente LangChain sobre el producto ya validado de la SEGUNDA
ENTREGA. No duplica el nucleo: importa `lib/componentes_producto.py` y
`scripts/10_evaluar_sentencia.py` desde `PROYECTO/SEGUNDA ENTREGA/`.

## Requisitos

- Interprete: `/home/carlos/venv/bin/python` (Python 3.14).
- `ollama` en `http://127.0.0.1:11434` con el modelo por defecto
  `qwen3.8:diario-8k-q3_k_s`.
- GPU AMD (ROCm) solo para la ruta `rerank=True`; la ruta por defecto corre en
  CPU.

Antes de usar GPU u ollama, verificar la VRAM libre (el modelo por defecto
ocupa ~13,5 GB):

```bash
rocm-smi --showmeminfo vram
rocm-smi --showpids
```

## Ejecutar

```bash
cd "PROYECTO/ENTREGA FINAL/app/backend"
/home/carlos/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

- OpenAPI: `http://127.0.0.1:8000/docs`
- La carga de componentes es perezosa: la primera peticion paga la
  inicializacion (BM25, red de citas, spaCy/LogReg) y luego se reutiliza.

## Endpoints

| Metodo | Ruta | Descripcion |
|---|---|---|
| GET | `/health` | estado, artefactos presentes, componentes cargados, ollama |
| GET | `/v1/models` | modelos de ollama y `tool_calling` estimado |
| POST | `/v1/validate` | M0: `{"texto": ...}` o `{"id": "2195"}` |
| POST | `/v1/link` | M2+M7+M4+M5: `{"id": "T-293/97", "k": 10, "rerank": false}` |
| GET | `/v1/article/{numero}` | texto recortado y bloque tematico (ej. `/13`) |
| GET | `/v1/article/{numero}/jurisprudencia` | sentencias que citan el articulo (P2) |
| POST | `/v1/evidencia` | fragmentos de la sentencia por articulo (P1) |
| POST | `/v1/ask` | `{"pregunta": "...", "model": null, "rerank": false}` |
| GET | `/v1/eval/summary` | metricas clave con su procedencia exacta |

`id` acepta el id numerico del corpus (2195) o el canon (T-293/97). En
`/v1/ask`, `rerank=true` activa la ruta B2+B1 (~14 s por consulta) y `model`
elige cualquier modelo de ollama.

### `/v1/link`: campos nuevos (compatibles hacia atras)

- `area`: `{"top3": [{"area", "prob"}], "via", "exploratoria": true}`. El
  top-3 sale del clasificador M4; `exploratoria` marca que se entreno con
  etiquetas silver (ADR-19), no es una clasificacion verificada.
- `sentencias_citadas`: lista de `{canon, id, resuelta, ano, pagerank, area}`
  resuelta por M5 contra la red de citas, excluyendo la autocita del
  encabezado (self-loop).
- `resumen`: linea legible construida con `construir_resumen` del entry
  `10_evaluar_sentencia.py` (misma redaccion que el producto congelado).

### `/v1/article/{numero}/jurisprudencia` (P2)

Devuelve `{numero, total, por_ano, top: [{id, canon, ano, tipo, pagerank,
area_silver}], aviso}` con las sentencias cuyo bloque `arts_cp` (M2) incluye
el articulo. El indice invertido se construye una sola vez desde
`entidades_sentencias.parquet` + `red_citas_resumen.parquet` (carga perezosa,
~0,18 s en frio y ~9 ms en caliente); el ranking usa PageRank y `area_silver`
va rotulada como referencia automatica. Un articulo fuera de la CP devuelve
404.

### `/v1/evidencia` (P1)

Recibe `{"id": "2195", "articulos": ["86"], "rerank": false}` (o `texto` en
lugar de `id`) y devuelve por articulo los 2-3 chunks de la sentencia con
mayor score hacia ese articulo, reutilizando `_chunks_query` y
`_scores_bm25_caso` de M7 sin tocar el nucleo. Con `rerank=true` los
candidatos se repuntuan con el cross-encoder B1/bge (GPU).

## Guardrails de `/v1/ask`

1. **Router determinista**: saludo/small talk, fuera de alcance y mensajes
   ambiguos se responden sin LLM ni herramientas. El mensaje de fuera de
   alcance describe el dominio; nunca dice "no esta en la base".
2. **Agente con tool calling**: `validar_sentencia`, `vincular_cp`,
   `contexto_citas`, `explorar_articulo`. Si no hay evidencia, la respuesta
   fija es "No encontre evidencia suficiente en el corpus para responder."
3. **Grounding**: todo canon de sentencia o numero de articulo citado en la
   respuesta debe existir en la salida de las herramientas. Si no, se
   descarta la redaccion libre y se devuelve la evidencia cruda; el campo
   `grounding` y `advertencia` dejan constancia.

El campo `tool_calls` expone la traza (herramienta, argumentos, ms, resumen).

### Fallback documentado

Si el modelo no emite tool calls o el bind/invoke falla, el backend no
inventa: ejecuta un plan determinista de herramientas (canon/id/articulo
detectado en la pregunta) y usa el LLM solo para redactar sobre la evidencia.
La respuesta lo declara con `modo="fallback_sin_tools"` y la causa en
`advertencia`. Verificado el 2026-10-31: `qwen3.8:diario-8k-q3_k_s` si emite
tool calls aunque ollama no declare la capacidad `tools`.

## Pruebas

```bash
cd "PROYECTO/ENTREGA FINAL/app/backend"
/home/carlos/venv/bin/ruff check .
/home/carlos/venv/bin/pytest                 # sin LLM
JURIS_LLM_TESTS=1 /home/carlos/venv/bin/pytest -m integration
```

La prueba de integracion se salta si falta ollama, si `rocm-smi` no reporta
VRAM o si hay menos de 14 GB libres.

## Limitaciones

- El agente no tiene memoria entre peticiones ni acceso a texto completo de
  la Constitucion: `explorar_articulo` devuelve 150 caracteres.
- `contexto_citas` solo resuelve sentencias del corpus 1992-2021.
- El resumen de evaluacion lee JSON congelados; si falta una clave devuelve
  `null` en lugar de fabricar el valor.

`/v1/ask` acepta `pensar: true` para habilitar el razonamiento nativo del modelo: amplia `num_predict` a 2048 y devuelve el pensamiento en el campo `pensamiento`. Cuesta latencia y contexto; por defecto esta desactivado.
