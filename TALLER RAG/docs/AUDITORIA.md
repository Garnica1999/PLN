# Auditoría del software final

Fecha: 2026-11-01. Alcance: `src/rag`, `src/scripts` y `src/tests`. Carga asumida: un taller académico con una demo y una consulta a la vez, 618.286 chunks y cuatro modelos locales. No es un servicio multiusuario.

## Must fix

**1. La interfaz se rompe con un saludo** (`src/rag/ui.py:75-80`, `src/rag/chat.py:230`)
- **Qué es:** la interfaz llama a `/chat` y lee `prompt_version`, `prompt_hash` y `context_tokens`.
- **Problema:** el camino de charla devuelve un `base` sin esos campos. Un "Hola" lanza `KeyError` en la interfaz.
- **Arreglo:** añadir esos campos al `base` de `chat` con `None` o `0`, y usar `.get` en la interfaz.
- **Si no:** la demo falla en el primer saludo.

**2. `/health` carga bge-m3 y puede marcar el contenedor como unhealthy** (`src/rag/api.py:57`, `src/rag/embeddings.py:104`)
- **Qué es:** el healthcheck llama a `get_embedder().health()`, que construye el modelo en la primera llamada.
- **Problema:** la carga tarda más que el timeout de 20 s y el contenedor queda unhealthy al arrancar. Ya ocurrió una vez.
- **Arreglo:** no cargar el modelo en `/health`; devolver el estado de configuración y comprobar el dispositivo sin inicializar pesos.
- **Si no:** el arranque parece roto aunque la API responda.

**3. El híbrido con una colección sparse ausente devuelve 500** (`src/rag/pipeline.py:search_hits`)
- **Qué es:** `RETRIEVER=hybrid` o `hybrid=true` llama a `retrieve_sparse`.
- **Problema:** si `SPARSE_COLLECTION` no existe, Milvus lanza `MilvusException`, que no es `ValueError` ni `LLMError`. El endpoint devuelve 500 sin mensaje útil.
- **Arreglo:** validar la colección sparse al arrancar o capturar la excepción y devolver 503 con el nombre de la colección.
- **Si no:** cambiar de recuperador rompe la API sin explicación.

## Should fix

**4. `hybrid_enabled` no se usa** (`src/rag/config.py:36`)
- **Qué es:** queda el flag viejo de la primera versión del híbrido.
- **Problema:** `search_hits` decide con `settings.retriever`, no con `hybrid_enabled`. Es código muerto que confunde la configuración.
- **Arreglo:** eliminarlo de `Settings` y de `load_settings`.

**5. El prefijo de qwen3 se aplica a cualquier modelo remoto** (`src/rag/embeddings.py:12`)
- **Qué es:** `OllamaEmbedder` antepone `Instruct: ... Query:` a toda consulta.
- **Problema:** el prefijo es de qwen3-embedding. Si se usa otro modelo en el servidor, el texto de la consulta queda sesgado.
- **Arreglo:** aplicar el prefijo solo si el modelo contiene `qwen3-embedding`, o renombrar la clase a `Qwen3OllamaEmbedder`.

**6. El juez puede bloquear `/chat` cuatro minutos** (`src/rag/chat.py:judge`)
- **Qué es:** el juez usa `call_llm` con el timeout global de 240 s.
- **Problema:** si llama3.2:3b se cuelga, la consulta entera espera cuatro minutos.
- **Arreglo:** pasar un timeout propio de 30 s al juez y, si falla, abstenerse.

**7. Faltan tests de los caminos nuevos** (`src/tests`)
- **Qué es:** no hay tests de `/chat`, de `search_hits`, de `weighted_rrf` ni del embedder bge.
- **Problema:** el bug 1 y el 3 no los detecta la suite actual.
- **Arreglo:** tests unitarios con mocks y un test de integración marcado.

**8. El modelo por defecto de la interfaz sigue siendo qwen** (`src/rag/ui.py:24`)
- **Qué es:** si `/models` falla, la interfaz propone `qwen2.5:1.5b`.
- **Problema:** el sistema final usa `gemma2:2b`. El fallback induce a error.
- **Arreglo:** cambiar el valor a `gemma2:2b`.

**9. Directorio vacío `src/rag/{prompts}`** (`src/rag/{prompts}`)
- **Qué es:** un directorio creado por una expansión de llaves del shell.
- **Problema:** no se usa y confunde la estructura.
- **Arreglo:** eliminarlo.

## Nice to have

**10. Los scripts de benchmark duplican `load_chunks` y `weighted_rrf`** (`src/scripts/bench_bm25.py`, `bench_hybrid.py`, `bench_retrieval.py`)
- **Qué es:** tres copias de la misma carga y de la misma fusión.
- **Problema:** un cambio en la fusión hay que hacerlo en tres sitios.
- **Arreglo:** un helper `src/scripts/bench_common.py` con las dos funciones.

**11. `store.count` con `limit=None`** (`src/rag/store.py:113`)
- **Qué es:** cuenta con una agregación de Milvus.
- **Problema:** funciona para `count(*)`, pero depende del comportamiento del cliente y no está probado con colecciones vacías.
- **Arreglo:** añadir un test de colección vacía y de colección con datos.

## Veredicto

El software es sólido para la carga prevista: una demo y una consulta a la vez. Los tres must-fix son de integración, no de modelo: el saludo que rompe la interfaz, el healthcheck que carga el modelo y el híbrido sin colección sparse. Corregirlos es media hora.

## Lean

- 1 setting muerto (`hybrid_enabled`).
- 1 directorio vacío (`{prompts}`).
- 2 helpers duplicados en tres scripts.

## No revisado

- `src/scripts/ingest_variant.py`, `cleaning.py` y `chunking.py` a fondo.
- `annotate.py` y `export_gold_excel.py`.
- La interfaz en un navegador real; solo se comprobó por HTTP y con `AppTest`.
- No hay archivo de diapositivas en el repositorio; las figuras del informe son la referencia visual.


## Estado al 2026-11-01

Corregidos: 1 (campos de la UI en charla), 2 (health sin cargar el modelo), 3 (sparse inexistente), 4 (`hybrid_enabled` eliminado), 5 (prefijo qwen3 condicional), 6 (timeout del juez), 8 (modelo por defecto de la UI) y 9 (directorio vacío). El 7 quedó cubierto con 5 tests unitarios nuevos y 6 de integración.

Pendientes: 10 (helpers duplicados en los benchmarks) y 11 (test de `count` con colección vacía).

## Hallazgo posterior: el juez rechazaba respuestas correctas

**Qué pasaba.** El juez por defecto era `llama3.2:3b` con un prompt de una línea. En el caso "¿Y en qué departamento?" la recuperación traía el documento correcto en rank 1 y `gemma2:2b` respondía "En el departamento de Boyacá [1]". El juez devolvía `NO` y el sistema se abstenía con `juez_no`.

**Alcance.** Sobre las 43 preguntas del gold ampliado, `llama3.2:3b` aceptó 2 de 27 respuestas con cita válida. `qwen2.5:1.5b` y `gemma2:2b` aceptaron las 27. En 10 casos adversariales (negación, cuantificador, causal, atribución, contradicción y cruce de números), `gemma2:2b` atrapó 8 y `qwen2.5:1.5b` 5. No es un problema de un documento: es del modelo juez.

**Arreglo.** El generador es `gemma4:e4b`; `JUDGE_MODEL` y `CONDENSE_MODEL` pasan a `gemma2:2b`. `qwen2.5:1.5b` y `llama3.2:3b` dejan de ser necesarios en el camino conversacional. El veredicto se parsea con `^s[ií]\b`.

**Guardarraíl.** Se corrigieron los arranques de frase ("Hay"), los números con separador ("1.000" contra "1000"), las citas múltiples ("[2, 3, 4]"), los metadatos del chunk citado (título, fecha, URL) y la cobertura léxica cuando el chunk está en otro idioma. Las abstenciones directas del gold bajaron de 12 a 5 y luego a 1 con el reranker; las 5 preguntas sin respuesta se siguen absteniendo.

**Límite.** Quedan 5 abstenciones directas (g-017, g-018, g-023, g-028 y g-040). Salvo g-017, que es una respuesta incorrecta, las otras son respuestas correctas cuya cita apunta a otro chunk. El recuperador no es el problema: 36 de 38 documentos de referencia están en top-5. El juez verifica respaldo en la evidencia citada, no corrección contra el gold; si la recuperación trae el documento equivocado, acepta una respuesta sustentada por ese documento.


## Estado actual de la arquitectura

La arquitectura final usa `gemma4:e4b` con MTP, `k=5`, reranker `BAAI/bge-reranker-v2-m3` en CPU, compuerta de dominio con clasificador y juez/condensador `gemma2:2b`.

| Frente | Resultado |
|---|---|
| Recuperación densa k=10 | 0,912 de Recall@10 y 0,818 de nDCG |
| Reranker top-5 | nDCG 0,865; P@1 0,868; recall 0,895; P@5 0,947 |
| Respuesta | 37 de 38; 1 abstención directa; 5 de 5 negativas; 89,2% manual |
| Chat | 12 de 12 con Modo estricto |
| RAGAS actual | 0,925 / 0,848 / 0,640 / 0,921 / 0,677 |
| Latencia media | 6.717 ms sin razonamiento; 6.400 ms con razonamiento |

La arquitectura completa, las capas y la comparación contra la versión anterior están en `docs/ARQUITECTURA.md`.
