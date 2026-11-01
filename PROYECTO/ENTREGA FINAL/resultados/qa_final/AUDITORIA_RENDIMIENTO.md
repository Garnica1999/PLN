# AUDITORIA DE RENDIMIENTO - Producto final Juris-PLN (10_evaluar_sentencia.py)

Fecha: 2026-09-09
Modo: **Deep acotado** (perfilado real de UNA ejecucion por caso, sin re-entrenar).
Instrumento: `scripts/checks_perf_producto.py` (nuevo; NO modifica el producto: importa las clases de `lib/componentes_producto.py` y replica el orden exacto de `construir_salida()`). Coherencia de la replica verificada: top-1 M4 y ranking M7 identicos al metodo publico en ambos casos.

Casos: id 6251 (T/01, mediano: 38.801 chars, 13 citas, QA 8.11 s) y id 20178 (C/13, largo: 159.022 chars, 51 citas, QA 11.37 s).

Fuentes de medicion (todas reales, ninguna estimada):
- `time.perf_counter` por componente (script de checks, JSON en `resultados/qa_perf/perf_<id>.json`).
- `cProfile` de `construir_salida()` (caso 6251, `.prof` en `resultados/qa_perf/`).
- RSS: `ru_maxrss` + `/proc/self/status` muestreado cada 50 ms por fase.
- CPU/hilos: `/proc/self/stat` y `/proc/self/status` muestreados cada 50 ms por fase.
- GPU: `torch.cuda.memory_allocated()` antes/despues de inferir + cProfile (reparto de funciones).

Entorno: Python 3.14.7 (/home/carlos/venv), 32 hilos logicos, 31 GB RAM, GPU AMD RX 9070 XT (torch 2.14.0+rocm7.2 presente en el venv).

## Scorecard

| Metrica | Valor | Fuente | Estado |
|---|---|---|---|
| VRAM usada en inferencia | 0 bytes (torch.cuda.memory_allocated antes/despues = 0) | torch.cuda | OK (CPU-only) |
| Tiempo total sentencia (frio, 1 proceso) | 8.7 s (6251) / 12.7 s (20178) | perf_counter (checks) | OK (consistente con QA 8.11/11.37 s) |
| Tiempo inferencia en caliente (3a pasada) | 1.18 s (6251) / 4.99 s (20178) | perf_counter (checks) | OK |
| RAM pico del proceso (RSS) | 8.1 GB (6251) / 8.0 GB (20178) | ru_maxrss + /proc | A mejorar (ver R1) |
| Uso CPU en inferencia | ~1 nucleo (99.8-101.5% prom.) | /proc/self/stat | OK (esperado: secuencial) |
| Uso CPU en carga | ~5.3 nucleos prom., pico 29.7 | /proc/self/stat | OK |
| Hilos del proceso (pico) | 55-57 (mayoria pools BLAS/OpenMP inactivos) | /proc/self/status | OK |

## Tabla de tiempos por componente (caso mediano vs largo)

Ejecucion fria = una invocacion del CLI tal como lo usa el QA (proceso nuevo). "Carga" ocurre una vez por proceso; "inferencia" es la 1a pasada (incluye la carga lazy de spaCy que en el CLI ocurre dentro de `clasificar()`).

| Fase / componente | 6251 (38.8k chars, 13 citas) | 20178 (159k chars, 51 citas) | Escalabilidad |
|---|---|---|---|
| Lectura corpus por id (`texto_por_id`) | 3.68 s | 3.84 s | O(corpus completo), constante por doc |
| Init M0+M2+M4+M7+M5 (modelos/indices) | 0.88 s | 0.91 s | constante |
| M0 validacion (heuristica) | 0.004 s | 0.018 s | O(L) |
| M2 extraccion regex (arts, citas, ponente) | 0.003 s | 0.012 s | O(L) |
| M4 carga lazy spaCy (`es_core_news_md`) | 2.88 s | 2.80 s | constante |
| M4 lematizacion spaCy | 0.47 s | 1.82 s | O(L) lineal (4.1x texto -> 3.9x tiempo) |
| M4 mascara + TF-IDF + LogReg | 0.010 s | 0.038 s | O(L) |
| M7 tokenizacion query (chunks+bigramas) | 0.004 s | 0.015 s | O(L) |
| M7 scoring BM25 (rank_bm25) | 0.84 s | 3.20 s | O(L) lineal (60 -> 252 chunks) |
| M7 pooling + ranking | 0.001 s | 0.001 s | O(P), P ~ 1.000 pasajes |
| M5 resolver citas al corpus | <0.001 s | <0.001 s | O(1) por cita (dict lookup) |
| **Total frio** | **8.7 s** | **12.7 s** | |
| Inferencia en caliente (3a pasada) | 1.18 s | 4.99 s | |

## Cuello de botella

**Uso real del CLI (frio): lectura del corpus completo (3.7-3.8 s, ~42% del total) + carga lazy de spaCy (2.8-2.9 s, ~33%) = ~75% del tiempo por sentencia.** `texto_por_id` descomprime y materializa en RAM las 23.742 sentencias con texto completo (~5.5 GB) para extraer UNA fila; spaCy se carga dentro de la primera llamada a `clasificar()`.

**Inferencia en caliente / escalabilidad: BM25 scoring (40.5% del caso largo) + lematizacion spaCy (23.0%)**, ambos lineales con la longitud del documento. El coste NO escala con el numero de citas: M5 resolvio 51 citas en <1 ms (lookups sobre dict) y M2 extrajo 51 citas + 25 arts en 12 ms.

Evidencia cProfile (6251): `rank_bm25.get_scores` acumula 1.79 s de 2.21 s (60 llamadas, 7.1M `dict.get` — implementacion Python puro); spaCy/thinc (tok2vec + tagger para lemas) 0.36 s. Todo el computo ocurre en numpy/thinc/rank_bm25 en CPU.

## Verificacion de recursos

- **CPU-only confirmado.** `torch` (2.14.0+rocm) aparece en `sys.modules` unicamente porque thinc, el backend de spaCy, lo importa como dependencia; la GPU AMD es visible para torch, pero `torch.cuda.memory_allocated() == 0` bytes antes y despues de inferir, y el cProfile no muestra ninguna funcion CUDA/hip. El producto no importa torch directamente ni ejecuta nada en GPU.
- RAM pico 8.1 GB, dominada por `pd.read_parquet` del corpus completo (fase carga: RSS pico 5.8 GB; la inferencia añade ~1.3 GB de estructuras spaCy/BM25). Cabe holgado en los 31 GB del equipo, pero es 8x el estrictamente necesario para una sentencia.
- Inferencia efectivamente secuencial (~1 nucleo); la carga usa ~5 nucleos (descompresion parquet multihilo).

## Recomendaciones (priorizadas por impacto/esfuerzo)

1. **[ALTO impacto / BAJO esfuerzo] Leer solo la fila pedida del corpus.** `texto_por_id` (10_evaluar_sentencia.py:86-89) lee el parquet completo con pandas. Sustituir por `pyarrow.parquet.read_table(..., filters=[("id","==",int(id))])` (o un indice ligero id -> fila guardado en data/). Impacto potencial (proyeccion, no medido): -3.7 s por invocacion (-42% del total frio) y RSS pico de ~8 GB a <2 GB. Riesgo nulo sobre resultados (mismos textos).
2. **[ALTO impacto / BAJO esfuerzo] Modo lote / proceso persistente.** Cada invocacion del CLI repaga carga de modelos + spaCy (~4.6 s). Un flag `--lote ids.txt` que inicialice componentes una vez y evalue N sentencias (o un servicio que mantenga el proceso vivo) amortiza esa carga: el QA de 10 casos pasaria de ~90 s a ~25 s (proyeccion a partir de las 3a pasadas calientes medidas: 1.2-5.0 s/caso). No cambia ninguna metrica.
3. **[ALTO impacto / MEDIO esfuerzo] Vectorizar el scoring BM25.** `rank_bm25.BM25Okapi.get_scores` es Python puro (7.1M `dict.get` en el caso mediano; 20-40% de la inferencia). Una implementacion con matrices dispersas de scipy (o paralelizacion de chunks) reduciria ~80% ese tiempo (impacto potencial, no medido). Como M7 v2 esta validado (F1@5 0.218), cualquier reemplazo exige re-ejecutar la validacion de 12_evaluar_vinculo_cp_v2.py y comparar scores identicos antes de integrarlo.

## Veredicto de rendimiento

**OK para entrega academica.** 8-12 s por sentencia en frio (1.2-5.0 s en caliente), CPU-only verificado, RAM pico 8.1 GB (31 GB disponibles), sin OOM ni degradacion, y consistente con los tiempos del QA final (8-11 s positivas). El cuello de botella es de inicializacion (lectura de corpus + carga de spaCy), no algoritmico, y la inferencia escala linealmente solo con la longitud del documento (no con el numero de citas). Las recomendaciones 1 y 2 son optimizaciones de ingenieria opcionales para uso por lotes; ninguna bloquea la entrega.
