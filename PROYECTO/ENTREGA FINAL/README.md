# Juris-PLN, entrega final

Producto de investigación jurídica sobre la jurisprudencia de la Corte
Constitucional de Colombia: dada una sentencia, recupera los artículos de la
Constitución que la fundamentan con evidencia textual, las sentencias que cita
y la jurisprudencia asociada a cada artículo.

Curso: Procesamiento de Lenguaje Natural, Maestría en Ingeniería de Sistemas y
Computación, Pontificia Universidad Javeriana.
Fecha límite de entrega: 2026-11-09.

## Qué hay en esta carpeta

| Carpeta | Contenido |
|---|---|
| `app/backend/` | API FastAPI con el núcleo del producto, el agente con guardrails y los tests |
| `app/frontend/` | Interfaz Streamlit separada (consume la API por HTTP) |
| `evaluacion/` | Gold set del agente (100 prompts), script de evaluación y resultados v1, v2 y v3 |
| `informe/` | Informe consolidado de las tres entregas (DOCX y PDF) |
| `presentacion/` | Deck de 12 diapositivas (PPTX y PDF), guion de la presentación y guion de la demo |
| `codigo/` | Código de las entregas 1 y 2 (scripts, notebooks, planes, configuración) |
| `resultados/` | Métricas y reportes JSON/MD de los experimentos |
| `figuras/` | Figuras del EDA y de los experimentos |
| `datos/` | Enlaces a los datos y modelos, con inventario y regeneración |
| `tasks/` | Plan y tablero de la entrega |
| `DESIGN.md` | Dirección de diseño de la app y de las diapositivas |

## Cómo correr la demo

Requisitos: `/home/carlos/venv/bin/python` (Python 3.14), ollama en
`http://127.0.0.1:11434` y, para la ruta con reranker, GPU AMD con ROCm.

```bash
# 0. Dependencias del frontend (el backend usa el venv del proyecto)
cd "ENTREGA FINAL/app/frontend"
/home/carlos/venv/bin/pip install -r requirements.txt

# 1. Verificar la GPU antes de usarla (el LLM por defecto ocupa ~13,5 GB)
rocm-smi --showmeminfo vram
rocm-smi --showpids

# 2. Backend
cd "ENTREGA FINAL/app/backend"
/home/carlos/venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8123

# 3. Frontend (en otra terminal; el 8501 puede estar ocupado por otro proyecto)
cd "ENTREGA FINAL/app/frontend"
API_URL=http://127.0.0.1:8123 \
  /home/carlos/venv/bin/python -m streamlit run streamlit_app.py \
  --server.port 8502 --server.headless=true
```

Documentación de la API en `http://127.0.0.1:8123/docs`.
El backend solo acepta peticiones CORS de los orígenes locales del frontend
(8501 y 8502). Si la demo se sirve desde otra dirección, se declara en
`JURIS_CORS_ORIGINS` (lista separada por comas) y el backend se mantiene en
`127.0.0.1` salvo que se exponga a propósito.
Modelo por defecto del agente: `qwen3.8:diario-8k-q3_k_s` (seleccionable en la
barra lateral de la interfaz).

En la pantalla de consulta asistida hay un interruptor de razonamiento
(`thinking`). Activado, el modelo razona antes de responder y el pensamiento se
muestra en un panel aparte; la respuesta pasa de unos 10 s a unos 40 s en
caliente y consume más contexto (el modelo por defecto tiene 8k). Desactivado,
el sistema responde igual que antes. No todos los modelos soportan
razonamiento: si el elegido no lo soporta, el backend cae al plan determinista
y lo indica en la respuesta.

## Endpoints

| Método | Ruta | Uso |
|---|---|---|
| GET | `/health` | estado, artefactos y componentes cargados |
| GET | `/v1/models` | modelos de ollama y si soportan tool calling |
| POST | `/v1/validate` | validación de entrada (M0) |
| POST | `/v1/link` | sentencia a artículos de la CP, área exploratoria, citas y resumen |
| POST | `/v1/evidencia` | fragmentos de la sentencia que activan cada artículo |
| GET | `/v1/article/{n}` | texto del artículo constitucional |
| GET | `/v1/article/{n}/jurisprudencia` | sentencias que citan el artículo, con PageRank |
| GET | `/v1/similar/{id}` | sentencias similares por similitud de texto |
| POST | `/v1/ask` | agente con tool calling y guardrails; `pensar=true` habilita el razonamiento del modelo |
| GET | `/v1/eval/summary` | métricas con procedencia, comparativa de rutas y guardrails |

## Resultados principales

Recuperación sentencia a artículos de la Constitución (conjunto de test):

| Ruta | nDCG@10 | Fuente |
|---|---:|---|
| BM25 con prior de hubs (referencia) | 0,6013 | `metricas.json:ciclo_v3` |
| v2.2 (B2 + B1) | 0,6907 | `metricas.json:b2_retriever` |
| v2.2 con gold humano (B3, 820 pares) | 0,7852 | `metricas.json:b3_gold_humano` |
| C3 (deriva temporal) | 0,7472 | `metricas.json:c3_deriva_temporal` |

Agente y guardrails (gold set de 100 prompts, `evaluacion/agente_eval_v3.json`):

| Métrica | Valor |
|---|---:|
| Ruteo exacto de intención | 90,0 % |
| Ruta jurídica alcanzada en consultas en alcance | 100,0 % |
| Saludos no rechazados | 100,0 % |
| Rechazo correcto de temas fuera de alcance | 100,0 % |
| Adversarios contenidos | 100,0 % |
| Tool call en consultas en alcance | 100,0 % |
| Grounding de citas | 100,0 % |
| Latencia del agente p50 / p95 | 13,4 s / 60,8 s |

Latencia del resto del producto: enlace por defecto 0,19 s en caliente (CPU),
enlace con reranker 13 a 15 s (GPU), evidencia 144 ms, jurisprudencia por
artículo 9 ms (cacheada), sentencias similares 95 ms en caliente.

## Limitaciones declaradas

- El área temática (M4) es exploratoria: su macro-F1 en test es 0,2870 y las
  etiquetas de origen no son fiables (ADR-19). No se usa para decidir.
- El corpus cubre 1992 a 2021; las sentencias fuera de ese rango no se pueden
  analizar.
- No hubo validación por un experto legal. Las etiquetas humanas se produjeron
  con apoyo de LLM y se reportan como tales.
- El agente depende de un LLM local: con la GPU ocupada la latencia crece o la
  respuesta cae al plan determinista, que siempre cita evidencia verificable.
- El despliegue es local; la separación entre API y frontend deja preparado el
  despliegue externo, que no forma parte de esta entrega.

## Paquete de entrega

`PROYECTO/Entrega_3_Juris-PLN.zip` (53 MB, 379 archivos) contiene esta carpeta
completa sin la caché derivada ni los enlaces a datos pesados. Los datos y
modelos se regeneran con los scripts de `codigo/segunda_entrega/` y se
documentan en `datos/LEEME_DATOS.md`.

El ensayo de la demo se reproduce con:

```bash
cd "ENTREGA FINAL/presentacion"
/home/carlos/venv/bin/python ensayo_demo.py
```

Requiere 14 GB de VRAM libre para la parte del chat; con `--sin-llm` mide el
resto del recorrido.

## Pruebas

```bash
cd "ENTREGA FINAL/app/backend"   && /home/carlos/venv/bin/python -m pytest -q   # 45 passed, 1 skipped
cd "ENTREGA FINAL/app/frontend"  && /home/carlos/venv/bin/python -m pytest -q   # 30 passed, 1 skipped
```

La prueba de integración con el LLM se activa con `JURIS_LLM_TESTS=1` y exige
14 GB de VRAM libre.
