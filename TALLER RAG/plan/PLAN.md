# Plan del Taller RAG con Milvus y LLM locales

Fecha: 2026-10-30
Estado: alcance aprobado (corpus completo al final, 3 a 4 semanas, equipo de 2 a 4).
Destino final: `TALLER RAG/plan/PLAN.md` dentro del repositorio.

## 1. Objetivo

Construir y evaluar un asistente RAG reproducible sobre el corpus de prensa y Bluesky, con Milvus como único motor vectorial, FastAPI, Streamlit y modelos locales servidos por Ollama. La pregunta de investigación es cuál modelo pequeño ofrece el mejor equilibrio entre exactitud factual, fidelidad a las fuentes, abstención correcta, latencia y recursos, con evidencia recuperada idéntica.

## 2. Alcance decidido

| Decisión | Valor |
|---|---|
| Corpus de trabajo | piloto estratificado de 2.000 documentos para desarrollar y comparar |
| Corpus final | 169.248 documentos, indexados con la configuración ganadora |
| Plazo | 3 a 4 semanas |
| Equipo | 2 a 4 personas, con dos anotadores para la evaluación humana |
| Modelos | los cuatro candidatos del PDF: qwen2.5:1.5b, llama3.2:1b, llama3.2:3b, gemma2:2b |
| LLM | local, Ollama Docker en el puerto 11435, GPU RX 9070 XT |
| Embeddings | producción en la API sobre CPU; servidor 192.168.1.100 con Ollama en Docker (11435) para la comparación E3 |
| Requisito del servidor | Docker Engine, Compose v2 y NVIDIA Container Toolkit, instalación única con sudo |
| Encoder de producción | intfloat/multilingual-e5-base en la API, CPU, 768 dimensiones, prefijos query/passage |
| Encoder de comparación | qwen3-embedding:8b en la GPU local, 4096 dimensiones truncadas a 1024 (E3); el servidor queda como alternativa |
| GPU local | AMD RX 9070 XT, 16 GB, gfx1201, solo generación |
| GPU servidor | NVIDIA GTX 1080, 8 GB, solo embeddings |
| Ollama local nativo | puerto 11434, queda libre como respaldo del LLM |
| Ollama Docker | puerto 11435 en el host, 11434 dentro del contenedor |
| API | puerto 8010 en el host, 8000 dentro del contenedor (8000 está ocupado por Graphiti) |
| Streamlit | puerto 8501 |
| Evaluación | 40 preguntas mínimo, 20 desarrollo y 20 prueba |
| Informe | PDF en `reports/informe/`, plantilla `tg_usta.cls`, español |

## 3. Arquitectura

```
Dataset/documentos_rag.parquet (solo lectura, fuera de Git)
        |
        v
ingesta -> chunking -> embeddings (servidor 192.168.1.100, Ollama 11434, GTX 1080)
        |                                    ^
        |                                    |
        v                                    |
   Milvus local (Docker)                     |
        |                                    |
Pregunta -> Streamlit 8501 -> FastAPI 8010 /ask -> recuperar top-k -> prompt
                                    |                                   |
                                    |                                   v
                                    +--> evidencias, tiempos, citas   Ollama local 11435
                                                                     (RX 9070 XT)
                                    |
                                    v
                    reports/corridas/*.jsonl -> evaluacion -> tablas y figuras
```

Capas y justificación, que va en `docs/ARQUITECTURA.md`:

- Ingesta: lee el Parquet en solo lectura, aplica chunking, deduplica y conserva offsets y procedencia.
- Segmentación: dos estrategias comparables, con IDs estables y trazabilidad documento a fragmento.
- Embedding: e5-base dentro de la API sobre CPU para producción, con prefijos `query: ` y `passage: `. El servidor con Ollama y qwen3-embedding:8b se conserva para la comparación E3 y como servicio alterno. La misma función codifica consultas y pasajes.
- Almacenamiento: Milvus standalone local con etcd y MinIO como infraestructura auxiliar, no como motores de búsqueda.
- Recuperación: filtros por colección, idioma y fecha, deduplicación por hash de texto y top-k configurable.
- Generación: Ollama local (contenedor, puerto 11435) en la RX 9070 XT, prompt versionado, presupuesto de contexto en tokens, citas verificadas y abstención.
- Interfaz: Streamlit local sobre la API, con selector de modelo, k, filtros, fuentes y tiempos.
- Evaluación: recuperación, generación, rendimiento, robustez y reproducibilidad, con resultados crudos en JSONL.

Puertos:

| Servicio | Dónde | Host | Contenedor | Motivo |
|---|---|---|---|---|
| Ollama del servidor | 192.168.1.100 | 11435 | 11434 | solo embeddings, en contenedor |
| Ollama local nativo | equipo local | 11434 | n/a | respaldo del LLM |
| Ollama local Docker | equipo local | 11435 | 11434 | LLM por defecto, sin choque |
| API | equipo local | 8010 | 8000 | 8000 lo ocupa Graphiti MCP |
| Streamlit | equipo local | 8501 | 8501 | libre |
| Milvus | equipo local | 19530 | 19530 | libre |
| Milvus métricas | equipo local | 9091 | 9091 | libre |
| etcd, MinIO | equipo local | sin exponer | internos | solo infraestructura |

## 4. Estructura del repositorio

```
TALLER RAG/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── rag/
│   │   ├── config.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── store.py
│   │   ├── prompts/
│   │   ├── pipeline.py
│   │   ├── api.py
│   │   ├── ui.py
│   │   └── annotate.py
│   ├── scripts/
│   │   ├── profile_corpus.py
│   │   ├── build_sample.py
│   │   ├── ingest.py
│   │   ├── neighbors.py
│   │   ├── make_gold_candidates.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   └── tests/
├── eval/
│   ├── gold/questions.jsonl
│   ├── gold/README.md
│   ├── pilot_docs.parquet
│   ├── annotations/
│   └── results/
├── reports/
│   ├── README.md
│   ├── articulo/
│   │   ├── articulo_ieee.tex
│   │   ├── secciones/
│   │   └── articulo_ieee.pdf
│   ├── informe/
│   │   ├── metricas_nueva_configuracion.md
│   │   └── tg_usta.cls
│   ├── figuras/
│   ├── tablas/
│   └── corridas/
├── tests/
│   ├── test_chunking.py
│   ├── test_citations.py
│   └── test_api.py
├── docs/
│   ├── ARQUITECTURA.md
│   ├── DECISIONES.md
│   ├── HALLAZGOS.md
│   ├── EXPERIMENTOS.md
│   └── GUIA_DEMO.md
└── plan/PLAN.md
```

`reports/README.md` es el índice: qué reporte existe, qué script lo produce y con qué fecha. Los JSONL de `reports/corridas/` y de `eval/results/` se versionan porque son la evidencia cruda; los pesos y el Parquet no.

Reglas de higiene:

- `Dataset/`, `*.parquet`, `.env` y los pesos van en `.gitignore`.
- El Parquet se monta en solo lectura en `/work/data`. No se copia ni se modifica.
- Un solo `docker-compose.yml` en la raíz.
- Límites declarados en el compose: Milvus 10 GB y 8 CPU, Ollama 12 GB y 8 CPU, API 6 GB y 8 CPU, frontend 1,5 GB y 2 CPU. Se documentan en el README y se miden con `docker stats`.
- Un solo `docker-compose.yml` para los dos equipos, con red de host porque dockerd tiene `iptables: false` y `bridge: none`. El stack local arranca con `docker compose up -d --build`. En el servidor, el mismo archivo arranca solo el servicio de embeddings con `docker compose up -d ollama-embeddings`, que vive en el perfil `embeddings` y no se levanta en local.
- Regla de puertos: el Ollama nativo de cada máquina conserva el 11434; todo Ollama en Docker se publica en el 11435. Por eso `EMBEDDINGS_URL=http://192.168.1.100:11435` y `OLLAMA_URL=http://ollama:11434` dentro de la red del compose.
- Ollama Docker local es la ruta por defecto (sección 9 del PDF). El nativo queda como respaldo cambiando `OLLAMA_URL` a `http://host.docker.internal:11434`, con `extra_hosts`.
- El informe y el README pasan por antislop antes de la entrega (modo during), igual que la interfaz.

## 5. Flujo de trabajo

La lógica es la que pidió Carlos: primero un piloto pequeño que permita comprobar que la recuperación funciona, recorrer los cuatro modelos del taller, elegir el mejor y solo entonces indexar el corpus completo con la configuración ganadora.

```
perfilado completo
      |
      v
muestra de humo 300 docs -> indice base -> COMPUERTA de recuperacion
      |                                          |
      |            si falla: corregir pipeline <--+
      v
piloto 2.000 docs -> dos chunkings -> encoder -> Milvus piloto -> metricas IR completas
      |
      v
API + Streamlit -> gold 40 -> 4 modelos -> metricas -> ganador
      |
      v
corpus completo con el ganador -> verificacion de escala -> informe y demo
```

## 6. Fases con compuerta

### Métricas de recuperación (transversal a F1.5, F5 y F6)

Las métricas deterministas sobre identificadores son la verdad primaria. eRAG y el juez LLM son complementarios y nunca sustituyen la referencia anotada.

| Métrica | Nivel | Definición | Dónde | Meta |
|---|---|---|---|---|
| Recall@k | documento | fracción de documentos relevantes recuperados en el top-k | humo, piloto, escala | humo 1,00@10 y 0,95@5; piloto 0,90@10 |
| P@1 y P@5 | documento | el relevante en el primer puesto y en el top 5 | piloto | se reporta |
| Hit@k | documento | 1 si al menos un relevante aparece en el top-k | humo, piloto | 1,00 en humo |
| MRR@k | documento | media de 1/rango del primer relevante | humo, piloto | humo 0,85@10 |
| MAP@k | documento | media de la precisión promedio | piloto | 0,75@10 |
| nDCG@k | documento | ganancia descontada por posición, relevancia binaria o graduada 0 a 2 | humo, piloto, ablaciones | humo 0,85@10; piloto 0,75@10 |
| Recall@k | fragmento | fracción de `reference_chunk_ids` recuperados | humo, piloto | 0,90@5 |
| Cobertura de respuesta | fragmento | el top-k contiene el `reference_span` exacto que responde | humo, piloto | 0,90 en top-5 |
| Context precision | contexto | fracción de chunks del contexto cuyo documento es relevante | F3 y F5 | 0,70 |
| Context recall | contexto | fracción de documentos de referencia que sobreviven al presupuesto de tokens | F3 y F5 | 0,90 |
| Tasa de duplicados | ranking | proporción de chunks repetidos en el top-k | humo | 0 |
| Documentos únicos | ranking | diversidad del ranking | humo, piloto | al menos min(k, 3) |
| Balance de fuentes e idiomas | ranking | proporción Bluesky/prensa y es/en en el top-k | piloto | se reporta, no se impone |
| Robustez de consulta | consulta | Recall@10 con paráfrasis, errores de tecleo y consulta cruzada de idioma | piloto | caída máxima de 10 puntos |
| eRAG | downstream | utilidad de cada documento recuperado, medida con el LLM y la respuesta de referencia | complementario en humo y prueba | se reporta |
| Latencia y costo | sistema | p50/p95 de recuperación y generación, chunks/s, RAM y VRAM | todas | p95 de recuperación menor a 2 s en el piloto |
| Intervalo de confianza | estadística | bootstrap de 1.000 remuestreos sobre las preguntas | todas | se reporta junto a la media |

Reglas de lectura:

- Con 20 preguntas los intervalos son anchos. Toda comparación entre configuraciones se reporta con IC del 95% por bootstrap y prueba pareada (bootstrap pareado o Wilcoxon), nunca solo con la media.
- Un Recall@10 de 1,00 en 20 preguntas no prueba calidad a escala. Prueba que el pipeline no está roto, que es justo lo que busca la compuerta de humo.
- `reference_span` y `reference_chunk_ids` se anotan a mano junto con el documento. Sin ellos no se pueden calcular las métricas de fragmento ni la cobertura de respuesta.
- eRAG se ejecuta sobre una submuestra (20 preguntas por k documentos) porque cuesta una llamada al LLM por documento.

Referencias de evaluación: BEIR (Thakur et al., 2021, arXiv:2104.08663), MTEB (Muennighoff et al., 2023, arXiv:2210.07316), MIRACL (Zhang et al., 2023, arXiv:2210.09984), RAGAS (Es et al., 2023, arXiv:2309.15217), ARES (Saad-Falcon et al., 2024, arXiv:2311.09476), eRAG (Salemi y Zamani, SIGIR 2024, arXiv:2404.13781), Lost in the Middle (Liu et al., 2024, arXiv:2307.03172) y RAGChecker (Ru et al., 2024, arXiv:2408.08067).

### F0. Entorno y esqueleto (día 1)

- Crear árbol, `.gitignore`, `.env.example`, `Dockerfile`, `docker-compose.yml`.
- Puertos: Milvus 19530 y 9091, Ollama Docker 11435, API 8010, Streamlit 8501.
- Servicio Ollama con `devices: ["/dev/kfd", "/dev/dri"]`, `group_add` para `video` y `render`, y volumen `ollama_data`.
- Descargar los cuatro modelos del PDF:

```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama pull llama3.2:1b
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull gemma2:2b
docker compose exec ollama ollama list
```

- Verificar que el contenedor usa la GPU: `docker compose exec ollama ollama run llama3.2:1b "hola"` y luego `docker compose exec ollama ollama ps`. Si dice 100% CPU, se cambia `OLLAMA_URL` al nativo y se registra como limitación.
- Configurar el servidor de embeddings en `192.168.1.100` con llave SSH, no con contraseña en archivos. Instalación única de Docker y del toolkit de NVIDIA (requiere sudo):

```bash
ssh carlos@192.168.1.100
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager addrepo --from-repofile=https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker carlos
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf -y install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

- Autenticarse de nuevo para que el grupo `docker` aplique y verificar `docker compose version`.
- Llevar el compose al servidor y levantar solo el servicio de embeddings:

```bash
scp docker-compose.yml carlos@192.168.1.100:~/taller-rag/
ssh carlos@192.168.1.100 'cd ~/taller-rag && docker compose up -d ollama-embeddings'
ssh carlos@192.168.1.100 'cd ~/taller-rag && docker compose exec ollama-embeddings ollama pull qwen3-embedding:8b'
ssh carlos@192.168.1.100 'cd ~/taller-rag && docker compose exec ollama-embeddings ollama pull qwen3-embedding:0.6b'
```

- Comprobar que el contenedor usa la GPU con una llamada a `/api/embed` contra `http://192.168.1.100:11435` y después `docker compose exec ollama-embeddings ollama ps`. El servidor tiene 98 GB libres, así que el espacio deja de ser límite. El Ollama nativo del servidor sigue en 11434 y no se toca.
- Medir desde el equipo local: latencia de red (p50 de 100 llamadas a `/api/tags`) y chunks por segundo con lotes de 32 textos de 512 tokens.
- Registrar ID, digest, cuantización y licencia de cada modelo en `docs/EXPERIMENTOS.md`.
- Compuerta: `docker compose up -d --build` deja todo sano en local, `/health` responde `ok` con Milvus y Ollama, los cuatro modelos aparecen en `ollama list`, el contenedor de embeddings del servidor devuelve 4096 dimensiones en GPU y el rendimiento queda medido.

### F1. Perfilado y muestra reproducible, fase A (día 1 a 2)

- `profile_corpus.py` produce `reports/tablas/perfilado.csv`, `reports/tablas/perfilado.json` y figuras en `reports/figuras/`: fuentes, longitudes, idiomas, duplicados, fechas, dominios, nulos y sesgos.
- Normalizar el idioma sin tocar el archivo: código ISO de dos letras y reporte de los 13.137 sin idioma.
- Verificar `text_sha256` contra el texto y contar duplicados por colección.
- `build_sample.py` genera el piloto estratificado de 2.000 documentos (semilla 42, estratos por colección y idioma) y guarda solo los `document_id` en `eval/pilot_docs.parquet`.
- Compuerta: cada afirmación del informe tiene figura o tabla y script; el piloto se reconstruye con la misma semilla.

### F1.5. Evaluación de humo de la recuperación (día 2, antes de escalar)

Objetivo: comprobar que el pipeline recupera la evidencia correcta antes de invertir horas en el piloto y en la comparación de modelos. Es una compuerta de correctitud, no una evaluación de calidad final.

- Conjunto de humo: 600 documentos, 350 de prensa y 250 de Bluesky, con español e inglés, muestreados con semilla fija.
- Al menos 40 preguntas de humo, una por documento, con `reference_doc_ids` y `reference_span` (el fragmento exacto que responde). Las preguntas se generan con el LLM desde el documento y se verifican contra el texto; la validación humana queda para el gold final.
- Tres ejecuciones con el mismo encoder y la misma k para comprobar determinismo.
- Métricas: Recall@5 y @10, Recall@5 de fragmento, cobertura de respuesta en top-5, MRR@10, nDCG@10, MAP@10, hit@10, tasa de duplicados, tasa de recuperaciones vacías y latencia p50/p95.
- Compuerta, todas obligatorias:
  - Recall@10 = 1,00 y Recall@5 mayor o igual a 0,95.
  - MRR@10 mayor o igual a 0,85 y nDCG@10 mayor o igual a 0,85.
  - El `reference_span` aparece en el top-5 en al menos el 90% de las preguntas.
  - Cero documentos duplicados en el top-10 y cero offsets rotos.
  - Top-10 idéntico en las tres ejecuciones.
  - Los filtros de idioma y fecha devuelven solo documentos que cumplen.
- Si falla, se corrige el pipeline (chunking, prefijos, normalización, índice o filtros) y se repite. No se escala con la compuerta en rojo.
- Cada fallo se registra en `docs/EXPERIMENTOS.md` con causa, corrección y evidencia.
- Esta compuerta no elige modelo ni configuración: solo dice si el recuperador está sano.

### F2. Piloto: segmentación, embeddings y Milvus, fases B, C y D (día 2 a 6)

- Estrategia `fixed_chars`: 1000 caracteres, 150 de solape.
- Estrategia `sentence_tokens`: máximo 384 tokens, solape de una oración, corte duro solo si una oración excede el límite.
- IDs estables: `sha256(document_id|estrategia|tamaño|solape|offset|sha256(texto))`.
- Deduplicación por hash de texto dentro del documento, con conteo registrado.
- Limpieza adoptada: V2 (filtro de páginas de error + NFKC, sin HTML ni URLs, espacios colapsados). Ver ADR-018.
- Cada chunk hereda `document_id`, `title`, `url`, `collection_name`, `source`, `language`, `created_at`, `position`, `end_offset` y `text_sha256`.
- Esquema Milvus: `chunk_id` (PK), `document_id`, `vector`, `text`, `title`, `url`, `collection_name`, `source`, `language`, `created_at`, `position`, `chunk_strategy`, `text_sha256`.
- Índice HNSW, `M=16`, `efConstruction=200`, COSINE, `ef=64` en búsqueda.
- Ingesta idempotente por `upsert`, con `--drop`, `--start`, `--limit` y verificación de conteo.
- Encoder qwen3-embedding:8b en el servidor por Ollama. Consulta con `Instruct: ...\nQuery: ...`, documentos sin prefijo, vector truncado a 1024 dimensiones y renormalizado L2. La truncación por MRL se valida en E3. Lotes de 32, `num_ctx=512`, tres reintentos con espera exponencial y `--start` para reanudar si el servidor se cae.
- Respaldo: qwen3-embedding:0.6b en el servidor, o e5-base en la API con prefijos `query: ` y `passage: `, siempre en colección aparte.
- Compuertas:
  - Conteos de Milvus iguales a los chunks generados.
  - Reconstrucción exacta: para 50 chunks al azar, el texto original en `[position, end_offset)` coincide con el chunk.
  - Persistencia: `docker compose restart milvus` y los conteos y la búsqueda siguen iguales.
  - Idempotencia: repetir la ingesta no aumenta el conteo.
  - Rendimiento medido en chunks por segundo, con ese número se decide el corpus final.

### F3. API y Streamlit, fases E y F (día 5 a 10)

- `pipeline.py` separa `retrieve()` y `generate()`, con prompt versionado y hash en cada respuesta.
- Presupuesto de contexto en tokens, 3000 por defecto, con el tokenizador del modelo.
- Citas `[n]` obligatorias, validación de índices citados y campo `invalid_citations`.
- Abstención sin llamar al LLM cuando no hay evidencia o cuando la mejor puntuación no supera el umbral calibrado en desarrollo.
- Respuesta con `answer`, `model`, `evidence`, `retrieval_ms`, `generation_ms`, `tokens_per_second`, `prompt_version`, `invalid_citations`.
- Endpoints `/health`, `/retrieve`, `/ask`, `/models`.
- Streamlit con selector de modelo (los cuatro), k, filtro de colección e idioma, pregunta, respuesta, fuentes con enlace y puntuación, y tiempos por etapa.
- Estados de carga, vacío y error visibles, no solo por color.
- Compuerta: los cuatro modelos aparecen en el selector, la misma pregunta con la misma k devuelve la misma evidencia, y ninguna respuesta sin evidencia se presenta como respuesta.

### F4. Silver standard (día 7 a 12, trabajo humano)

- 40 preguntas mínimo sobre documentos del piloto: 20 desarrollo y 20 prueba.
- Balance entre Bluesky y prensa, español e inglés, fechas de 2024 a 2026.
- Tipos: directas, comparativas, ambiguas y sin respuesta documental suficiente.
- Cada pregunta lleva `id`, `question`, `lang`, `type`, `split`, `reference_doc_ids`, `reference_chunk_ids`, `reference_span`, `reference_answer` y `criteria`.
- `make_gold_candidates.py` propone candidatos con `document_id` real y texto de respaldo; la validación y la redacción son humanas.
- Dos anotadores revisan una submuestra y se reporta el acuerdo.
- Compuerta: ninguna pregunta de prueba se usa para ajustar prompts o umbrales, y toda referencia existe en el piloto y en el corpus completo.
- Gold consolidado: `eval/gold/silver_standard.jsonl`, 43 preguntas (19 desarrollo, 24 prueba, 38 respondibles y 5 sin respuesta). Validado por Carlos; segunda revisión pendiente.

### F5. Comparación de los cuatro modelos y ablaciones (día 10 a 16)

| ID | Variable | Fijo | Métricas |
|---|---|---|---|
| E0 | humo: 300 documentos, 20 preguntas, 3 ejecuciones | configuración base | compuerta completa de F1.5 |
| E1 | chunking fijo vs oraciones | e5-base, k=10, limpieza V2, piloto | Recall@10, Precision@10, MRR@10, MAP@10, nDCG@10, Recall de fragmento, cobertura de respuesta, duplicados, latencia |
| E2 | k en {3,5,10} | mejor chunking | Recall@k, MRR, MAP@k, nDCG@k, context precision y recall, latencia |
| E3 | e5-base vs qwen3-embedding:8b en 1024 dimensiones, en la GPU local | mejor chunking, k óptimo, mismas preguntas | Recall@k, nDCG@k, latencia, costo e IC bootstrap |
| E4 | 4 LLM | evidencia idéntica, temperatura 0 | corrección, groundedness, citas, abstención, p50/p95, tokens/s, eRAG |
| E5 | robustez | ganador | injection, sin respuesta, fechas, discrepancias, paráfrasis y consulta cruzada |
| E6 | escala 2.000 vs 10.000 vs corpus completo | ganador | Recall@k, nDCG@k, latencia e IC |

Reglas:

- Tres repeticiones por condición y modelo; la primera se marca como arranque en frío.
- La anotación humana se hace sobre la repetición 1; las repeticiones 2 y 3 miden latencia y estabilidad de la respuesta.
- Semilla, versiones, digests, cuantización, hardware y fecha en cada corrida.
- Cuando una configuración no sea idéntica entre modelos (por ejemplo cuantización distinta), se escribe la amenaza a la validez en `docs/EXPERIMENTOS.md`.
- Toda comparación entre configuraciones se reporta con IC del 95% por bootstrap y prueba pareada, no solo con la media.
- La selección del ganador usa las 20 preguntas de prueba; la limitación de elegir y reportar sobre el mismo conjunto se declara en el informe.
- Resultados crudos en `reports/corridas/*.jsonl` y métricas en `reports/tablas/`. Nada se escribe a mano.

Hipótesis H1: llama3.2:3b tendrá mayor fidelidad y corrección que qwen2.5:1.5b y gemma2:2b con la misma evidencia, a costa de mayor latencia. Se refuta si un modelo de 1 a 2B iguala la fidelidad con latencia menor.

### F6. Escala: 10.000 y corpus completo con el ganador (día 15 a 20)

- Indexar 10.000 documentos estratificados con la configuración ganadora como punto intermedio de la curva de escala.
- Indexar los 169.248 documentos con la configuración ganadora, en la colección `rag_full_{encoder}_{chunking}_v1`.
- Si el rendimiento medido en el piloto no permite terminar en 12 horas, se indexan 50.000 y la desviación se documenta con el número real.
- E6: mismas 20 preguntas de prueba sobre 2.000, 10.000 y corpus completo, solo recuperación, para medir la degradación por escala.
- Verificar conteos, persistencia tras reinicio y búsqueda con filtros en cada nivel.
- Compuerta: el conteo coincide con lo planificado, las referencias del gold siguen apareciendo y la latencia de recuperación queda registrada en los tres niveles.

### F7. Anotación y análisis (día 16 a 24)

- Rúbrica 0 a 3 en corrección, fidelidad, citas y abstención.
- `annotate.py` como app de anotación que exporta CSV.
- Dos anotadores sobre las 80 respuestas de prueba; kappa reportado.
- LLM-as-a-judge solo como complemento, nunca como verdad.
- Análisis de errores con los cuatro casos que pide la guía: discrepancias entre modelos, aciertos sin evidencia, invenciones y abstenciones correctas.

### F8. Informe, README y demo (día 24 a 28)

- Informe en PDF en `reports/informe/` con la estructura de la guía: problema, hipótesis, alternativas, decisiones, diagrama propio, método, experimentos, resultados, errores, limitaciones, conclusiones y referencias.
- README desde cero: requisitos, instalación sin pago, cómo obtener el corpus con `open_corpus_co_es` (`dragons_rag_2026_10_19`) o cómo colocarlo en `Dataset/`, descarga de modelos, indexación, inferencia, pruebas, límites de CPU, RAM y GPU, y solución de problemas.
- `reports/README.md` con el índice de reportes, tablas y figuras.
- Demo: preguntar, ver evidencias, alternar los cuatro modelos, mostrar tiempos y explicar un error.
- Compuerta: un tercero sigue el README desde cero y llega a una respuesta con fuentes.

## 7. Cronograma

| Semana | Entregable |
|---|---|
| 1 | F0, F1 y F2 con el piloto indexado y el rendimiento medido |
| 2 | F3, F4 y arranque de F5 |
| 3 | F5 completo, F6 y anotación |
| 4 | F7, F8, informe y demo |

## 8. Riesgos

| Riesgo | Señal | Mitigación |
|---|---|---|
| Embeddings lentos o sin GPU | menos de 20 chunks/s en el piloto | qwen3-embedding:0.6b o e5-base en CPU local |
| ROCm dentro de Docker no soporta gfx1201 | `ollama ps` muestra 100% CPU | cambiar `OLLAMA_URL` al nativo y registrar la limitación |
| GTX 1080 es Pascal y las compilaciones CUDA 13 la dejan fuera | el contenedor no ve la GPU o `ollama ps` muestra CPU | fijar una imagen de Ollama con runner CUDA 12 o mover el encoder a la RX 9070 XT |
| Servidor sin Docker ni toolkit de NVIDIA | `docker: orden no encontrada` o `--gpus` falla | instalación única de F0; si falla el repo, Ollama nativo como respaldo |
| Disco del servidor lleno | menos de 10 GB libres | hoy hay 98 GB; solo viven los pesos del encoder |
| Servidor caído o LAN inestable | timeouts durante la ingesta | reintentos con espera exponencial, `--start` y e5-base local como respaldo |
| Ollama del servidor sin autenticación | el puerto 11435 responde a toda la LAN | no exponerlo a internet y documentarlo en el README |
| Ollama nativo del servidor ocupa 11434 | `bind: address already in use` | el contenedor publica 11435 y el nativo se queda intacto |
| Choque de puertos | `bind: address already in use` | 11435 para Ollama Docker y 8010 para la API |
| Duplicados inflan métricas | mismos textos en top-k | deduplicación por `text_sha256` y gold sin duplicados |
| Gold sesgado | preguntas concentradas en un dominio | muestreo estratificado y dos anotadores |
| Inyección de instrucciones | el modelo obedece texto del documento | delimitadores, instrucción explícita y colección de prueba separada |
| Fuga de datos a Git | `git status` muestra el Parquet | `.gitignore` desde el primer commit y revisión antes de publicar |
| Resultados inventados | tablas sin JSONL de respaldo | toda cifra se genera desde `eval/results` |

## 9. Cobertura de la guía

| Requisito del PDF | Dónde se cumple |
|---|---|
| Verificar esquema, nulos, IDs, fuentes, URLs, hash, sin alterar el archivo | F1 |
| Muestra reproducible 500 a 2.000, luego 10.000 y corpus completo | F1 y F6 |
| Preservar texto, offsets y procedencia por fragmento | F2, con prueba de reconstrucción |
| Milvus único motor; etcd y MinIO como infraestructura | ADR-003 |
| Documentar dónde vive el encoder y la codificación simétrica | ADR-002 y test de simetría |
| Arquitectura desacoplada en ocho capas justificada | `docs/ARQUITECTURA.md` |
| Compose único, Dockerfiles, `.env.example`, volúmenes, README | F0 y F8 |
| Todo el entorno en contenedores, en dos equipos, con un solo archivo | ADR-013 |
| PyArrow/Pandas y encoder abierto local | F2 |
| FastAPI con `/health`, `/retrieve`, `/ask` | F3 |
| Streamlit con modelo, k, fuentes y tiempos | F3 |
| Al menos tres modelos Ollama | F0 y F5 |
| A. Perfilado con evidencia cuantitativa | F1 |
| B. Dos chunkings con tokens, solape, offsets, IDs y dedup | F2 |
| C. Encoder multilingüe, dimensión, normalización, métrica, prefijos, vecinos, sensibilidad | F2 y E3 |
| D. Colección, campos, índice, filtros, métrica, reindex idempotente, conteos, persistencia | F2 |
| E. Recuperar y generar separados, prompts versionados, contexto, citas, abstención, parámetros | F3 |
| F. Pregunta, modelo, respuesta, tiempo, evidencias, enlaces, errores, carga, modelo dinámico | F3 |
| G. Código, contenedores, instrucciones, pull, pruebas, persistencia, logs, límites | F0 y F8 |
| Hipótesis refutable y protocolo | H1 |
| Experimento principal solo cambia el LLM; cuantización y amenazas a la validez | E4 |
| Ablaciones de una dimensión; colección nueva al cambiar encoder | E1, E2 y E3 |
| Tres repeticiones, frío/caliente, hardware, semilla, versiones, tokens, memoria, fecha | E4 |
| Gold de 40, ambas colecciones, idiomas, fechas, tipos, referencia, dev/test | F4 |
| Compuerta de correctitud antes de escalar (600 documentos, 40+ preguntas) | F1.5 |
| Métricas de recuperación: Recall, Precision, Hit, MRR, MAP, nDCG, fragmento, contexto, duplicados e IC bootstrap | F1.5, E0, E1, E2, E3 y E6 |
| Métricas de generación | E4 |
| Rendimiento p50/p95, tokens/s, RAM, VRAM/CPU, errores | E4 |
| Robustez: alucinaciones, citas, fechas, discrepancias, injection | E5 y F7 |
| Reproducibilidad: versiones, anotaciones, CSV/JSONL, scripts, gráficos | F8 |
| Evaluación humana con rúbrica y dos evaluadores; LLM-judge complementario | F7 |
| Entregables: repo, README, informe PDF, artefactos, demo | F8 |
| `open_corpus_co_es` y nombre `dragons_rag_2026_10_19` | README |
| Licencias y requisitos de los modelos | F0 |
| No publicar el Parquet ni los pesos | `.gitignore` y revisión previa |

## 10. Criterios de aceptación por rúbrica

- 10 puntos, perfilado: figuras y tablas de fuentes, longitudes, idiomas, duplicados, fechas y sesgos, con script reproducible.
- 15 puntos, arquitectura: `docs/ARQUITECTURA.md` con diagrama propio y un ADR por cada decisión costosa.
- 20 puntos, chunking, embeddings y Milvus: dos estrategias comparadas, offsets, IDs estables, deduplicación trazable, colección e índice documentados, conteos y persistencia verificados.
- 15 puntos, API, interfaz y despliegue: compose único, `.env.example`, README desde cero, UI con modelo, k, fuentes y estados.
- 15 puntos, diseño experimental y gold: 40 preguntas, dev y prueba separados, referencia verificable.
- 20 puntos, comparación de LLM: cuatro modelos, evidencia constante, métricas de recuperación, generación, latencia y recursos, análisis de errores.
- 5 puntos, documentación y pruebas: tests que corren sin Milvus y README con solución de problemas.

## 11. Siguiente paso al salir de Plan mode

1. Crear el árbol y los archivos base.
2. Copiar este plan y los documentos de decisiones y hallazgos al repositorio.
3. Construir la imagen, levantar Milvus y Ollama local en 11435, y descargar los cuatro modelos.
4. Instalar Docker y NVIDIA Container Toolkit en el servidor, copiar el compose y levantar allí `ollama-embeddings`.
5. Ejecutar el perfilado y el piloto de 2.000 documentos.
6. Reportar chunks por segundo medidos y la calidad de recuperación del piloto antes de escalar.
