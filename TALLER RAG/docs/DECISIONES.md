# Registro de decisiones del Taller RAG

Formato: ADR. Cada decisión costosa de revertir queda con contexto, alternativas y consecuencias.
Destino final: `TALLER RAG/docs/DECISIONES.md`.

## ADR-001. Corpus completo con compuerta de rendimiento

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: la guía permite piloto de 500 a 2.000, luego 10.000 y corpus completo si el hardware lo permite. Carlos eligió el corpus completo. La única medición de rendimiento disponible hoy es una estimación.
- Decisión: indexar los 169.248 documentos con la configuración ganadora, después de medir chunks por segundo en el piloto de 500. Si el rendimiento medido no permite terminar en 12 horas, se indexan 50.000 y la desviación se documenta con el número real.
- Alternativas: quedarse en 10.000 desde el inicio (más rápido), o lanzar el corpus completo sin medir (riesgo de bloquear una semana).
- Consecuencias: la colección final ocupa del orden de 3 a 5 GB. Las ablaciones se hacen sobre una muestra fija de 10.000 para no multiplicar el costo.

## ADR-002. Encoder en un servicio separado, en el servidor

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: Carlos pidió ejecutar el LLM en el equipo local y el modelo de embeddings en el servidor `192.168.1.100`, con todo el entorno levantado desde Docker Compose. El servidor tiene una GTX 1080 de 8 GB, 62 GB de RAM, 98 GB libres y no tiene Docker instalado todavía. El equipo local tiene una RX 9070 XT de 16 GB y el host solo tiene Python 3.14, sin wheels de torch. La guía permite que el encoder viva en un servicio separado y exige documentarlo.
- Decisión: el encoder vive en el servidor, en un contenedor Ollama definido en el mismo `docker-compose.yml`, publicado en `http://192.168.1.100:11435` porque el Ollama nativo del servidor ocupa el 11434. Modelo por defecto `qwen3-embedding:8b`, vector de 4096 dimensiones truncado a 1024 por MRL y renormalizado L2, métrica COSINE. Las consultas llevan `Instruct: ...\nQuery: ...`; los documentos van sin prefijo. El cliente usa lotes de 32, `num_ctx=512` y tres reintentos con espera exponencial. Respaldo: `qwen3-embedding:0.6b` en el mismo contenedor, o `intfloat/multilingual-e5-base` en la API sobre CPU.
- Alternativas: encoder en la API local sobre CPU (más lento para 169.248 documentos), encoder en la GPU local (compite con el LLM y la GPU local es la que se muestra en la demo), e5-base en el servidor (peor calidad y ya hay un modelo multilingüe más fuerte instalado).
- Consecuencias: la ingesta depende de la red local y del servidor. Se mitiga con reintentos, `--start` y el respaldo local. Los vectores viajan por LAN y Milvus queda en el equipo local. El endpoint del servidor no tiene autenticación y no debe exponerse a internet.

## ADR-003. Milvus standalone y una colección por configuración

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: la guía exige Milvus como único motor vectorial y prohíbe mezclar ablaciones de recuperación.
- Decisión: Milvus standalone con etcd y MinIO como infraestructura. Colecciones con nombre `rag_{alcance}_{encoder}_{chunking}_v1`. Esquema con campos escalares para filtrar por colección, idioma y fecha. Índice HNSW con M=16, efConstruction=200, COSINE, ef=64.
- Alternativas: una sola colección con campo de partición por configuración (más barato de operar, más fácil de contaminar métricas con vectores de otra configuración).
- Consecuencias: más espacio en disco y un `drop` explícito por colección. A cambio, cada número de recuperación es atribuible a una sola configuración.

## ADR-004. Dos estrategias de chunking

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: la guía pide comparar dos alternativas y conservar offsets y trazabilidad.
- Decisión: `fixed_chars` con 1000 caracteres y 150 de solape como línea base, y `sentence_tokens` con máximo 384 tokens, solape de una oración y corte duro solo cuando una oración excede el límite. IDs deterministas por documento, estrategia, tamaño, solape y offset.
- Alternativas: chunking semántico por embeddings (caro, difícil de reproducir), chunking por párrafos (frágil con texto de redes).
- Consecuencias: la comparación es de una sola dimensión. Se reportan chunks generados, longitud, solape efectivo y duplicados descartados por estrategia.

## ADR-005. Ollama en Docker por defecto, host como respaldo

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: la sección 9 de la guía arranca los modelos con `docker compose exec ollama ollama pull MODELO`, así que la ruta reproducible es el contenedor. Ollama 0.32.13 también funciona de forma nativa en el host, pero esa ruta no queda cubierta por el compose. El soporte de gfx1201 dentro de la imagen Docker no está verificado en esta máquina.
- Decisión: servicio `ollama` en el `docker-compose.yml` con la imagen oficial, publicado en el puerto 11435 del host y escuchando en 11434 dentro del contenedor, volumen `ollama_data` para los pesos y acceso a la GPU con `devices: ["/dev/kfd", "/dev/dri"]` y `group_add` para los grupos `video` y `render`. La API usa `OLLAMA_URL=http://ollama:11434`. Se descargan los cuatro modelos del PDF: qwen2.5:1.5b, llama3.2:1b, llama3.2:3b y gemma2:2b. Respaldo: `OLLAMA_URL=http://host.docker.internal:11434` con `extra_hosts: ["host.docker.internal:host-gateway"]`, que apunta al Ollama nativo que ya usa Carlos, documentado en `.env.example` y en el README.
- Alternativas: solo Ollama nativo (rápido, pero fuera del compose y menos reproducible), solo contenedor sin respaldo (una demo fallida si ROCm no engancha la GPU).
- Consecuencias: la primera descarga son unos 6 GB y queda en el volumen. En F0 se verifica con `ollama ps` que el modelo corre en GPU. Si corre en CPU, se cambia `OLLAMA_URL` al host y se registra como limitación del entorno, sin tocar el resto del código.

## ADR-006. Contexto por tokens, citas obligatorias y abstención

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: la guía exige controlar el presupuesto de contexto, exigir citas y permitir abstención. Un presupuesto en caracteres no es comparable entre modelos.
- Decisión: presupuesto en tokens con el tokenizador del encoder, 3000 por defecto. Citas `[n]` obligatorias, validación posterior de los índices citados y campo `invalid_citations`. Abstención sin llamar al LLM cuando no hay evidencia o cuando la mejor puntuación no supera el umbral fijado en desarrollo. Prompt versionado con hash en cada corrida.
- Alternativas: confiar en el prompt sin verificación (no se puede medir), abstención delegada al modelo (poco fiable en modelos de 1 a 3B).
- Consecuencias: el umbral se calibra en desarrollo y se congela antes de prueba. Las preguntas sin respuesta se evalúan como una categoría propia.

## ADR-007. Protocolo de evaluación

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: la guía pide gold de 40 preguntas, separar desarrollo y prueba, tres repeticiones y dos anotadores en una submuestra.
- Decisión: 20 preguntas de desarrollo y 20 de prueba. Ablaciones de recuperación sobre el piloto fijo de 2.000 documentos. Comparación de los cuatro LLM sobre las 20 preguntas de prueba con evidencia idéntica, temperatura 0 y tres repeticiones; la repetición 1 se anota y las otras dos miden latencia y estabilidad. Rúbrica humana de 0 a 3 en corrección, fidelidad, citas y abstención. Dos anotadores sobre las 80 respuestas de prueba.
- Alternativas: juez automático único (la guía lo permite solo como complemento), una sola repetición (no permite separar ruido de efecto), anotar las tres repeticiones (triplica el trabajo humano sin mejorar la comparación).
- Consecuencias: 80 respuestas anotadas, más 60 corridas extra para latencia. La selección del ganador usa las mismas 20 preguntas de prueba, y esa limitación se declara en el informe; el conjunto de desarrollo se reserva para prompts, umbrales y ablaciones.

## ADR-008. Duplicados: no tocar el origen, deduplicar en recuperación

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: 14.771 filas comparten texto. La guía prohíbe alterar el archivo de forma silenciosa y pide trazabilidad de duplicados.
- Decisión: el Parquet se lee en solo lectura y no se modifica. La ingesta guarda `text_sha256` en cada chunk. La recuperación descarta chunks con el mismo hash antes de armar el contexto. El gold no usa documentos duplicados como referencia.
- Alternativas: borrar duplicados en la ingesta (pierde procedencia), dejarlos (infla el top-k y las métricas).
- Consecuencias: el top-k efectivo puede ser menor que k cuando hay duplicados. Se reporta la tasa de descarte.

## ADR-009. Inyección de instrucciones en documentos

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: la rúbrica de robustez pide evaluar resistencia a prompt injection dentro de los documentos.
- Decisión: colección separada `rag_injection_e5base_v1` con documentos sintéticos marcados y una instrucción maliciosa. El prompt delimita la evidencia y ordena ignorar instrucciones internas. Se mide si el modelo obedece, y esos documentos quedan fuera de las métricas de recuperación normales.
- Alternativas: inyectar en la colección principal (contamina el corpus y las métricas).
- Consecuencias: un experimento pequeño y aislado, con resultados reportados aparte.

## ADR-010. Datos y pesos fuera de Git

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: la guía prohíbe publicar el Parquet sin autorización. El repositorio padre ya trackea otros proyectos.
- Decisión: `.gitignore` dentro de `TALLER RAG` con `Dataset/`, `*.parquet`, `.env` y los pesos descargados. El Parquet se monta en solo lectura desde `Dataset/`. Los JSONL de `reports/corridas/` y `eval/results/` sí se versionan porque son la evidencia cruda. Antes de publicar se revisa `git status` y el tamaño del commit.
- Alternativas: confiar en la revisión manual (un descuido publica 257 MB de datos restringidos).
- Consecuencias: el README explica cómo obtener el corpus con `open_corpus_co_es` o cómo colocarlo en `Dataset/`.

## ADR-011. Primero el piloto, después el corpus completo

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: Carlos pidió probar con pocos datos, suficientes para comprobar que la recuperación funciona, recorrer los cuatro modelos del taller y elegir el mejor antes de construir el RAG final. Indexar 169.248 documentos antes de saber si el pipeline recupera bien cuesta horas por configuración.
- Decisión: el desarrollo, el gold, las ablaciones y la comparación de modelos se hacen sobre un piloto estratificado de 2.000 documentos. El corpus completo se indexa una sola vez, al final, con la configuración ganadora, y se valida con E6 (mismas 20 preguntas de prueba, solo recuperación).
- Alternativas: indexar el corpus completo desde el inicio (realimentación lenta y costo alto por cada cambio de chunking o encoder), quedarse en el piloto (el informe no muestra comportamiento a escala).
- Consecuencias: la comparación de modelos ocurre en un corpus pequeño y la degradación por escala se mide y se reporta. Si el ganador cambia al escalar, se documenta como hallazgo, no se oculta.

## ADR-012. Acceso al servidor sin secretos en el repositorio

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: el servidor se entregó con usuario y contraseña, y su Ollama escucha en `0.0.0.0:11434` sin autenticación. La guía prohíbe publicar datos y el repositorio será público o compartido con el curso.
- Decisión: configurar una llave SSH con `ssh-copy-id carlos@192.168.1.100` y no escribir la contraseña en ningún archivo del proyecto. `EMBEDDINGS_URL` va en `.env` (ignorado por Git) con un valor de ejemplo en `.env.example`. El endpoint de Ollama se usa solo en la red local y no se expone a internet.
- Alternativas: guardar la contraseña en `.env` (un secreto en disco y en el historial), usar el Ollama del servidor a través de un túnel SSH (más seguro, más lento y más pasos para la demo).
- Consecuencias: la ingesta no necesita SSH, solo el endpoint HTTP. Si la contraseña se compartió por un canal inseguro, conviene cambiarla. El README advierte que el endpoint no tiene autenticación.

## ADR-013. Un compose, dos hosts, perfiles

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: Carlos quiere el LLM en local y los embeddings en el servidor, con todo levantado desde Docker Compose. La guía exige un único `docker-compose.yml` en la raíz. Son dos máquinas con GPUs distintas: AMD en local, NVIDIA en el servidor.
- Decisión: un solo `docker-compose.yml` con dos grupos. Los servicios locales (etcd, MinIO, Milvus, Ollama del LLM, API y frontend) arrancan con `docker compose up -d --build`. El servicio `ollama-embeddings` vive en el perfil `embeddings` y solo se levanta cuando se nombra explícitamente: en el servidor se copia el compose y se ejecuta `docker compose up -d ollama-embeddings`. La GPU local se declara con `devices: ["/dev/kfd", "/dev/dri"]`; la del servidor con `deploy.resources.reservations.devices` y driver `nvidia`. Regla de puertos: el Ollama nativo se queda en 11434 y todo Ollama en Docker publica 11435.
- Alternativas: dos archivos compose (incumple el requisito de un único compose), un clúster o contexto remoto de Docker (más maquinaria para un taller), todo en una sola máquina (pierde el reparto que pidió Carlos y deja la GPU local compitiendo entre embeddings y generación).
- Consecuencias: el servidor necesita Docker Engine, Compose v2 y NVIDIA Container Toolkit, instalados una vez con sudo. El mismo archivo sirve en los dos equipos y `docker compose config` lo valida antes de arrancar. El README documenta los dos comandos y el orden.

## ADR-014. MinIO desde el espejo legacy de Bitnami

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: al levantar el stack, `minio/minio:RELEASE.2024-12-18T13-15-44Z` ya no existe. MinIO retiró sus imágenes oficiales de Docker Hub y de quay.io, y `dl.min.io` responde 410. Milvus necesita almacenamiento compatible con S3.
- Decisión: usar `bitnamilegacy/minio`, fijado por digest (`sha256:451fe685...`), con `user: "0"` y volumen en `/bitnami/minio/data`. Se verificó el healthcheck en 9000 y el arranque del servidor.
- Alternativas: `common.storageType: local` de Milvus (elimina el servicio, pero se aleja de la arquitectura de referencia y no se probó a escala), RustFS (backend S3 que el propio blog de Milvus propone como reemplazo, pero es otra dependencia joven), `chainguard/minio` (imagen mantenida, pero sin shell y con permisos de volumen más delicados).
- Consecuencias: se conserva la arquitectura Milvus + etcd + MinIO. El espejo es legacy: si desaparece, se cambia a RustFS o a almacenamiento local de Milvus. El digest queda registrado en `docs/EXPERIMENTOS.md`.

## ADR-015. Red de host porque dockerd no tiene bridge ni iptables

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: `/etc/docker/daemon.json` de este equipo tiene `"iptables": false` y `"bridge": "none"`. La red `bridge` no existe, los contenedores no tienen salida y la construcción de la imagen falla al resolver `deb.debian.org`. Cambiar la configuración global exige sudo y reinicia Docker, lo que tumba el contenedor de Graphiti que está en uso.
- Decisión: todos los servicios usan `network_mode: host` y la construcción usa `build.network: host`. Los servicios se encuentran por `localhost` y los puertos son los del host: Milvus 19530 y 9091, MinIO 9000 y 9001, Ollama 11435, API 8010, Streamlit 8501. Ollama dentro del contenedor escucha en `0.0.0.0:11435` con `OLLAMA_HOST`.
- Alternativas: habilitar iptables y el bridge por defecto (cambio global y reinicio de Docker), crear una red con NAT propio (requiere iptables, que está desactivado), ejecutar todo fuera de Docker (incumple el requisito del taller).
- Consecuencias: el compose funciona en esta máquina y en el servidor, pero no es portable a Docker Desktop. En un Docker estándar se quitan los `network_mode: host` y se usan los nombres de servicio (`milvus`, `ollama`) como en el boilerplate. Queda documentado en el README y en `docs/ARQUITECTURA.md`.

## ADR-016. e5-base en CPU para el corpus completo; qwen3-embedding queda como comparación

- Estado: aceptada.
- Fecha: 2026-10-30.
- Contexto: se midió el rendimiento real de cada encoder con lotes de 16 textos de unos 900 caracteres. qwen3-embedding:8b en la GTX 1080 del servidor da 1,9 chunks/s; qwen3-embedding:0.6b en el servidor da 2,5; qwen3-embedding:8b en la RX 9070 XT local da 7,3; e5-base en el Ryzen 9 con 8 hilos da 12,2 y con 24 hilos da 25,5. El corpus completo son unas 600.000 unidades: con el servidor serían 67 a 88 horas, con la GPU local 23 horas y con e5-base en CPU 6,5 horas.
- Decisión: el corpus completo y el piloto se indexan con `intfloat/multilingual-e5-base` en la API, sobre CPU, 768 dimensiones, prefijos `query: ` y `passage: `. qwen3-embedding:8b se conserva en el servidor para la comparación E3 sobre el piloto de 2.000 documentos, donde el costo es de minutos.
- Alternativas: indexar el corpus completo con qwen3-embedding en la GPU local (23 horas y deja la GPU ocupada todo el día), reducir el corpus a 50.000 documentos (autorizado por la compuerta, pero pierde cobertura), usar qwen3-embedding:0.6b (2,5 chunks/s en el servidor, peor calidad y aún 67 horas).
- Consecuencias: la arquitectura de embeddings en el servidor se mantiene para E3, pero la producción usa CPU local. El informe debe reportar la comparación de calidad y la comparación de costo, y declarar que la elección final se basa en el rendimiento medido, no en una preferencia. El umbral de la compuerta de 20 chunks/s se cumple solo con e5-base y 24 hilos.

## ADR-017. E3 se ejecuta en la GPU local, no en el servidor

- Estado: aceptada.
- Fecha: 2026-10-31.
- Contexto: E3 compara calidad de encoders y es solo recuperación; el LLM está inactivo. Medido: qwen3-embedding:8b rinde 7,3 chunks/s en la RX 9070 XT y 1,9 en la GTX 1080. Sobre el piloto de 7.783 unidades son unos 18 minutos en local contra unos 68 en el servidor.
- Decisión: E3 indexa el piloto en la GPU local con qwen3-embedding:8b, 4096 dimensiones truncadas a 1024, colección `rag_pilot_qwen3emb8b_d1024_fixed_v1`, y compara contra la colección de e5-base con las mismas preguntas y la misma k. El servidor queda como alternativa documentada y se valida con una muestra de 100 textos.
- Alternativas: ejecutar E3 completo en el servidor (misma comparación, cuatro veces más espera), ejecutar E3 en CPU con qwen (no hay pesos GGUF para sentence-transformers y sería más lento).
- Consecuencias: la comparación de calidad usa el mismo modelo y la misma truncación; solo cambia el dispositivo, que no afecta los vectores salvo por el orden de las operaciones de coma flotante. Se reporta que la ruta del servidor se validó con la muestra de humo y no con el piloto completo.

## ADR-018. Limpieza V2: filtro de ruido más normalización conservadora

- Estado: aceptada.
- Fecha: 2026-10-31.
- Contexto: se compararon cuatro variantes sobre el piloto de 2.000 documentos, 30 preguntas específicas (25 de prensa y 5 de Bluesky), 3 repeticiones, k=10, e5-base y el mismo índice HNSW con `ef=200`. V0 es texto crudo; V1 descarta páginas de error; V2 añade NFKC, quita HTML y URLs y colapsa espacios; V3 añade deduplicación de near-duplicates por chunk.
- Decisión: adoptar V2 para el RAG final por consistencia de las estimaciones puntuales y costo nulo, sin afirmar superioridad estadística con 30 preguntas. No adoptar V3: elimina 159 chunks y no supera a V2 en ninguna métrica.
- Evidencia: V2 obtiene P@1 0,833 contra 0,733 de V0, P@5 0,900 contra 0,833, recall@10 0,900 contra 0,833 y nDCG@10 0,875 contra 0,790. Con corrección de Bonferroni sobre 21 pruebas, el bootstrap pareado da P@1 +0,100 con IC [0,000; 0,267], recall +0,067 con IC [0,000; 0,233] y nDCG +0,086 con IC [0,000; 0,252]. Ninguna diferencia de calidad excluye cero con 30 preguntas. La precisión@10 se descartó como evidencia porque con un solo documento relevante por pregunta está acotada por 1/k. V1 sola no mueve el recall (0,833) y su ganancia de nDCG (+0,019) no es concluyente. V3 queda por debajo de V2 en todas las métricas.
- Alternativas: V0 (texto crudo), V1 (solo filtro), V3 (filtro + normalización + dedup de chunks), V4 (normalización agresiva, no ejecutada por ser un control negativo de costo conocido).
- Consecuencias: la limpieza se aplica al texto de cada chunk en el momento de indexar; `position` y `end_offset` siguen referidos al texto original, y `text_sha256` es el del texto limpio. La métrica de cobertura de span es menos fiable en variantes limpias porque el span anotado viene del texto crudo. El subconjunto de Bluesky (5 preguntas) es demasiado pequeño para concluir; las 5 son de tipo directa, así que el desglose por tipo no aporta. El gold final debe ser humano e incluir preguntas comparativas, ambiguas y sin respuesta.

## ADR-019. Chunking fijo, k=5 y gold standard de 43 preguntas

- Estado: aceptada.
- Fecha: 2026-10-31.
- Contexto: E1 comparó chunking fijo (1000/150) contra oraciones (384 tokens, solape de una oración) con limpieza V2, 30 preguntas, 3 repeticiones, e5-base y `ef=200`. E2 comparó k=3, 5 y 10 sobre la mejor colección. Además se consolidó un gold standard de 43 preguntas.
- Decisión: mantener el chunking fijo de 1000 caracteres con 150 de solape. Fijar k=5 en el sistema final, sabiendo que k=3 da la misma calidad en este gold y que k=5 deja margen para preguntas comparativas. Usar `eval/gold/gold_standard.jsonl` como gold de evaluación, con 43 preguntas (38 respondibles y 5 sin respuesta), 19 de desarrollo y 24 de prueba.
- Evidencia E1: fijo P@1 0,833, P@5 0,900, recall 0,900, nDCG 0,875, 38,5 ms; oraciones P@1 0,767, P@5 0,833, recall 0,833, nDCG 0,804, 42,1 ms. Con Bonferroni, la diferencia pareada de nDCG es -0,071 con IC [-0,221; 0,025]; no es concluyente, pero el fijo gana en estimación puntual y es 3,65 ms más rápido (IC [2,25; 4,97]).
- Evidencia E2: k=3, 5 y 10 dan exactamente las mismas métricas de calidad (P@1 0,833, recall 0,900, nDCG 0,875); todos los relevantes están en el top 3. La diferencia pareada de calidad es 0,000; solo cambia la latencia en 1,6 ms.
- Evidencia del gold: sobre la colección `rag_pilot_e5base_d768_v2_v1`, 38 preguntas respondibles dan recall@10 0,947, MRR 0,908 y nDCG@10 0,918 (IC [0,870; 0,959]). Los dos fallos son preguntas reales, no referencias rotas.
- Alternativas: oraciones (peor estimación puntual y más lento), k=3 (misma calidad, menos contexto), k=10 (sin ganancia medible, más contexto).
- Consecuencias: el gold está generado por LLM y pendiente de validación humana (`validated: false`). No tiene preguntas comparativas y solo 5 de Bluesky y 10 en inglés; el informe debe declararlo. Las 5 preguntas sin respuesta se evalúan por abstención, no por Recall@k.


## ADR-020. Ingesta del corpus completo en la GPU local con ROCm

- Estado: aceptada.
- Fecha: 2026-10-31.
- Contexto: e5-base en CPU local rinde 14,5 chunks/s de extremo a extremo y el corpus completo tarda 11 a 12 horas. La RX 9070 XT estaba libre y el host tiene ROCm 7.2.4.
- Decisión: construir una imagen ROCm con `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` y un servicio `ingest-rocm` en el perfil `rocm`. La ingesta usa `EMBEDDING_DEVICE=cuda` y `EMBEDDINGS_BATCH=128`. La API sigue en CPU.
- Evidencia: e5-base mide 106,9 chunks/s en la GPU contra 25,5 en CPU. La prueba de extremo a extremo da 66,7 chunks/s, así que el corpus completo baja de 11 a 12 horas a 2 o 3 horas.
- Alternativas: CPU local (11 a 12 horas), servidor con 16 vCPU (15 a 18 horas y dependencia de red), GPU del servidor con qwen (88 horas).
- Consecuencias: la imagen ROCm pesa del orden de 30 GB. Los vectores se generan en la GPU y las consultas en CPU con el mismo modelo; las diferencias de coma flotante son despreciables. La GPU se comparte con Ollama, así que no conviene generar texto durante la indexación.
