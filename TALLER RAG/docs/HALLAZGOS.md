# Hallazgos del corpus y del entorno

Fecha: 2026-10-30
Fuente: `Dataset/documentos_rag.parquet` y hardware de Carlos.
Script: `scripts/profile_corpus.py` (por crear). Los números de esta página salen de la lectura directa del Parquet con PyArrow.

## 1. Integridad del archivo

| Comprobación | Resultado |
|---|---|
| Filas | 169.248 |
| Columnas | 34 en el Parquet, la guía menciona 33 |
| Tamaño | 256,9 MB |
| `document_id` único | sí, 0 duplicados |
| `text` vacío | 0 |
| `text_sha256` duplicado | 14.771 filas en 4.434 grupos, máximo 839 copias |
| Duplicados por colección | bluesky_posts 9.972, url_crawls 9.233 |
| `url` nula o no http | 0 |
| `title` nulo | 38,21% global, 0% en url_crawls |

Conclusión: el archivo es consistente en identificadores y URLs. Los 14.771 duplicados coinciden con el manifiesto del docente y no se corrigen en el origen.

## 2. Distribución por colección y fuente

| Colección | Documentos |
|---|---|
| url_crawls | 104.578 |
| bluesky_posts | 64.670 |

`source_type`: social_post 64.670, news_article 5.252, nulo 99.326. El campo no sirve como etiqueta semántica, tal como advierte la guía.

Dominios principales: bsky.app 64.670, infobae 19.766, eltiempo 12.229, theguardian 7.216, elespectador 5.456, independent.co.uk 4.799, theconversation 3.721, dailymail 3.305, extra.com.co 1.942, larepublica.co 1.716.

## 3. Longitud

| Métrica | Palabras | Caracteres |
|---|---|---|
| Media | 473 | 2.977 |
| Mediana | 371 | 2.367 |
| p90 | 1.010 | 6.361 |
| p99 | 2.420 | 15.142 |
| Máximo | 58.181 | 363.528 |

Implicación: con chunks de 1000 caracteres y 150 de solape se esperan unas 600.000 unidades. El documento más largo produce cientos de chunks y exige control de lote.

## 4. Fechas

Rango declarado: 2008-07-19 a 2026-09-11. La distribución real se concentra en 2024 (171), 2025 (1.592) y 2026 (3.515). El resto son fechas sueltas antiguas que conviene revisar antes de usarlas como filtro. `created_at` nulo en 0,01%.

Implicación: el filtro temporal solo es fiable en el tramo 2024 a 2026, y el informe debe decirlo.

## 5. Idioma

| Idioma declarado | Documentos |
|---|---|
| en | 76.642 |
| es | 54.918 |
| nulo | 13.137 |
| es-co | 4.668 |
| hu | 4.108 |
| en-global | 3.720 |
| es-mx | 3.698 |
| es-es | 2.011 |
| en-gb | 1.174 |
| es-ar | 809 |
| en-us | 538 |
| en-US | 464 |

Defecto detectado: el campo mezcla códigos ISO con locales y variantes de mayúsculas, y `language_base` no se normaliza de forma consistente. Hay 13.137 documentos sin idioma. La normalización se hace al vuelo en la ingesta y el archivo original no se toca.

Implicación: los filtros de idioma y el gold deben construirse sobre el idioma normalizado, no sobre el campo crudo. El húngaro (4.108) es una bolsa que conviene revisar antes de tratarla como idioma del corpus.

## 6. Sesgos de recolección

- Bluesky aporta 64.670 documentos y llega casi todo en inglés: el corpus no representa la conversación pública en español.
- La prensa está dominada por Infobae, El Tiempo, The Guardian y El Espectador. La cobertura no equivale a veracidad.
- Los textos de redes son afirmaciones sin verificación editorial. El prompt y el informe deben distinguir hecho, atribución y alegación.
- El tramo temporal útil es corto (2024 a 2026), así que las preguntas comparativas deben ceñirse a esa ventana.

## 7. Entorno medido

| Recurso | Valor |
|---|---|
| CPU | AMD Ryzen 9 5950X, 16 núcleos, 32 hilos |
| RAM | 31 GB totales, 17 GB disponibles en reposo |
| GPU | AMD Radeon RX 9070 XT, 16 GB, gfx1201, ROCm cargado |
| Disco | 376 GB libres en /home |
| Docker | 29.9.2, Compose v5.5.0 |
| Ollama | 0.32.13 nativo, activo |
| Python del host | 3.14, sin pip y sin wheels de torch |
| Caché Hugging Face | multilingual-e5-base y bge-reranker-v2-m3 |

Modelos Ollama instalados hoy: solo modelos grandes (qwen3.8 de 13 a 17 GB, gemma4:26b de 17 GB). No están descargados qwen2.5:1.5b, llama3.2:1b, llama3.2:3b ni gemma2:2b.

### Servidor de embeddings

| Recurso | Valor |
|---|---|
| IP | 192.168.1.100 |
| SO | Fedora Linux 44 Server |
| CPU | 12 núcleos |
| RAM | 62 GB, 61 GB disponibles |
| GPU | NVIDIA GTX 1080, 8 GB, driver 580.178.04 |
| Ollama | 0.20.7 nativo, escucha en 0.0.0.0:11434 |
| Docker | no instalado; los repos de Docker CE y NVIDIA Container Toolkit responden para Fedora 44 |
| Grupos | carlos en wheel y ollama; sudo pide contraseña |
| Disco | 98 GB libres de 198 GB (51% usado) |
| Latencia LAN | 0,6 ms de ida y vuelta |
| Modelos | gemma4:latest y gemma4:latest-16k, ninguno de embeddings |

Implicación: torch y sentence-transformers van en el contenedor con Python 3.12, no en el host. La GPU local se reserva a la generación con el Ollama del contenedor. El encoder vive en el servidor en otro contenedor Ollama y se consume por HTTP, así que la ingesta depende de la red local. El disco dejó de ser límite (98 GB libres) y el endpoint no tiene autenticación.

## 8. Estado del índice de código

El proyecto `home-carlos-Git-PLN` de codebase-memory está indexado con generación 2026-10-05, anterior a la carpeta `TALLER RAG`. `check_index_coverage` para el scope `TALLER RAG` devuelve 0 entradas. Hay que reindexar cuando exista código.

## 9. Hallazgos de la puesta en marcha (2026-10-30)

- El daemon de Docker de este equipo tiene `"iptables": false` y `"bridge": "none"`. No existe la red `bridge`, los contenedores no tienen salida a internet y la construcción de imágenes falla al resolver los repositorios Debian. Solución aplicada: red de host en todos los servicios y `build.network: host`.
- MinIO retiró sus imágenes oficiales: `minio/minio` ya no existe en Docker Hub ni en quay.io, y `dl.min.io` responde 410. Se usa `bitnamilegacy/minio` fijado por digest, con `user: "0"`.
- El daemon no resuelve el grupo `render` por nombre al crear el contenedor (`unable to find group render`). Se usan los GID numéricos del host: 983 (video) y 987 (render).
- Con la red de host, `extra_hosts: host.docker.internal:host-gateway` falla (`unable to derive the IP value for host-gateway`). No hace falta: todo se habla por `localhost`.
- El stack local quedó sano: Milvus, MinIO, Ollama, API y frontend arriba, `/health` con Milvus y LLM en verde y embeddings en rojo hasta terminar el servidor.
- Pruebas unitarias: 11 pasan.
- Perfilado ejecutado: 169.248 filas, 14.771 duplicados, 13.140 sin idioma. Figuras en `reports/figuras/`.
- Muestra del piloto: 2.000 documentos, 1.216 de prensa y 784 de Bluesky, semilla 42.

## 10. Ruido del crawler y sensibilidad del índice (2026-10-31)

- 7.378 documentos (4,36% del corpus) son la misma página de error del crawler: "Haz excedido el máximo de peticiones" y "Error 505". Están todos en `url_crawls`.
- 9.021 documentos (5,33%) contienen al menos un patrón de ruido (peticiones, error 505, cookies, inicio de sesión). El filtro aplicado solo descarta los marcadores fuertes de error, no las menciones legítimas de cookies.
- En el conjunto de humo, 18 de 600 documentos se descartaron por calidad.
- HNSW con `ef=64` sobre 2.064 vectores perdía recall por la cercanía de las puntuaciones de e5. Con `ef=200` el recall@10 del conjunto curado pasa de 0,88 a 1,00. La búsqueda es determinista una vez sellado el segmento.


## 11. Correcciones de la auditoría (2026-10-31)

Se corrigieron 18 hallazgos de la auditoría y 4 del revisor independiente:

- El `chunk_id` incluye solape y parámetros de segmentación; la ingesta quedó en un solo bucle en `src/rag/ingest_variant.py` y `src/scripts/ingest.py` es un envoltorio de la variante v1.
- La API valida idioma y fechas, rechaza colecciones de otro encoder y responde 400/422 con mensaje.
- `store` verifica la dimensión de la colección, cuenta sin flush y devuelve el `chunk_id` real en cada hit.
- El gold preserva `reviewer`, `validated` y `split` al regenerarse; el Excel refleja `validated` y calcula los contadores reales.
- Las métricas usan P@1 y P@5, reportan `n_registros` y `n_preguntas`, y aplican Bonferroni en las comparaciones. `precision@k` se retiró del resumen.
- El healthcheck de Ollama apunta a 11435. La imagen se reconstruyó con `openpyxl`.
- 37 pruebas en verde y gate de QA 12/12.

## 12. Ingesta completa: Milvus limita los VARCHAR en bytes (2026-10-31)

El primer intento de indexar los 169.248 documentos falló en la fila 346: el campo `title` medía 1.048 bytes y el máximo del esquema es 1.024. La truncación era por caracteres, no por bytes, y un título con acentos o emojis ocupa más bytes que caracteres.

Se agregó `truncate_bytes` en `src/rag/corpus.py` y se aplica a `text`, `title`, `url`, `collection_name`, `source` y `chunk_strategy` en la ingesta. El título más largo del corpus mide 8.170 bytes; los demás campos no se acercan a su límite. La corrida se relanzó con `--drop`.
