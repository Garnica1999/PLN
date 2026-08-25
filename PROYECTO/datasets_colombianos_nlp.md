# Datasets libres de PLN en español con temática colombiana

Catálogo de datasets de libre acceso para Procesamiento de Lenguaje Natural (PLN), en español y orientados a temas de Colombia. Cada ficha incluye: **título**, **link de acceso**, **de qué trata**, **autores** y **aplicaciones posibles**.

> Nota: toda la información de las fichas fue verificada directamente en las tarjetas oficiales de cada dataset (Hugging Face y repositorios de origen).

---

## Índice

### Corpus generales y dialectología
1. [CoNELI — Corpus de dialectos del español colombiano](#1-coneli--corpus-de-dialectos-del-español-colombiano)
2. [OSCAR / mC4 / CC-100 (filtrado por dominios .co)](#2-oscar--mc4--cc-100-filtrado-por-dominios-co)
3. [PRESEEA — Corpus sociolingüístico (ciudades de Colombia)](#3-preseea--corpus-sociolingüístico-ciudades-de-colombia)

### Sentimiento, emociones y detección de acoso
4. [Tweets y emociones en elecciones de Colombia](#4-tweets-y-emociones-en-elecciones-de-colombia)
5. [Emociones en español colombiano (pysentimiento)](#5-emociones-en-español-colombiano-pysentimiento)
6. [SentiCorp-NT — Sentimiento en tweets colombianos](#6-senticorp-nt--sentimiento-en-tweets-colombianos)
7. [Cyberbullying en español colombiano (Dataset 1)](#7-cyberbullying-en-español-colombiano-dataset-1)
8. [Cyberbullying en español colombiano (Dataset 2)](#8-cyberbullying-en-español-colombiano-dataset-2)

### Noticias, conflicto y memoria
9. [Historias del conflicto colombiano](#9-historias-del-conflicto-colombiano)
10. [Colombian Conflict SQA — Preguntas y respuestas sobre el conflicto](#10-colombian-conflict-sqa--preguntas-y-respuestas-sobre-el-conflicto)
11. [Colombia News — Noticias colombianas](#11-colombia-news--noticias-colombianas)
12. [Violence and Conflict Events in Colombia (OCHA)](#12-violence-and-conflict-events-in-colombia-ocha)

### Legal y gubernamental
13. [Sentencias de la Corte Constitucional de Colombia 1992-2021](#13-sentencias-de-la-corte-constitucional-de-colombia-1992-2021)
14. [Reglamento Aeronáutico Colombiano (RAC) 2024](#14-reglamento-aeronáutico-colombiano-rac-2024)
15. [Actas del Banco de la República de Colombia](#15-actas-del-banco-de-la-república-de-colombia)

### Voz y habla
16. [Google Colombian Spanish — Voz colombiana](#16-google-colombian-spanish--voz-colombiana)
17. [Colombian Spanish Speech Recognition Corpus (Dataocean)](#17-colombian-spanish-speech-recognition-corpus-dataocean)

### Datos abiertos
18. [datos.gov.co — Portal de datos abiertos de Colombia](#18-datosgovco--portal-de-datos-abiertos-de-colombia)

---

## 1. CoNELI — Corpus de dialectos del español colombiano

- **Título:** CoNELI (Corpus of Dialects of Colombian Spanish) — tarea compartida de IberLEF 2023.
- **Link de acceso:** [IberLEF 2023 — CoNELI](https://sites.google.com/view/iberlef-2023/tasks/coneli) · [GitHub de la tarea](https://github.com/josecalvo6/CoNELI)
- **De qué trata:** Corpus diseñado para la **identificación de dialectos** del español de Colombia. Distingue entre variedades regionales como el bogotano, paisa, costeño, valluno, etc. Incluye también una subtarea de **identificación de dialecto de baja recurso**. Fue usado como *shared task* en IberLEF 2023.
- **Autores:** Organizado por investigadores del grupo de investigación en PLN del español colombiano; participación de equipos de la Universidad de Antioquia, la Pontificia Universidad Javeriana y otras instituciones colombianas (ver sitio de la tarea).
- **Aplicaciones posibles:** Clasificación/detección de dialectos, análisis de variación dialectal, adaptación de ASR y TTS a variedades regionales, normalización de textos coloquiales, estudios sociolingüísticos.

---

## 2. OSCAR / mC4 / CC-100 (filtrado por dominios .co)

- **Título:** Corpora web masivos multilingües (OSCAR, mC4, CC-100) con filtrado a contenido colombiano.
- **Link de acceso:**
  - [OSCAR](https://huggingface.co/datasets/oscar)
  - [mC4](https://huggingface.co/datasets/mc4)
  - [CC-100](https://huggingface.co/datasets/cc100)
- **De qué trata:** Grandes colecciones de texto extraídas de la web (Common Crawl) en cientos de idiomas, incluido el español. No son exclusivamente colombianos, pero se pueden **filtrar por el dominio `.co`** o por frecuencia de términos localistas para obtener subcorpus de contenido colombiano.
- **Autores:** Comunidades abiertas: OSCAR (Inria, NLLB, ONU), mC4 (Google), CC-100 (Facebook AI / Meta).
- **Aplicaciones posibles:** Pre-entrenamiento de modelos de lenguaje (LLMs) y embeddings para español, generación de vocabularios, modelos de lenguaje especializados en español colombiano, análisis léxico y de frecuencia.

---

## 3. PRESEEA — Corpus sociolingüístico (ciudades de Colombia)

- **Título:** Proyecto para el Estudio Sociolingüístico del Español de España y América (PRESEEA) — Colombia.
- **Link de acceso:** [PRESEEA](https://preseea.linguas.net)
- **De qué trata:** Corpus de **entrevistas semidirigidas** de habla espontánea, estratificadas por sexo, edad y nivel educativo. En Colombia participan ciudades como **Bogotá, Cali, Medellín y Barranquilla**, con material transcrito y normalizado.
- **Autores:** Red internacional de equipos de sociolingüística coordinada por la Universidad de Alcalá; nodos colombianos en universidades locales.
- **Aplicaciones posibles:** Análisis sociolingüístico, variación y cambio lingüístico, reconocimiento de voz con acento colombiano, entrenamiento de TTS, estudios de discurso, normalización de transcripciones orales.

---

## 4. Tweets y emociones en elecciones de Colombia

- **Título:** tweets_emotions_elections_colombia.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/jjiguaran/tweets_emotions_elections_colombia)
- **De qué trata:** ~1.199 tweets en español colombiano recolectados durante la **campaña presidencial de Colombia de 2022**, etiquetados con emociones (**alegría, miedo, asco, tristeza**), con etiquetas *multi-label* y metadatos de tweet (ID y texto). Licencia MIT, formato CSV.
- **Autores:** jjiguaran (publicado en Hugging Face).
- **Aplicaciones posibles:** Análisis de sentimiento y emociones en redes sociales, monitoreo de opinión pública y campañas políticas, análisis de polarización, entrenamiento de clasificadores de emociones en español colombiano.

---

## 5. Emociones en español colombiano (pysentimiento)

- **Título:** emociones_colombia.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/pysentimiento/emociones_colombia)
- **De qué trata:** ~1.199 tweets de las elecciones colombianas 2022 anotados con **4 emociones multi-etiqueta (alegría, miedo, asco, tristeza)**, e incluye además anotaciones de **razonamiento (chain-of-thought) generadas con GPT-3.5** (`gpt-3.5_cot`) que justifican cada etiqueta. Formato parquet.
- **Autores:** Organización pysentimiento (basado en el corpus de jjiguaran).
- **Aplicaciones posibles:** Clasificación de emociones, análisis de opinión en contexto político, *annotation* y explicabilidad (CoT), evaluación de LLMs en tareas de emociones en español.

---

## 6. SentiCorp-NT — Sentimiento en tweets colombianos

- **Título:** SentiCorp-NT (Corpus de sentimiento para tweets de Colombia).
- **Link de acceso:** [Universidad Nacional de Colombia (descripción del corpus)](https://revistas.unal.edu.co/index.php/maripaz/article/view/... ) — el corpus se publica bajo demanda académica; ver descripción en el artículo *"SentiCorp-NT: A sentiment corpus for Colombian tweets"*.
- **De qué trata:** Corpus de **tweets colombianos etiquetados con polaridad de sentimiento** (positivo, negativo, neutro) y clasificados por categorías temáticas. Fue construido por el grupo de investigación en PLN de la Universidad Nacional de Colombia y utilizado en múltiples estudios de sentimiento en español.
- **Autores:** Grupo de PLN, Universidad Nacional de Colombia (Jorge E. Camargo et al.).
- **Aplicaciones posibles:** Análisis de sentimiento en redes sociales, monitoreo de marca y opinión, clasificación de tweets por tema y polaridad, evaluación de modelos de sentimiento en español latinoamericano.

---

## 7. Cyberbullying en español colombiano (Dataset 1)

- **Título:** Colombian_Spanish_Cyberbullying_Dataset_1.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_1)
- **De qué trata:** 3.570 tweets en español colombiano etiquetados manualmente como **ciberacoso / no ciberacoso**. Característica distintiva: por cada palabra ofensiva hay tweets anotados como ciberacoso y otros que la usan sin ser ciberacoso, para que el modelo aprenda del **contexto**. Categorías base: insulto, amenaza, maldición y difamación. Licencia MIT, formato CSV.
- **Autores:** Felipe Guerra (etiquetado por terapeuta ocupacional asociada al proyecto).
- **Aplicaciones posibles:** Detección de ciberacoso y discurso de odio, moderación de contenido en redes sociales, clasificación por categorías de agresión, análisis de lenguaje ofensivo en español colombiano.

---

## 8. Cyberbullying en español colombiano (Dataset 2)

- **Título:** Colombian_Spanish_Cyberbullying_Dataset_2.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_2)
- **De qué trata:** 2.566 tweets en español colombiano, balanceados entre **ciberacoso y no ciberacoso**. Los tweets de "no ciberacoso" provienen de discusiones populares variadas de usuarios colombianos (sin palabras obscenas), mientras los de "ciberacoso" cubren insulto, amenaza, maldición y difamación. Licencia MIT, formato CSV.
- **Autores:** Felipe Guerra (mismo esquema de anotación que el Dataset 1).
- **Aplicaciones posibles:** Detección de ciberacoso, sistemas de moderación, análisis de discurso tóxico en contextos colombianos, entrenamiento de clasificadores robustos al *contexto*.

---

## 9. Historias del conflicto colombiano

- **Título:** historias_conflicto_colombia.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/jorge-henao/historias_conflicto_colombia)
- **De qué trata:** 456 relatos/testimonios sobre el **conflicto armado colombiano**, con columnas `title` y `content` (textos extensos de testimonios y narraciones). Incluye historias de víctimas, memoria y procesos de verdad. Licencia Apache-2.0, formato CSV. Tiene un Space asociado (`jorge-henao/historias-conflicto-colombia`).
- **Autores:** Jorge Henao.
- **Aplicaciones posibles:** Análisis de memoria histórica, generación de resúmenes de testimonios, análisis narrativo del conflicto, construcción de bases para RAG sobre paz y memoria, estudios de discurso y derechos humanos.

---

## 10. Colombian Conflict SQA — Preguntas y respuestas sobre el conflicto

- **Título:** colombian-conflict-SQA.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/jdavit/colombian-conflict-SQA)
- **De qué trata:** ~2.930 pares de **pregunta-contexto-respuesta** (`question`, `context`, `answer`) construidos a partir de informes sobre el conflicto armado colombiano (p. ej. informes de la Comisión de la Verdad). Las preguntas/respuestas fueron **generadas con GPT-4o** y verificadas. Licencia CC-BY-4.0, formato parquet, con división train/test.
- **Autores:** David Alejandro (jdavit).
- **Aplicaciones posibles:** *Question Answering* (QA) extractivo sobre documentos del conflicto, *Retrieval-Augmented Generation* (RAG), chatbots sobre verdad y memoria, evaluación de modelos de lectura comprensiva en español.

---

## 11. Colombia News — Noticias colombianas

- **Título:** colombia-news.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/Villaitech/colombia-news)
- **De qué trata:** ~1.016 artículos de noticias de medios colombianos (**El Tiempo, Portafolio, La Nación, Diario Occidente, The City Paper Bogotá**) recolectados en abril de 2025. Cada fila contiene `id`, `document` (texto completo) y `metadata` (autor, título, fecha, fuente, URL, imagen). Formato parquet.
- **Autores:** Villaitech.
- **Aplicaciones posibles:** Clasificación de noticias por tema, extracción de información y entidades, resumen automático, análisis de agenda noticiosa colombiana, entrenamiento de modelos de lenguaje sobre periodismo colombiano.

---

## 12. Violence and Conflict Events in Colombia (OCHA)

- **Título:** violence-and-conflict-events-in-colombia.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/3iS/violence-and-conflict-events-in-colombia)
- **De qué trata:** Registro de **eventos de violencia y violaciones al Derecho Internacional Humanitario (DIH) en Colombia durante 2024**, compilado por **OCHA Colombia** a partir de informes de informantes clave y fuentes de prensa (El Tiempo, Alerta Paisa, Analisisurbano.org). Incluye ID, tipo de evento, fecha, título, descripción, categorías DIH ("Acciones armadas", "Ataques contra la población civil"), subcategorías (emboscada, homicidio intencional), víctimas (estado y condición), fuentes y ubicación (departamento/municipio). Licencia MIT.
- **Autores:** OCHA Colombia (publicado por la organización 3iS).
- **Aplicaciones posibles:** Análisis de tendencias de violencia, monitoreo humanitario, extracción de eventos (event extraction), NER de entidades de conflicto, sistemas de alerta temprana, investigación en paz y derechos humanos.

---

## 13. Sentencias de la Corte Constitucional de Colombia 1992-2021

- **Título:** sentencias-corte-cons-colombia-1992-2021.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/Manuel/sentencias-corte-cons-colombia-1992-2021)
- **De qué trata:** 23.750 **sentencias de la Corte Constitucional de Colombia** (1992-2021). Cada fila contiene `ID` y `Texto` (el texto completo de cada providencia). Licencia CC-BY-4.0, formato CSV (~1.68 GB).
- **Autores:** Manuel (publicado en Hugging Face).
- **Aplicaciones posibles:** Análisis de jurisprudencia colombiana, búsqueda semántica en documentos legales, clasificación de fallos, análisis de precedentes y líneas jurisprudenciales, entrenamiento de modelos legales en español, *legal tech* y RAG sobre derecho constitucional.

---

## 14. Reglamento Aeronáutico Colombiano (RAC) 2024

- **Título:** Reglamento_Aeronautico_Colombiano_2024 (corpus RAC).
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/somosnlp/Reglamento_Aeronautico_Colombiano_2024)
- **De qué trata:** ~24.500 pares **pregunta-respuesta** sobre el **Reglamento Aeronáutico de Colombia (RAC)**, extraído y curado desde el sitio de la **Aeronáutica Civil (Aerocivil)**. Incluye columnas `rac` (texto del reglamento), `pagina`, `pregunta` y `respuesta`. Creado en el hackathon de **SomosNLP** con herramientas de anotación **Argilla** y validación de especialistas en ingeniería aeronáutica. Licencia Apache-2.0, formato parquet.
- **Autores:** SomosNLP + Fundación Universitaria Los Libertadores (Alec Mauricio, Danny Stevens, Sergio Nicolas, Edison Bejarano, Nicolai Potes, Santiago Pineda). Curado por Fundación Universitaria Los Libertadores; financiado por SomosNLP, Hugging Face y Argilla.
- **Aplicaciones posibles:** QA y RAG sobre normativa aeronáutica, chatbots de asistencia a profesionales y estudiantes de aviación, *instruction tuning* de LLMs en dominios legales/técnicos, generación de texto especializado.

---

## 15. Actas del Banco de la República de Colombia

- **Título:** bank_of_the_republic_colombia.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/gtfintechlab/bank_of_the_republic_colombia)
- **De qué trata:** 1.000 oraciones (en inglés, anotadas) tomadas de las **actas de la Junta Directiva del Banco de la República de Colombia**, con **3 tareas de anotación**: *stance detection* (hawkish/dovish/neutral/irrelevant), *temporal classification* (forward-looking o no) y *uncertainty estimation* (cierto/incierto). Formato parquet (~3.000 filas con subdivisiones train/test/val). Licencia CC-BY-NC-SA-4.0.
- **Autores:** Georgia Tech Financial Services Innovation Lab (Agam Shah, Siddhant Sukhani, Huzaifa Pardawala et al.), paper *"Words That Unite The World"* (2025).
- **Aplicaciones posibles:** Análisis de comunicación de bancos centrales, detección de postura de política monetaria, clasificación temporal de declaraciones, estimación de incertidumbre, aplicaciones *fintech* y de macroeconomía computacional.

---

## 16. Google Colombian Spanish — Voz colombiana

- **Título:** google-colombian-spanish.
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/ylacombe/google-colombian-spanish)
- **De qué trata:** 4.903 muestras de **audio con su transcripción en español colombiano** (subconjuntos `female` y `male`), procedente del corpus de habla "Google" (colección con voces por idioma). Incluye columnas `audio` (con duración), `text` (transcripción) y `speaker_id`. Formato parquet (~2.08 GB).
- **Autores:** Yoach Lacombe (ylacombe), basado en los datos de Google.
- **Aplicaciones posibles:** Entrenamiento y *fine-tuning* de **text-to-speech (TTS)** y **ASR** en español colombiano, evaluación de síntesis de voz, sistemas de voz conversacional con acento colombiano, estudios fonéticos.

---

## 17. Colombian Spanish Speech Recognition Corpus (Dataocean)

- **Título:** Colombian_Spanish_Speech_Recognition_Corpus (King-ASR-711).
- **Link de acceso:** [Hugging Face](https://huggingface.co/datasets/DataoceanAI/Colombian_Spanish_Speech_Recognition_Corpus) · [Página del producto (DataoceanAI)](https://dataoceanai.com/datasets/asr/colombian-spanish-speech-recognition-corpus-conversations-mobile/)
- **De qué trata:** Corpus de **reconocimiento de voz en español colombiano**: ~200 horas de audio, 200 hablantes, grabado en **teléfono móvil**, a **44.1 kHz / 16 bits**, con transcripciones. Ideal para entrenar ASR sobre habla espontánea y acento colombiano. (El repositorio en HF está vacío; la descarga se gestiona vía el sitio de DataoceanAI.)
- **Autores:** DataoceanAI (ID de corpus King-ASR-711).
- **Aplicaciones posibles:** Entrenamiento de **reconocimiento automático del habla (ASR)** en español colombiano, transcripción de llamadas y conversaciones, asistentes de voz, subtitulado automático, evaluación de modelos acústicos en habla móvil.

---

## 18. datos.gov.co — Portal de datos abiertos de Colombia

- **Título:** Datos Abiertos Colombia (Portal Nacional de Datos Abiertos).
- **Link de acceso:** [datos.gov.co](https://www.datos.gov.co)
- **De qué trata:** Portal oficial del **gobierno colombiano** que publica miles de conjuntos de datos abiertos (CSV, JSON, API Socrata) en español: presupuesto, contratación, indicadores DANE, salud, educación, seguridad, geografía, etc. No es un corpus de PLN per se, pero es una fuente valiosa de datos tabulares y documentales para enriquecer proyectos de NLP.
- **Autores:** Gobierno de Colombia (Ministerio TIC / Datos Abiertos Colombia).
- **Aplicaciones posibles:** Construcción de conjuntos de entrenamiento para QA y RAG sobre trámites y servicios públicos, normalización de texto de dominios administrativos, NER sobre instituciones colombianas, análisis de contratación pública y transparencia, generación de datos para *benchmarks* en español.

---

## Resumen comparativo

| # | Dataset | Tarea principal | Formato | Tamaño | Licencia |
|---|---------|-----------------|---------|--------|----------|
| 1 | CoNELI | Identificación de dialectos | Texto | Tarea compartida | Investigación |
| 2 | OSCAR / mC4 / CC-100 | Pre-entrenamiento | Texto | Masivo | Abierta |
| 3 | PRESEEA | Sociolingüística | Audio+Texto | Corpus oral | Registro/académico |
| 4 | tweets_emotions_elections_colombia | Emociones | CSV | ~1.2k | MIT |
| 5 | emociones_colombia | Emociones (multi-etiqueta + CoT) | Parquet | ~1.2k | Abierta |
| 6 | SentiCorp-NT | Sentimiento | CSV | ~varios miles | Académico |
| 7 | Cyberbullying Dataset 1 | Detección de ciberacoso | CSV | 3.570 | MIT |
| 8 | Cyberbullying Dataset 2 | Detección de ciberacoso | CSV | 2.566 | MIT |
| 9 | historias_conflicto_colombia | Testimonios del conflicto | CSV | 456 | Apache-2.0 |
| 10 | colombian-conflict-SQA | QA extractivo | Parquet | ~2.9k | CC-BY-4.0 |
| 11 | colombia-news | Noticias | Parquet | ~1.0k | Abierta |
| 12 | violence-and-conflict-events-in-colombia | Eventos de conflicto/DIH | CSV | 2024 (OCHA) | MIT |
| 13 | sentencias-corte-cons-colombia-1992-2021 | Jurisprudencia | CSV | 23.750 | CC-BY-4.0 |
| 14 | Reglamento_Aeronautico_Colombiano_2024 | QA legal/técnico | Parquet | ~24.5k | Apache-2.0 |
| 15 | bank_of_the_republic_colombia | Stance/temporal/incertidumbre | Parquet | 1.000 oraciones | CC-BY-NC-SA-4.0 |
| 16 | google-colombian-spanish | TTS/ASR (voz) | Audio+Texto | 4.903 | Abierta |
| 17 | Colombian_Spanish_Speech_Recognition_Corpus | ASR | Audio | 200 h | Comercial/libre vía Dataocean |
| 18 | datos.gov.co | Datos abiertos | CSV/JSON/API | Miles de conjuntos | Abierta |
