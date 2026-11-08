# Calidad de respuesta del RAG (v1)

Fecha: 2026-11-01T01:13:08-05:00. Casos: 30 (28 validos).
Generador: `qwen3.8:diario-8k-q3_k_s`. Juez: `alia-legal:latest` (gold-LLM, no humano).

## Metricas

| Metrica | Media | IC 95 % | Min | Max |
|---|---:|---|---:|---:|
| fidelidad | 0.7749 | [0.7067, 0.8442] | 0.5 | 1.1081 |
| relevancia | 1.2857 | [1.0, 1.8571] | 1.0 | 9.0 |
| precision_contexto | 0.7742 | [0.7067, 0.8387] | 0.5 | 1.0644 |

## Detalle

| Caso | Fidelidad | Relevancia | Precision contexto | Respuesta (extracto) |
|---|---:|---:|---:|---|
| alc-01 | 0.8 | 1.0 | 0.8 | Para la sentencia **T-293/97** (id 2195), los artículos vinculados son:

**Citados explíci |
| alc-02 | 0.8 | 1.0 | 0.8 | Evidencia recuperada por las herramientas. Sentencia analizada: T-760-2008 (id 11705). Art |
| alc-04 | 0.8 | 1.0 | 0.8 | La SU-480/97 (id 2725) cita 46 sentencias, todas resueltas en el corpus. Entre las princip |
| alc-07 | 0.8 | 1.0 | 0.8 | Evidencia recuperada por las herramientas. Sentencia analizada: T-760-2008 (id 11705). Art |
| alc-09 | 0.8 | 1.0 | 0.8 | La T-293/97 cita explícitamente seis artículos de la Constitución:

- **Art. 11**: derecho |
| alc-10 | 0.95 | 1.0 | 0.95 | La sentencia SU-480/97 (canon detectado SU-480-1997, id 2725) menciona explícitamente esto |
| alc-12 | 0.8 | 1.0 | 0.8 | Evidencia recuperada por las herramientas. Sentencia analizada: C-543-1992 (id 224). Artíc |
| alc-13 | 0.8 | 1.0 | 0.8 | Para la sentencia **T-293-1997** (id 2195) se vincularon los siguientes artículos de la Co |
| alc-14 | 0.8 | 1.0 | 0.9 | La sentencia con id 11705 corresponde al canon **T-760-2008**.

**Artículos citados explíc |
| alc-15 | 0.8 | 1.0 | 0.8 | Evidencia recuperada por las herramientas. Sentencia analizada: C-355-2006 (id 13921). Art |
| alc-16 | 0.8 | 1.0 | 0.8 | El documento 2725 (SU-480-1997) cita 46 sentencias, todas resueltas en el corpus. Entre la |
| alc-17 | 0.5 | 1.0 | 0.5 | Evidencia recuperada por las herramientas. Sentencia analizada: C-543-1992 (id 224). Artíc |
| alc-18 | 0.5 | 1.0 | 0.5 | Evidencia recuperada por las herramientas. Artículos de la Constitución vinculados: 241 (r |
| alc-19 | 0.6 | 1.0 | 0.8 | Para la sentencia T-760-2008 (id 11705):

**Artículos extraídos (8):** 86, 241, 1, 94, 50, |
| alc-20 | 1.1081 | 9.0 | 1.0644 | Evidencia recuperada por las herramientas. Sentencia analizada: T-293-1997 (id 2195). Artí |
| alc-21 | 0.5 | 1.0 | 0.5 | El artículo 86 de la Constitución Política establece que toda persona tendrá acción de tut |
| alc-22 | 0.5 | 1.0 | 0.5 | El artículo 13 de la Constitución Política está clasificado en el bloque de **derechos fun |
| alc-24 | 1.0865 | 1.0 | 1.0 | El artículo 93, según el corpus, se ubica en el bloque temático de protección y aplicación |
| alc-25 | 0.8 | 1.0 | 0.8 | El artículo 44 de la Constitución Política de Colombia consagra los derechos fundamentales |
| alc-27 | 0.5 | 1.0 | 0.5 | El artículo 241 (bloque: rama_judicial) establece que a la Corte Constitucional se le conf |
| alc-28 | 1.0 | 1.0 | 1.0 | El artículo 11 de la Constitución Política de Colombia, según el corpus, establece:

"El d |
| alc-30 | 0.9243 | 1.0 | 0.9035 | Evidencia recuperada por las herramientas. Artículos de la Constitución vinculados: 241 (r |
| alc-32 | 0.75 | 1.0 | 0.75 | Evidencia recuperada por las herramientas. Artículos de la Constitución vinculados: 86 (pr |
| alc-33 | error |  |  | Expecting ',' delimiter: line 20 column 3 (char 302) |
| alc-34 | 0.667 | 1.0 | 0.5 | Sí. La validación (heurística M0) la reconoce como providencia de la Corte Constitucional: |
| alc-35 | 0.5 | 1.0 | 0.5 | Para la sentencia T-293/97 (id 2195), los artículos de la Constitución vinculados son:

** |
| alc-37 | 0.9 | 1.0 | 0.9 | La sentencia SU-480/97 es de **1997** (canon detectado: SU-480-1997, id 2725).

Artículos  |
| alc-38 | 1.0 | 1.0 | 1.0 | El artículo 86 de la Constitución Política, que consagra la acción de tutela, pertenece al |
| alc-39 | 0.91 | 1.0 | 0.91 | Evidencia recuperada por las herramientas. Artículos de la Constitución vinculados: 241 (r |
| alc-40 | error |  |  | sin contexto recuperado |
