# Contexto SOTA — Segunda entrega (Juris-PLN)

10 refs nuevas verificadas (DOI/URL con webfetch, 2026-09-08); cubren los 7 temas (a-g); 2 en español de primera mano (Samy 2021 SEPLN; BETO/modelos ES) más MultiEURLEX con español.
(a) Clasificación: MultiEURLEX exige partición cronológica por deriva conceptual temporal.
(b) Retrieval: COLIEE gana con pasajes + BM25 primero y re-ranking denso después; añadir estatutos extraídos mejora.
(c) NER: baseline español jurídico con spaCy híbrido (F1 0,93); sumar resolución de menciones.
(d) Morfología: MTLD mide diversidad léxica sin sesgo de longitud (clave en T/C/SU).
(e) Diacronía: embeddings por periodo + alineación revelan leyes del cambio semántico.
(f) Jurimetría: PageRank/eigenvector sobre el grafo de citas supera al conteo crudo de citas.
(g) Embeddings: BETO cased como baseline ES; E5 multilingüe supera a BM25 en zero-shot.
RECOMENDACIÓN retrieval CP (M2): BM25 a nivel de pasaje + expansión con artículos citados, luego re-ranking con multilingual-e5 (h-K multi-etapa estilo DoSSIER/COLIEE).
RECOMENDACIÓN NER (M3): spaCy v3 híbrido reglas+estadístico con etiqueta de referencia normativa (Samy), más coreferencia (Dozier).
RECOMENDACIÓN preprocesamiento (M4): sin lemas para modelos (BPE); lematizar solo para textometría; stopwords jurídico-procedimentales propias.
RECOMENDACIÓN morfológico (M4/M5): POS con spaCy/BETO + MTLD y TTR por tipo T/C/SU y lustro.
