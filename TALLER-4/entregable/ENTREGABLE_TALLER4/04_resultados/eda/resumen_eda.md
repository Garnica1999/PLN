# Resumen EDA (Fase 5) - corpus CEV

- Fecha: 2026-09-04 | Script: scripts/03_eda.py | Corpus: 917 segmentos de libro (10 tomos) y 2.486 entrevistas.

## 1. Tamano y longitudes
1. Libros: 917 docs, 928,253 lemas de contenido (2,624,040 tokens crudos spaCy), 33,277 tipos; media 1012 lemas/segmento (min 117, max 1765).
2. Entrevistas: 2486 docs, 9,702,534 lemas de contenido (40,346,344 tokens crudos), 109,391 tipos; media 3903 lemas/entrevista (min 0, max 45278); chars normalizados min-max 0-824,072 (el texto crudo iba de 3 a 850K).
3. Asimetria extrema en entrevistas (media muy superior a la mediana; p99 recorta la cola hasta ~50K lemas). 2 docs vacios legitimos documentados en longitudes_por_doc.csv (no se eliminan).

## 2. Diversidad lexica (lemas de contenido)
4. TTR muestreado n=1000: libros media 0.609 vs entrevistas 0.514.
5. Yule's K: libros 32 vs entrevistas 39 (menor = mas diverso): la oralidad transcrita es lexicamente mas repetitiva (ruido sistematico, Moreno-Sandoval 2024).

## 3. Frecuencias y nubes
6. Top-10 unigramas libros: armado, víctima, grupo, persona, violencia, comisión, mujer, conflicto, año, entrevista. Entrevistas: gente, llegar, año, pasar, cosa, venir, salir, conocido, persona, querer (verbo-narracion oral).
7. Nubes uni/bi/trigramas: 6 figuras (3 por corpus) en imagenes/; tablas top-30 en resultados/eda/frecuencias_*.csv.

## 4. Near-duplicates entre entrevistas
8. 345 pares con Jaccard de shingles(5) > 0.85 (de 368 pares candidatos LSH verificados con Jaccard exacto). Son casi-duplicados reales del corpus: se REPORTAN sin eliminar (near_duplicates_entrevistas.csv).

## 5. Comparacion lexica libros <-> entrevistas
9. Vocabulario: libros 33,235 vs entrevistas 109,391 tipos; compartidos 25,851 (Jaccard 0.221). El vocabulario de libros cubre el 95.7 % de los tokens de entrevistas y el de entrevistas el 98.3 % de los de libros.
10. Distintivos de libros (top-5): transmedia (8.0), externas (7.8), microdato (7.4), catalogación (7.4), macroterritorio (7.0).
11. Distintivos de entrevistas (top-5): anonimo (9.1), organizacion_social (8.5), organizacion_publica (8.2), conocida (7.1), institucion_educativa (7.1).
12. Hallazgo de calidad en libros: fragmentos de palabras partidas por el guion de linea del PDF (p. ej. 'cion' 542 oc, 'nes' 189) y ruido bibliografico ('ibid.' 852 oc); se excluyen del analisis comparativo y se listan en config/tfidf_config.json para que la Fase 6 decida su tratamiento.
13. Siglas asimetricas: 'far' (519 oc) es distintivo de entrevistas (testimonios: 'las FAR'); en libros predomina 'farc': desajuste lexico del registro oral a tener en cuenta en el matching.

## 6. Tesauro CEV (ADR-004, lectura cualitativa)
14. 14 de 50 distintivos tienen etiqueta SKOS (prefLabel/altLabel) en el tesauro; ejemplos:
    - 'romaní': romaní ...
    - 'rutas': rutas del conflicto ...
    - 'muse': asociación de víctimas sobrevivientes por map y muse del conflicto armado de san juan de arama meta - caminemos ...
    - 'cnmh': centro nacional de memoria histórica - cnmh ...

## 7. Congelaciones para la Fase 6
- config/stopwords_habla.txt: lista final con evidencia (muletillas_evidencia.csv y muletillas_decision.csv); parada en consumo = spacy STOP_WORDS + esta lista.
- config/lemas_correcciones.json: 62 correcciones de lemas (corruptos del lematizador, formas de 2a persona sin lematizar y truncamientos orales; ver el archivo y lemas_anomalos_candidatos.csv).
- config/tfidf_config.json: min_df=2, max_df=0.85, ngram_range=(1,2), sublinear_tf=True, norm=l2 (ADR-005 confirmado con cifras del EDA).
