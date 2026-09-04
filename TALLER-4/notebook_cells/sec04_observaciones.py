# %% [markdown]
"""# Sec 04 — Observaciones, limitaciones y mejoras futuras

## Limitaciones del trabajo

1. **No hay gold estandar.** La CEV no publica juicios de relevancia
   query->segmento; la evaluacion es una proxy: etiquetado humano-proxy de
   30 pares (con contexto ampliado del corpus) con etiquetas
   {relacionado, parcial, no}. Las metricas P@1/P@3 son indicativas y
   cargan el sesgo del etiquetador; ampliar la muestra o usar una segunda
   cohorte cambiaria los numeros.

2. **El lematizador spaCy (es_core_news_md) falla en formas de la oralidad**
   colombiana: voseo y preteritos irregulares sin lematizar (p. ej.
   'atrevieron' -> 'atreveir', 154 ocurrencias de 10,9 M de lemas). La
   solucion fueron 62 correcciones (config/lemas_correcciones.json)
   aplicadas EN CONSUMO, no sobre el corpus: el archivo *_lemas.txt queda
   intacto y cualquier consumo futuro debe reaplicarlas (como hace este
   notebook via `aplicar_parada()`).

3. **Near-duplicates de entrevistas documentados, no eliminados.** 345 pares
   con Jaccard de shingles(5) > 0,85 (526 entrevistas unicas, ~21 % del
   corpus) se reportan en resultados/eda/near_duplicates_entrevistas.csv
   pero se conservan por integridad del corpus oficial; el ranking puede
   devolver segmentos casi identicos para queries equivalentes.

4. **PII hablada y etica de datos con testimonios.** La normalizacion
   estructural mapea los anonimizados del transcriptor a 8 categorias
   canonicas (ADR-002), pero el intermedio
   `data/corpus/entrevistas_normalizadas.jsonl` conserva nombres y datos
   personales que el transcriptor dejo en claro (PII hablada): es un
   archivo de trabajo, no de distribucion. Los corpus finales
   (`data/corpus/preprocesado/*_lemas.txt` y `*_tokens.txt`) no contienen
   superficie PII (solo lemas y categorias). Los testimonios pertenecen a
   victimas de la CEV: el uso debe ser exclusivamente academico, sin
   re-identificacion ni publicacion de fragmentos atribuibles.

5. **Otras:** 2 entrevistas vacias legitimas (sin texto de origen) que
   quedan sin filas de ranking; TF-IDF es bag-of-words (ignora orden y
   sintaxis); ruido tipografico de los PDF (palabras partidas, 'ibid.',
   42 tokens) excluido en el EDA pero presente en los lemas si otro consumo
   no aplica la lista de config/tfidf_config.json.

## Mejoras futuras (incluyen los puntos 4 y 5 del taller)

1. **Punto 4 del taller: BM25 (Okapi) y Rocchio.** Sustituir o combinar el
   tf-idf con BM25 (ponderacion por longitud de documento) y evaluar
   relevance feedback de Rocchio sobre la muestra etiquetada como
   pseudo-relevancia; comparar P@1/P@3 proxy contra el base.

2. **Punto 5 del taller: heatmap tomo x tomo.** Visualizar la confusion
   entre el tomo real de la entrevista y el tomo del top-1 (los datos ya
   existen en resultados/ranking_top5.csv y la distribucion TESTIMONIAL
   del 76,77 % pide este analisis por tomo y por periodo).

3. **Embeddings multilingues** (SBERT/multilingual E5 o similares) para el
   matching semantico: atacaria el problema diagnosticado de especificidad
   (7/30 top-1 parciales con mejor match en r2-r3) mejor que ampliar la
   parada, porque discrimina por sentido y no solo por solape lexico.

4. **Enrutamiento por tomo**: un clasificador previo tomo-query (o un
   reranker por tomo) podria corregir el prior TESTIMONIAL cuando la query
   pide un hecho o poblacion especifica (masacres, raizal, exilio).

5. **Gold estandar externo**: etiquetado por expertos del dominio (o una
   segunda cohorte de estudiantes) sobre una muestra mayor, para pasar de
   metricas proxy a P@k con acuerdo inter-etiquetador (kappa).
"""
