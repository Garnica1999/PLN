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
   no aplica la lista de config/tfidf_config.json; la longitud de documento
   de BM25 se mide en terminos del vocabulario (unigramas+bigramas,
   avgdl 1.466,26), no en lemas de contenido (1.007,5), como documenta la
   nota metodologica de la Sec 03b.

## Conclusiones de los puntos 4 y 5 (BM25, Rocchio y comparacion de corpus)

1. **BM25 y TF-IDF ordenan parecido pero no igual.** Sobre las 2.484
   queries con match, BM25 (k1=1,5, b=0,75) comparte con el TF-IDF base el
   57,0 % del top-1 y el 65,7 % del CONJUNTO del top-5 (la posicion 5
   exacta solo coincide en el 11,8 %): la senal lexica del conflicto es
   comun, pero la saturacion de tf, la normalizacion por longitud y el IDF
   de Robertson-Sparck Jones reordenan el top-5 sin cambiar la mayoria del
   inventario recuperado. El top-1 sigue concentrado en TESTIMONIAL
   (74,9 % sobre las 2.486 entrevistas; 75,0 % sobre las 2.484 con match,
   frente a 76,77 % del base): el sesgo de registro oral es del dominio, no
   de la funcion de puntuacion.

2. **Rocchio re-rankea, no compite desde cero.** Con pseudo-relevancia
   (top-5 = D_r, bottom-5 = D_nr; alpha=1, beta=0,75, gamma=0,15) conserva
   el 70,1 % de los top-1 del base y cambia el 29,9 % restante; su CONJUNTO
   de top-5 es identico al del base en el 100,0 % de las queries (reordena,
   no sustituye; la posicion 5 exacta coincide en el 48,8 %). El efecto
   dominante es beta (expansion hacia el centroide de los 5 primeros): los
   bottom-5 casi siempre tienen score 0 (sin terminos en comun), asi que
   gamma apenas aporta. Sin gold es pseudo-relevance feedback y hereda el
   *query drift* del ranking base.

3. **La comparacion es de acuerdo, no de acierto.** Sin juicios de
   relevancia no hay P@k real para BM25/Rocchio; los ejemplos cualitativos
   de la Sec 03b (coincidencias y divergencias con tomo y titulo) y los 30
   pares etiquetados de la Sec 03 son la evidencia humana disponible. Los
   tres metodos comparten el sesgo TESTIMONIAL; el acuerdo alto base-BM25
   y base-Rocchio es consistencia, no correccion.

4. **El heatmap (Sec 03c) valida la asignacion cruzada y visualiza el
   sesgo.** Agrupando las entrevistas por el tomo de su top-1 BM25, la
   media de similitud TF-IDF de la diagonal (0,0725) supera a la de fuera
   de la diagonal (0,0608) y en los 10 grupos la diagonal esta por encima
   de la media de su fila: la asignacion por top-1 no es ruido. Pero el
   maximo de fila coincide con la diagonal solo en 3 de 10 grupos (en 6
   de los 7 restantes cae en la columna TESTIMONIAL y en HALLAZGOS cae en
   VIOLACIONES): la afinidad cruzada hacia el registro oral domina el
   mapa. El grupo CONVOCATORIA (n=2) es estadisticamente fragil.

5. **El cuadro de vinculos es la evidencia cualitativa del heatmap.**
   Las 30 parejas (3 por tomo) muestran que las afinidades altas
   corresponden a solapes tematicos reconocibles (secuestro, masacres,
   pueblos etnicos, violencia sexual), no a coincidencias vacias; y que
   las afinidades cruzadas entre tomos analiticos existen pero son menores
   que la diagonal, como espera el diseno del informe por generos.

## Mejoras futuras

1. **Embeddings multilingues** (SBERT/multilingual E5 o similares) para el
   matching semantico: atacaria el problema diagnosticado de especificidad
   (7/30 top-1 parciales con mejor match en r2-r3) mejor que ampliar la
   parada, porque discrimina por sentido y no solo por solape lexico.

2. **Gold estandar externo**: etiquetado por expertos del dominio (o una
   segunda cohorte de estudiantes) sobre una muestra mayor, para pasar de
   metricas proxy y de acuerdo entre rankings a P@k con acuerdo
   inter-etiquetador (kappa); con ese gold, Rocchio podria usar
   pseudo-relevancia real (D_r/D_nr por juicio) y compararse con P@1.

3. **BM25 tuneado (k1 y b)**: el grid actual usa los valores estandar de
   la literatura; una busqueda k1 in [0,9, 2,0] y b in [0,3, 1,0]
   (Optuna o grid con validacion) puede ajustar la normalizacion por
   longitud al corpus (segmentos de 420 a ~1.400 tokens) y mover el
   ranking sin cambiar el vocabulario.

4. **Reranking supervisado del top-k**: un cross-encoder (o un clasificador
   ligero con features de solape y tomo) entrenado con la muestra
   etiquetada puede reordenar el top-5 de cualquiera de los tres metodos;
   es la via natural para convertir el acuerdo actual en mejora de P@1.

5. **Enrutamiento por tomo**: un clasificador previo tomo-query (o un
   reranker por tomo) podria corregir el prior TESTIMONIAL cuando la query
   pide un hecho o poblacion especifica (masacres, raizal, exilio).
"""
