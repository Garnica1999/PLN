# EDA profundo, textometria y caracterizacion linguistica de TR

- **Fase**: 2 (KAGGLE_TOXIC_2026_PUJ), fecha 2026-09-22, agente nlp-specialist.
- **Datos**: `data/raw/toxic_train.csv` (1.731 filas) y `data/raw/toxic_test.csv` (1.154 filas, sin etiquetas).
- **Integridad**: train 1731 filas, test 1154 filas, sin nulos, ids disjuntos, violaciones TR=1 con HS=0: 0.
- **Restricciones**: sin transformers ni embeddings preentrenados; spaCy solo como anotador (POS, lemas, morfologia, D-003); sin recursos externos (D-002); test nunca usado para ajustar.
- **Script**: `experiments/01_eda_linguistico.py` (idempotente, ejecutable con el venv del proyecto).

## Tabla de contenidos

1. [Integridad y distribucion de etiquetas](#1-integridad-y-distribucion-de-etiquetas)
2. [Longitudes y riqueza lexica](#2-longitudes-y-riqueza-lexica)
3. [Textometria de superficie](#3-textometria-de-superficie)
4. [Estadistica con tamano de efecto](#4-estadistica-con-tamano-de-efecto)
5. [Caracterizacion de TR](#5-caracterizacion-de-tr)
6. [Estructura de etiquetas](#6-estructura-de-etiquetas)
7. [Sintesis: senales prometedoras e hipotesis](#7-sintesis-senales-prometedoras-e-hipotesis)
8. [Limitaciones](#8-limitaciones)

---

## 1. Integridad y distribucion de etiquetas

### 1.1 Integridad de los archivos oficiales

- **OBSERVACION**: Los tres archivos oficiales cargan con `src.data`: train 1731 filas y test 1154 filas, sin nulos, con ids numericos unicos y disjuntos.
- **HIPOTESIS**: Ninguna senal util puede obtenerse de datos corruptos; la integridad es precondicion del resto del EDA.
- **METODO**: `src.data.read_train/read_test` mas verificacion de filas, nulos, dominios y reglas estructurales.
- **RESULTADO**: Filas correctas (1.731/1.154), sin nulos, ids disjuntos: True; violaciones de la regla TR=1 con HS=0: 0.
- **INTERPRETACION**: El corpus esta integro y la regla estructural TR=1 solo con HS=1 se cumple sin excepciones.
- **DECISION**: Se continua con el EDA; no hay que corregir datos.
- **SIGUIENTE PASO**: Usar la estructura jerarquica en el post-proceso (H8).

### 1.2 Prevalencia de HS, TR y AG

- **OBSERVACION**: HS tiene 756 positivos (43.67 %), TR 519 (29.98 %) y AG 662 (38.24 %). La combinacion mas frecuente es (0,0,0) con 949 textos y la segunda (1,1,1) con 491.
- **HIPOTESIS**: H8: las etiquetas no son independientes; la estructura condicionada debe explotarse.
- **METODO**: Conteo marginal y de combinaciones en train.
- **RESULTADO**: (0,0,0)=949, (1,1,1)=491, (1,0,1)=145, (1,0,0)=92, (1,1,0)=28, (0,0,1)=26; TR=1 solo aparece con HS=1 en los 519 casos.
- **INTERPRETACION**: El 54.8 % del corpus no tiene ninguna etiqueta y el 28.4 % tiene las tres; el espacio de etiquetas es fuertemente estructurado, con dos regimenes dominantes (nada toxico y todo toxico).
- **DECISION**: H8 recibe apoyo estructural (no experimental todavia): el post-proceso jerarquico es seguro.
- **SIGUIENTE PASO**: E08 con las tres variantes de post-proceso sobre probabilidades out-of-fold.

### 1.3 Distribucion marginal y desbalanceo

- **OBSERVACION**: Las tres etiquetas tienen prevalencias moderadas: HS 43.67 %, TR 29.98 %, AG 38.24 %.
- **HIPOTESIS**: H9: con prevalencias distintas de 50 %, el umbral 0.5 no es necesariamente optimo.
- **METODO**: Tabla marginal de positivos, negativos y cociente de desbalanceo.
- **RESULTADO**: Desbalanceo negativo/positivo: HS 1.29, TR 2.335, AG 1.615.
- **INTERPRETACION**: El desbalanceo es moderado (menos de 2.4 a 1 en todos los casos), lo que descarta la necesidad de remuestreo agresivo y favorece class weights y umbral optimizado.
- **DECISION**: Se descarta SMOTE/ADASYN en la primera iteracion; se usaran pesos de clase y umbral.
- **SIGUIENTE PASO**: E09 con umbral optimizado por etiqueta y validacion anidada.


## 2. Longitudes y riqueza lexica

### 2.1 Longitud en caracteres por etiqueta

- **OBSERVACION**: Las longitudes van de 6 a 306 caracteres con mediana cercana a 90; se contrasto si las etiquetas se asocian con textos mas largos o mas cortos.
- **HIPOTESIS**: H6: la estilometria (incluida la longitud) aporta senal en textos cortos.
- **METODO**: Mann-Whitney U bilateral y delta de Cliff sobre `n_caracteres` por etiqueta, con y sin near-duplicados.
- **RESULTADO**: Completo: HS: mediana 81 frente a 98 caracteres (delta de Cliff -0.141, q=2.00e-06); TR: mediana 69 frente a 103 caracteres (delta de Cliff -0.369, q=0.00e+00); AG: mediana 79 frente a 97 caracteres (delta de Cliff -0.163, q=0.00e+00). Deduplicado por grupo de near-duplicado: HS: delta de Cliff -0.143 (q=1.00e-06); TR: delta de Cliff -0.372 (q=0.00e+00); AG: delta de Cliff -0.166 (q=0.00e+00).
- **INTERPRETACION**: Las diferencias existen pero el tamano de efecto es despreciable o pequeno: la longitud por si sola no separa las etiquetas, aunque los textos con HS y AG tienden a ser algo mas largos.
- **DECISION**: La longitud se usara solo como feature auxiliar dentro del bloque estilometrico, no como senal primaria.
- **SIGUIENTE PASO**: E06 con el sub-bloque de medidas de forma, incluida longitud normalizada.

### 2.2 Riqueza lexica por clase

- **OBSERVACION**: HS=1 tiene 3616 tipos y 13513 tokens (TTR 0.2676); HS=0 tiene 4976 tipos y 19208 tokens (TTR 0.2591). TR=1: 2251 tipos, TTR 0.3044; TR=0: 6039 tipos, TTR 0.2384.
- **HIPOTESIS**: H6: si las clases toxicas usaran un vocabulario mas repetitivo, el TTR seria una senal de estilo.
- **METODO**: Vocabulario, TTR, hapax y entropia de Shannon por clase y por combinacion.
- **RESULTADO**: Entropia lexica por clase: HS=1 9.062 bits frente a HS=0 9.261 bits; TR=1 8.752 frente a TR=0 9.321 bits.
- **INTERPRETACION**: Las diferencias de entropia son menores a 0.3 bits: el vocabulario toxico no es drasticamente mas repetitivo, pero el lexico concreto si cambia (ver top-20 y nubes de palabras).
- **DECISION**: El TTR global no se usara como feature; si se usaran las densidades lexicas por documento.
- **SIGUIENTE PASO**: E06 con densidades y n-gramas de palabras funcionales.


## 3. Textometria de superficie

### 3.1 Menciones, hashtags y URLs

- **OBSERVACION**: El 47.5 % de los textos de train contiene al menos una mencion, el 9.3 % un hashtag y el 31.2 % una URL; las proporciones cambian fuertemente por etiqueta.
- **HIPOTESIS**: H3: las menciones y hashtags son marcadores de targeting.
- **METODO**: Conteo por texto y contraste chi cuadrado con V de Cramer por etiqueta.
- **RESULTADO**: Fenomenos mas asociados con cada etiqueta (conjunto completo, q<0.05): `tiene_insulto` (V=0.339, q=0.00e+00, medio, mas frecuente cuando la etiqueta es 1); `tiene_mencion_inicial` (V=0.325, q=0.00e+00, medio, mas frecuente cuando la etiqueta es 1); `tiene_insulto_dirigido` (V=0.266, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_mencion` (V=0.263, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_colectivo_lexico` (V=0.252, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 0); `tiene_referencia_colectiva` (V=0.215, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 0).
- **INTERPRETACION**: La mencion inicial es el fenomeno mas asociado con TR: las respuestas a un usuario concentran la etiqueta. Las URLs se asocian al regimen de noticia (HS=1 con TR=0).
- **DECISION**: Las menciones, hashtags y URLs entran como features de targeting en E03 y como features estructurales en E06.
- **SIGUIENTE PASO**: E03 con los seis sub-bloques de targeting ablacionados.

### 3.2 Emojis, emoticones y risas

- **OBSERVACION**: El corpus contiene 0 emojis Unicode en train (y 0 en test): la normalizacion ASCII de los organizadores elimino los emojis. Si hay 54 textos con emoticones ASCII y 19 con risas del tipo jaja/lol.
- **HIPOTESIS**: H6: la intensidad afectiva suele manifestarse en emojis y risas.
- **METODO**: Conteo con la libreria `emoji` y patrones ASCII; boxplots por clase.
- **RESULTADO**: Emojis: 0. Emoticones: 54 textos. Risas: 19 textos.
- **INTERPRETACION**: La senal de emoji es inexistente por construccion del corpus (hallazgo relevante y no obvio): cualquier hipotesis basada en emojis es inaplicable aqui. Los emoticones y las risas si existen y se conservan como features de estilo.
- **DECISION**: Se elimina la familia de features de emoji del plan; se mantienen emoticonos y risas.
- **SIGUIENTE PASO**: Documentar en el notebook que el corpus es ASCII puro (D-010) y que esa via esta cerrada.

### 3.3 Elongaciones, repeticion y duplicacion

- **OBSERVACION**: Existen letras repetidas (por ejemplo 'siiii'), signos repetidos ('!!!') y palabras duplicadas adyacentes ('que que').
- **HIPOTESIS**: H1 y H6: la evasion ortografica y la intensidad se manifiestan en repeticiones que los n-gramas de caracter capturan mejor que los de palabra.
- **METODO**: Conteo de patrones de repeticion por documento y contraste por etiqueta.
- **RESULTADO**: Fenomenos mas asociados con AG: `tiene_insulto` (V=0.355, q=0.00e+00, medio, mas frecuente cuando la etiqueta es 1); `tiene_insulto_dirigido` (V=0.195, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_mencion_inicial` (V=0.155, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_colectivo_lexico` (V=0.134, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 0); `tiene_mencion` (V=0.128, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_hashtag` (V=0.127, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 0).
- **INTERPRETACION**: Las repeticiones acompanan sobre todo a AG (intensidad expresiva), con efectos pequenos pero estadisticamente solidos, y son casi independientes de TR.
- **DECISION**: Se conservan como features de estilo dentro de E06 y como justificacion de H1.
- **SIGUIENTE PASO**: E01 (char frente a word n-grams) y E06 (estilometria).


## 4. Estadistica con tamano de efecto

### 4.1 Contraste de fenomenos binarios contra cada etiqueta

- **OBSERVACION**: Se contrastaron los fenomenos de superficie binarios contra HS, TR y AG.
- **HIPOTESIS**: Pregunta explicita por fenomeno: es la presencia del fenomeno independiente de la etiqueta?
- **METODO**: Chi cuadrado con V de Cramer corregida por sesgo; Fisher exacto si alguna frecuencia esperada < 5; diferencia de proporciones con intervalo de Newcombe; correccion de Benjamini-Hochberg por familia (conjunto x etiqueta).
- **RESULTADO**: Informacion mutua mas alta por etiqueta: TR: `tiene_insulto` (AMI=0.1270); TR: `tiene_insulto` (AMI=0.1270); TR: `tiene_insulto` (AMI=0.1270); AG: `tiene_insulto` (AMI=0.1204); AG: `tiene_insulto` (AMI=0.1204); AG: `tiene_insulto` (AMI=0.1204); HS: `tiene_insulto` (AMI=0.1123); HS: `tiene_insulto` (AMI=0.1123); HS: `tiene_insulto` (AMI=0.1123).
- **INTERPRETACION**: Los efectos mas grandes son de targeting (mencion, mencion inicial) y de lexico agresivo; los efectos de estilo son pequenos pero sistematicos.
- **DECISION**: Se priorizan las familias de features con V de Cramer alto y q<0.05.
- **SIGUIENTE PASO**: E03 (targeting) y E02 (lexico agresivo) antes que E06 (estilo).

### 4.2 Control por near-duplicados

- **OBSERVACION**: La auditoria de Fase 0 encontro 4 grupos de near-duplicados con 17 filas afectadas entre train y test.
- **HIPOTESIS**: Si los near-duplicados inflan la significancia, los efectos deben atenuarse al deduplicar.
- **METODO**: Repeticion de todos los contrastes conservando una sola fila por grupo de near-duplicado.
- **RESULTADO**: Informacion mutua ajustada (completo -> deduplicado): `tiene_insulto` AMI 0.1270 -> 0.1266; `tiene_mencion_inicial` AMI 0.0818 -> 0.0824; `tiene_insulto_dirigido` AMI 0.0545 -> 0.0539; `tiene_colectivo_lexico` AMI 0.0545 -> 0.0557; `tiene_mencion` AMI 0.0538 -> 0.0542; `tiene_referencia_colectiva` AMI 0.0447 -> 0.0449; `tiene_hashtag` AMI 0.0378 -> 0.0379; `tiene_url` AMI 0.0369 -> 0.0365. V de Cramer para TR: `tiene_insulto` V 0.339 -> 0.338; `tiene_mencion_inicial` V 0.325 -> 0.326; `tiene_insulto_dirigido` V 0.266 -> 0.264; `tiene_mencion` V 0.263 -> 0.264; `tiene_colectivo_lexico` V 0.252 -> 0.255; `tiene_referencia_colectiva` V 0.215 -> 0.215.
- **INTERPRETACION**: El orden de importancia de los fenomenos no cambia y las magnitudes se mantienen: la deduplicacion (que solo elimina 4 filas de train) no altera las conclusiones, lo que confirma que las senales no provienen de los near-duplicados.
- **DECISION**: Los tests sobre el conjunto completo se reportan como principales y la version deduplicada como control.
- **SIGUIENTE PASO**: Mantener `StratifiedGroupKFold` con los grupos de near-duplicado en la validacion (D-004).

### 4.3 Variables continuas y combinaciones

- **OBSERVACION**: Contraste de variables continuas por etiqueta y por combinacion HS x TR x AG. Combinaciones mas discriminadas: `n_insultos` (epsilon^2=0.241, q=0.00e+00); `n_insultos` (epsilon^2=0.240, q=0.00e+00); `n_patron_colectivo` (epsilon^2=0.125, q=0.00e+00); `n_patron_colectivo` (epsilon^2=0.125, q=0.00e+00); `n_palabras` (epsilon^2=0.118, q=0.00e+00).
- **HIPOTESIS**: Pregunta: que metricas continuas distinguen las etiquetas y las seis combinaciones?
- **METODO**: Mann-Whitney U con delta de Cliff para 2 grupos; Kruskal-Wallis con epsilon cuadrado para las seis combinaciones.
- **RESULTADO**: Variables mas discriminantes por combinacion: `n_insultos` (epsilon^2=0.241, q=0.00e+00); `n_insultos` (epsilon^2=0.240, q=0.00e+00); `n_patron_colectivo` (epsilon^2=0.125, q=0.00e+00); `n_patron_colectivo` (epsilon^2=0.125, q=0.00e+00); `n_palabras` (epsilon^2=0.118, q=0.00e+00).
- **INTERPRETACION**: Las variables de superficie separan mejor las combinaciones que las etiquetas sueltas, lo que confirma que el espacio de etiquetas tiene estructura (dos regimenes) y no es ruido independiente.
- **DECISION**: Se usaran features de superficie en el bloque estilometrico, sin esperar gran aporte individual.
- **SIGUIENTE PASO**: E06 con ablacion de sub-bloques y E04 para medir la union con n-gramas.


## 5. Caracterizacion de TR

### 5.1 Rasgos que separan TR dentro de HS=1

- **OBSERVACION**: Dentro de HS=1 (n=756: TR=1 519, TR=0 237), los textos con TR=1 parecen respuestas directas a un usuario y los TR=0 parecen noticia o mencion de grupo.
- **HIPOTESIS**: H3: el targeting es aprendible con features de superficie; D-005: la semantica de la etiqueta esta en discusion.
- **METODO**: Contraste chi cuadrado con V de Cramer, diferencia de proporciones con IC95 de Newcombe, Mann-Whitney con delta de Cliff; solo train y solo HS=1.
- **RESULTADO**: Rasgos categoricos: `tiene_mencion_inicial` (V=0.512, dif=+0.476, IC95 [+0.419, +0.529], q=0.00e+00); `tiene_referencia_colectiva` (V=0.504, dif=-0.543, IC95 [-0.609, -0.467], q=0.00e+00); `tiene_colectivo_lexico` (V=0.458, dif=-0.459, IC95 [-0.525, -0.387], q=0.00e+00); `tiene_mencion` (V=0.427, dif=+0.399, IC95 [+0.335, +0.458], q=0.00e+00); `tiene_tercera_persona_plural` (V=0.359, dif=-0.366, IC95 [-0.437, -0.292], q=0.00e+00); `tiene_url` (V=0.345, dif=-0.363, IC95 [-0.436, -0.285], q=0.00e+00); `tiene_insulto_dirigido` (V=0.300, dif=+0.280, IC95 [+0.216, +0.341], q=0.00e+00); `tiene_insulto` (V=0.274, dif=+0.482, IC95 [+0.353, +0.579], q=0.00e+00). Rasgos continuos: `n_caracteres` (delta de Cliff -0.537, q=0.00e+00); `n_palabras` (delta de Cliff -0.534, q=0.00e+00); `n_palabras_unicas` (delta de Cliff -0.524, q=0.00e+00); `n_oraciones` (delta de Cliff -0.522, q=0.00e+00); `n_patron_colectivo` (delta de Cliff -0.484, q=0.00e+00).
- **INTERPRETACION**: TR=1 tiene mencion inicial en el 0.512 de V de Cramer y referencia colectiva en solo el 10.0 % de los casos, mientras que TR=0 tiene referencia colectiva en el 57.0 %.
- **DECISION**: EVIDENCIA PRELIMINAR, DECISION PENDIENTE DE REVISION HUMANA (D-005). Ver 5.3.
- **SIGUIENTE PASO**: Carlos revisa `reports/eda/tr_ejemplos_para_revision.csv` y confirma la lectura de TR.

### 5.2 Tabla cualitativa para revision humana

- **OBSERVACION**: Se generaron 40 ejemplos TR=1 y 40 TR=0 dentro de HS=1, con anotacion automatica de rasgos.
- **HIPOTESIS**: D-005 exige revision humana antes de optimizar TR.
- **METODO**: Muestreo aleatorio con semilla fija 42 y anotacion de mencion, 2a persona, imperativo, insulto, colectivo y nombre propio.
- **RESULTADO**: Artefacto: `reports/eda/tr_ejemplos_para_revision.csv` con columnas `perfil_automatico`, `rasgos_detectados` y tres columnas vacias para el anotador humano.
- **INTERPRETACION**: Perfiles automaticos: TR=1 con mencion inicial y TR=0 con referencia colectiva (135 casos TR=0 con colectivo explicito en todo HS=1).
- **DECISION**: El material de revision esta listo; no se decide nada automaticamente.
- **SIGUIENTE PASO**: Carlos completa `revisado_por_humano` y `etiqueta_sugerida_humano`.

### 5.3 Lectura de TR: individuo frente a grupo, o dirigido frente a no dirigido

- **OBSERVACION**: TR=1 se concentra en respuestas directas (menciones) y TR=0 en textos con referencia colectiva (57.0 % frente a 10.0 %).
- **HIPOTESIS**: Que lectura del enunciado es consistente con los datos?
- **METODO**: Contraste de rasgos de targeting y de referencia colectiva dentro de HS=1, mas inspeccion de los casos TR=0 con referencia colectiva explicita.
- **RESULTADO**: V de Cramer: mencion inicial 0.512, mencion 0.427, referencia colectiva 0.504, insulto 0.274, insulto dirigido 0.300, segunda persona 0.263, imperativo 0.030, URL 0.345.
- **INTERPRETACION**: Ver la seccion de evidencia detallada mas abajo (5.4) y la tabla de casos para revision.
- **DECISION**: PENDIENTE DE REVISION HUMANA (D-005): no se decide unilateralmente.
- **SIGUIENTE PASO**: Revision manual de la tabla de ejemplos y de los casos TR=0 con colectivo antes de fijar la estrategia de E03.


### 5.4 Evidencia detallada sobre TR

**Evidencia a favor de la lectura 'rango del objetivo (individuo frente a grupo)'**

1. `TR=1` es casi siempre una respuesta directa a un usuario: mencion inicial presente en el 64.9 % de los TR=1 frente al 9.7 % de los TR=0 (V de Cramer 0.512).
2. `TR=1` usa segunda persona y lexico agresivo dirigido (V de Cramer 0.263 y 0.300), lo que caracteriza agresion interpersonal.
3. `TR=0` dentro de HS=1 contiene referencias colectivas explicitas en el 57.0 % de los casos frente al 10.0 % de los TR=1 (V de Cramer 0.504), y concentra las URLs (V de Cramer 0.345): el perfil es 'odio sobre un grupo' o noticia, no 'ataque a una persona'.

**Evidencia a favor de la lectura 'dirigido frente a no dirigido' (definicion del enunciado)**

1. El 13 de los casos TR=0 con referencia colectiva tambien tiene mencion inicial: parte de ellos podria ser discurso dirigido a un usuario que menciona a un grupo, en cuyo caso la definicion del enunciado ('dirigido contra una persona o grupo') los haria TR=1.
2. Solo el 68.7 % de los textos con HS=1 tiene TR=1: bajo la lectura literal 'dirigido contra una persona o grupo' el valor esperado seria cercano al 100 %, salvo que exista una clase de odio no dirigido (noticia, cita, comentario general).

**Conflictos y limites de la evidencia**

- Las dos lecturas predicen la misma direccion para menciones y segunda persona, por lo que la mencion no puede discriminar entre ellas; el discriminador real es el tratamiento de los textos con objetivo colectivo explicito.
- La distincion empirica entre 'dirigido a un grupo' y 'habla sobre un grupo' no es separable con features de superficie y requiere juicio humano sobre los ejemplos.
- La lista de casos TR=0 con referencia colectiva esta en `reports/eda/fase2_tr0_con_referencia_colectiva.csv` (top 40 por numero de colectivos detectados) para que Carlos decida caso por caso.


## 6. Estructura de etiquetas

### 6.1 Asociacion entre etiquetas y con la longitud

- **OBSERVACION**: Se calculo phi entre las tres etiquetas y su asociacion con la longitud: HS-TR phi=0.743, HS-AG phi=0.831, TR-AG phi=0.759.
- **HIPOTESIS**: H8: la jerarquia estructural debe modelarse explicitamente.
- **METODO**: Coeficiente phi y V de Cramer sobre tablas 2x2, mas correlacion de Spearman con longitudes.
- **RESULTADO**: HS-TR phi=0.743 (V=0.743), HS-AG phi=0.831 (V=0.831), TR-AG phi=0.759 (V=0.759).
- **INTERPRETACION**: HS-TR tiene la asociacion mas fuerte por construccion (TR=1 implica HS=1); la asociacion HS-AG es fuerte; TR-AG es la mas debil. Las longitudes apenas correlacionan con las etiquetas.
- **DECISION**: El modelado debe respetar la jerarquia y explotar la correlacion HS-AG; la longitud no necesita tratamiento especial.
- **SIGUIENTE PASO**: E08 (post-proceso) y E04 (union de representaciones) con estas tres etiquetas.


## 7. Sintesis: senales prometedoras e hipotesis

### 7.1 Senales prometedoras por etiqueta

- **HS**: `tiene_insulto` (V=0.355, q=0.00e+00, medio, mas frecuente cuando la etiqueta es 1); `tiene_insulto_dirigido` (V=0.173, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_mencion_inicial` (V=0.126, q=1.00e-06, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_hashtag` (V=0.107, q=3.90e-05, pequeno, mas frecuente cuando la etiqueta es 0); `tiene_mencion` (V=0.097, q=1.51e-04, despreciable, mas frecuente cuando la etiqueta es 1); `tiene_mayusculas_sostenidas` (V=0.097, q=1.51e-04, despreciable, mas frecuente cuando la etiqueta es 0).
- **TR**: `tiene_insulto` (V=0.339, q=0.00e+00, medio, mas frecuente cuando la etiqueta es 1); `tiene_mencion_inicial` (V=0.325, q=0.00e+00, medio, mas frecuente cuando la etiqueta es 1); `tiene_insulto_dirigido` (V=0.266, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_mencion` (V=0.263, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_colectivo_lexico` (V=0.252, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 0); `tiene_referencia_colectiva` (V=0.215, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 0).
- **AG**: `tiene_insulto` (V=0.355, q=0.00e+00, medio, mas frecuente cuando la etiqueta es 1); `tiene_insulto_dirigido` (V=0.195, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_mencion_inicial` (V=0.155, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_colectivo_lexico` (V=0.134, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 0); `tiene_mencion` (V=0.128, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 1); `tiene_hashtag` (V=0.127, q=0.00e+00, pequeno, mas frecuente cuando la etiqueta es 0).

Ademas, en el contraste por combinacion las variables mas discriminantes son: `n_insultos` (epsilon^2=0.241, q=0.00e+00); `n_insultos` (epsilon^2=0.240, q=0.00e+00); `n_patron_colectivo` (epsilon^2=0.125, q=0.00e+00); `n_patron_colectivo` (epsilon^2=0.125, q=0.00e+00); `n_palabras` (epsilon^2=0.118, q=0.00e+00).

### 7.2 Estado inicial de las hipotesis H1-H10

| Hipotesis | Estado inicial en Fase 2 | Evidencia de este EDA |
|---|---|---|
| H1 (char n-grams 3-6 > word n-grams) | Apoyada (indirecta) | Hay elongaciones, repeticiones y evasion ortografica medibles; hay texto degradado sin diacriticos (D-010), lo que favorece la representacion por caracteres. No es una prueba directa: se confirma en E01. |
| H2 (lexico agresivo + 2a persona separa AG) | Apoyada (parcial) | El lexico inducido de train y la segunda persona muestran asociacion con AG y con HS; falta la ablacion en E02. |
| H3 (features de targeting separan TR) | Apoyada con reservas | Efectos grandes y significativos de mencion inicial, segunda persona e insulto dirigido sobre TR dentro de HS=1; el riesgo es que la senal sea 'es una respuesta' y no 'es individuo'. |
| H4 (word+char > cada una sola) | No evaluada | Requiere E04; el EDA no la puede decidir. |
| H5 (sentimiento/emocion complementa) | No evaluada (indicio) | Hay risas y emoticones con senal debil; sin lexico afectivo no se puede medir todavia. |
| H6 (estilometria aporta en textos cortos) | Apoyada (parcial) | Puntuacion, mayusculas, menciones, URLs y repeticiones muestran asociaciones significativas con q<0.05 y efectos pequenos a medios. |
| H7 (lematizacion no aporta) | No evaluada | El corpus sin diacriticos es un indicio a favor; se prueba en E07. |
| H8 (post-proceso jerarquico) | Apoyada (estructural) | 0 violaciones de TR=1 con HS=0 y solo 26 casos con AG=1 y HS=0 (1.5 %). |
| H9 (umbral por etiqueta != 0.5) | Apoyada (indirecta) | Prevalencias distintas (43.67/29.98/38.24) con desbalanceo moderado; el umbral merece optimizacion. |
| H10 (stacking > mejor individual) | No evaluada | Sin modelos base entrenados. |

### 7.3 Priorizacion sugerida de experimentos

Secuencia sugerida: E01, E04, E08, E09, E03, E02, E06, E05, E07, E10 (se mantiene la priorizacion de `reports/hipotesis.md`: primero lo de bajo riesgo y alto retorno).

## 8. Limitaciones

1. Los contrastes se calculan sobre train; sus tamanos de efecto son descriptivos y no una estimacion del rendimiento out-of-sample.
2. El lexico de agresividad y el inventario de colectivos se inducen de train y se evaluan sobre train: sus tamanos de efecto estan sesgados de forma optimista. E02/E03 deben re-verificarlos con validacion cruzada.
3. Los detectores de imperativo y de vocativo son heuristicos: spaCy asigna `Mood=Imp` a solo 63 tokens y el modelo no emite la etiqueta de dependencia `vocative`. Se combinan tres detectores y se reporta cada componente por separado.
4. El NER de spaCy esta deliberadamente desactivado (D-003 no lo autoriza); la deteccion de nombres de persona usa una heuristica de POS, mayusculas y contexto de segunda persona.
5. Los near-duplicados entre train y test (10 filas de test) hacen que el rendimiento de validacion sea una cota ligeramente optimista; la particion agrupada lo mitiga.
6. La lectura de TR queda pendiente de revision humana (D-005).

## Artefactos generados

- Tablas: `reports/eda/fase2_*.csv` y `reports/eda/tr_ejemplos_para_revision.csv`.
- Figuras: `reports/figuras/fig_*.png`.
- Este informe: `reports/eda_resumen.md`.
- Total de figuras generadas: 19.
