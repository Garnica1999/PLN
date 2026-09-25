# Procedencia del corpus — `tweet-age-2026-puj`

> Informe de identificación del corpus de la competencia académica privada `tweet-age-2026-puj`
> (host `NLP_Course_PUJ`). Fecha: 2026-09-22. Alcance: SOLO lectura de `data/raw/*.csv` y consultas
> web. No se descargó ningún dataset externo ni se modificó código del proyecto.

## 1. Huella del corpus (mediciones propias, reproducibles)

| Rasgo | Evidencia medida | Comentario |
|---|---|---|
| Tamaño | train 169.336 filas · test 112.892 filas (ratio 60/40) | splits separados, no aleatorios 70/15/15 |
| `id` | train `1..169336`; test `1..112892`, ambos empezando en 1 | **IDs re-indexados de forma independiente**: NO son IDs de tweet ni permiten rastrear el original |
| Formato | BOM UTF-8 + separador `;` + fin de línea CRLF; `age_submission.csv` usa coma | export de herramienta ofimática/BD, distinto formato para el submission |
| Ancho fijo | **99,88 % (train) y 99,89 % (test)** de los textos miden exactamente 140 caracteres en bruto; mediana tras `rstrip` = 93 | compatibles con una columna `CHAR(140)` (MySQL rellena con espacios a la derecha y trunca a 140) |
| Acentos | 99,97 % de los textos son ASCII puro (en una muestra de 20.000 solo aparecen 4 caracteres `¿`) | **el corpus fue normalizado a ASCII: los acentos se perdieron** (`medellín`=0 vs `medellin`=755; `bogotá`=0 vs `bogota`=1620; `oposición`=0 vs `oposicion`=265) |
| Longitud | mediana 13 tokens (media 13,0); mediana de caracteres útiles 93 | tweets cortos, típicos de 140 caracteres |
| Retuits | 36,3 % de los textos empiezan por `RT @` (idéntico en train y test: 36,3 %) | retuits conservados, no eliminados |
| Superficie | 39,1 % contienen `t.co/` · 41,6 % contienen `http` · 63,9 % tienen `@` · 20,2 % tienen `#` | señal superficial muy rica (forma > contenido) |
| Idioma | español latinoamericano + retuits en inglés; aparecen referencias a Argentina (`@LaPlataMLP`, `#BuenosAires`), Colombia y Venezuela | no es un corpus de un solo país |
| Ventana temporal (inferida) | `colombia` 3,53 % · `venezuela` 1,55 % · `maduro` 1,14 % · `santos` 1,02 % · `uribe` 0,78 % · `farc` 0,64 % · `petro` 0,37 % · `chavez` 0,36 % · `odebrecht` 0,17 % · `zuluaga` 0,13 % · `duque` 0,09 %; **`pandemia` 2 casos, `covid` 1, `coronavirus` 0, `chatgpt` 0** | perfil léxico consistente con **≈2016-2017** (plebiscito FARC oct-2016, escándalo Odebrecht 2016-2017) y ausencia de eventos posteriores a 2019 |
| Clases | 6 rangos casi uniformes: 13-17 17,0 % · 18-24 16,9 % · 25-34 16,7 % · 35-49 16,7 % · 50-64 16,6 % · 65-xx 16,0 % | balance casi perfecto: improbable en un muestreo orgánico de Twitter → **remuestreo/downsampling por clase o etiquetado de un panel curado** |
| Metadatos | ninguno (sin `user_id`, fecha, bio, seguidores, geolocalización) | impide cualquier agrupación por autor (ver D-003) |
| Duplicados | 1,26 % de filas de test aparecen literalmente en train; 4,24 % en clústeres de firma de 60 caracteres; 0,80 % de duplicados exactos intra-train | pool único de tweets **partido en train/test sin agrupar por autor** |

Comandos de referencia (solo lectura) usados para esta huella:
`head -c 500 data/raw/age_train.csv | cat -A`, y un script puntual en `/tmp/opencode` con
`pandas.read_csv(sep=";", encoding="utf-8-sig")` que reportó las cifras de la tabla.

## 2. Candidatos públicos evaluados

### 2.1 PAN Author Profiling 2013 (Rangel, Rosso, Koppel, Stamatatos & Inches)
- **Esquema de edad**: 3 clases — `10s (13-17)`, `20s (23-27)`, `30s (33-47)`.
- **Idioma / tamaño**: inglés y español; corpus pequeño de documentos (blogs/social media, con estructura por autor), no tweets sueltos.
- **Licencia / acceso**: notas de CLEF 2013, CEUR-WS Vol-1179; corpus descargable desde Webis.
- **Evidencia**: página oficial de la tarea (`https://pan.webis.de/clef13/pan13-web/author-profiling.html`) y overview (`https://ceur-ws.org/Vol-1179/CLEF2013wn-PAN-RangelEt2013.pdf`).
- **Veredicto**: **NO COINCIDE**. 3 clases (no 6), textos de blog sin truncado a 140 caracteres, y la clase 13-17 aparece en un rango distinto (10s).

### 2.2 PAN Author Profiling 2014 (Rangel et al.)
- **Esquema de edad**: 5 clases — `18-24, 25-34, 35-49, 50-64, 65-xx` (**sin 13-17**).
- **Idioma / tamaño**: inglés y español; blogs, tweets, textos de social media y reseñas de hotel.
- **Licencia / acceso**: CEUR-WS Vol-1180. **Solo se distribuyen URLs de los tweets** por política de Twitter (declarado en la página oficial).
- **Evidencia**: `https://pan.webis.de/clef14/pan14-web/author-profiling.html` (guarda además: "we cannot provide tweets directly, but only URLs referring to them").
- **Veredicto**: **NO COINCIDE**. Falta la clase 13-17 y el corpus público no contiene texto de tweets (solo URLs): no puede ser el origen de un dataset con texto completo.

### 2.3 PAN Author Profiling 2015-2016 (ediciones en Twitter con español)
- **Esquema de edad**: PAN 2015 usa **4 clases** `18-24, 25-34, 35-49, 50-xx`; PAN 2016 usa **5 clases** `18-24 ... 65-xx` (cross-genre). En ninguna aparece 13-17.
- **Idioma / tamaño**: PAN 2015 EN/ES (edad y género) + IT/NL; PAN 2016 EN/ES/NL. Estructura **por autor** (perfil con varios tweets), no tweets sueltos.
- **Licencia / acceso**: CEUR-WS Vol-1391 y Vol-1609; mismos límites de política de Twitter.
- **Evidencia**: páginas oficiales de las tareas y overviews verificados en `pan.webis.de/publications.html`.
- **Veredicto**: **NO COINCIDE** (esquema de clases sin 13-17 + estructura por autor).

### 2.4 PAN Author Profiling 2017-2019
- **Esquema**: 2017 género + variedad de lengua; 2018 género multimodal; 2019 bots + género. **Ninguna edición 2017-2019 predice edad.**
- PAN 2019 sí usa Twitter EN/ES con 100 tweets por autor **sin limpieza (conserva RT)**, en XML por autor.
- **Veredicto**: **NO COINCIDE** (sin etiqueta de edad; formato por autor).

### 2.5 TwiSty (Verhoeven, Daelemans & Plank, LREC 2016)
- **Esquema**: corpus multilingüe de estilometría con **español incluido**, perfilado de género y personalidad, con metadatos de edad entre los atributos anotados (los rangos exactos **no se pudieron verificar** en esta consulta).
- **Idioma / tamaño**: 6 idiomas; organización **por usuario** (perfil con muchos tweets), no filas de tweets independientes.
- **Licencia / acceso**: LREC 2016, DOI `10.63317/3yne5b7ve4p2` (verificado en Crossref); licencia de investigación.
- **Veredicto**: **NO COINCIDE** en formato y granularidad (perfiles frente a tweets sueltos) y **INDETERMINADO** en el esquema exacto de rangos de edad.

### 2.6 Blog Authorship Corpus (Schler et al. 2006; Argamon et al. 2007)
- **Esquema de edad**: **6 rangos idénticos a los de esta competencia** (`13-17, 18-24, 25-34, 35-49, 50-64, 65+`).
- **Idioma / tamaño**: inglés; blogs de blogger.com (del orden de cientos de miles de posts, cifras **no verificadas** en la fuente primaria: la página del corpus no respondió a la consulta).
- **Veredicto**: **NO COINCIDE** como origen (idioma, género textual, formato), pero **COINCIDE en el esquema de etiquetado**: el patrón de 6 rangos con 13-17 como primer cubo procede de esa línea de trabajo (blogs), y no del esquema PAN (que empieza en 18-24 salvo en 2013 con 3 clases).

### 2.7 Corpus y datasets de tweets en español: TASS / MEX-A3T / IberEval / COSET
- Estas colecciones etiquetan sentimiento, agresividad, variedad dialectal o stance (incluido el español latinoamericano), **no edad del autor**.
- **Veredicto**: **NO COINCIDE** (tarea distinta). No se localizó ningún corpus español de Twitter con etiquetas de edad de ~282.000 filas.

### 2.8 Kaggle y Hugging Face ("age prediction from tweets")
- **Método**: búsqueda en la API pública de Hugging Face (`/api/datasets?search=...`) con los términos `age twitter`, `age_tweets` y `demographics`: **sin coincidencias relevantes** (solo datasets demográficos tabulares, de sesgo de modelos, o sintéticos). Kaggle no permite búsqueda sin credenciales desde este entorno.
- **Veredicto**: **INDETERMINADO**. La búsqueda negativa es evidencia débil: **no prueba la inexistencia** de un dataset público.

## 3. Veredicto de procedencia

**La fuente exacta del corpus NO es identificable con la evidencia disponible: se declara IRRESOLUBLE.**

Lo que sí sostiene la evidencia:
1. **No es una redistribución de PAN** (2013-2019) ni de TwiSty: difieren el esquema de edades (3/4/5 clases frente a 6 con 13-17), la granularidad (perfiles por autor frente a tweets sueltos), el idioma/base de blogs, y sobre todo PAN distribuye **URLs o XML por autor**, nunca texto plano de tweets con formato de ancho fijo.
2. El corpus parece **generado por el propio organizador** a partir de un barrido de Twitter en español latinoamericano de **≈2016-2017**, con etiquetas de edad obtenidas por fuera del texto (no verificables), y posteriormente:
   - remuestreado para equilibrar las 6 clases,
   - **normalizado a ASCII** (pérdida de acentos y probablemente de emojis no ASCII),
   - almacenado/exportado en un **campo de ancho fijo de 140 caracteres** y luego dividido en train/test sin agrupar por autor.
3. La coincidencia del esquema de 6 rangos con el Blog Authorship Corpus sugiere que **la convención de etiquetas proviene de la tradición académica de author profiling**, no de una restricción de la plataforma.

## 4. Implicaciones para el proyecto

**(i) Corpus no público ⇒ se confirma la decisión de no usar datos externos ni filtrar un test público.**
No existe dataset público identificado del que provenga este corpus, y tampoco hay un conjunto de prueba público que "filtrar": el test de la competencia (112.892 filas) solo es accesible vía Kaggle. Por tanto, la decisión de D-001 (sin datos externos etiquetados) no sacrifica ninguna ventaja recuperable, y no hay atajo de fuga de información disponible. Además, cualquier espejo público de Twitter con **texto completo** (y no IDs) sería una redistribución contra las políticas de la plataforma: que la competencia provea texto completo refuerza que el organizador preparó el material por su cuenta.

**(ii) Baselines publicados comparables para calibrar expectativas de macro-F1.**
La tarea más cercana es la clasificación de edad en Twitter con 4-6 clases:
- PAN 2014 (5 clases, 18+): mejor sistema **0,2895** de accuracy frente a baseline 0,1404.
- PAN 2015 (4 clases, 18+): mejor sistema **0,8404** de accuracy media edad+género (el número sube por el ancho de la última clase 50-xx).
- PAN 2016 (5 clases, cross-genre, Twitter EN/ES): ganador **0,5258** de accuracy media edad+género.
- PAN 2013 (3 clases amplias): ganador **0,3894** de accuracy; baseline 0,1650.
- Nguyen et al. (COLING 2014): **los humanos no superan al clasificador** estimando edad desde tweets.
- Morgan-Lopez et al. (PLOS ONE 2017): con solo texto la señal es modesta (sus mejores resultados usan metadatos).
Con 6 clases (una de ellas 13-17, la más difícil de aislar por estilo) y tweets de 140 caracteres, el rango realista de **macro-F1** es **≈0,30-0,45**: un baseline TF-IDF + modelo lineal debería moverse en ≈0,30-0,36, y un sistema bien ensamblado con ajuste de umbral por clase en ≈0,38-0,45. Cualquier resultado >0,50 exigiría desconfiar del esquema de validación (leakage) antes que celebrarlo.

**(iii) Sobre el truncado a 140 y el esquema de 6 rangos.**
- El **truncado/padding de 140 caracteres es intrínseco al corpus y es un artefacto de exportación**, no una decisión de modelado: se confirma la política D-005 (`rstrip` del padding) y se añade que el truncado a 140 es irreversible para los textos que lo alcanzan (mediana útil 93 caracteres: el artefacto afecta a una minoría de filas pero elimina su cola de contexto).
- La **normalización a ASCII** (acentos y emojis no ASCII perdidos) es un artefacto adicional, no documentado por el organizador, con consecuencias directas: degrada POS taggers, lexicones en español y cualquier feature sensible a la ortografía correcta, y **favorece las features de carácter** (el `n` sin tilde es un token distinto del `n` con tilde en el vocabulario original, pero consistente en todo el corpus). Refuerza la elección de char n-grams y de features superficiales como bloque principal (evidencia externa: en PAN 2019, un baseline de char n-grams + Random Forest superó al baseline de embeddings, 0,8972 vs 0,8444 en bots y 0,7289 vs 0,7156 en género, en español).
- El **esquema de 6 rangos es una convención de etiquetado** (herencia de Schler et al. 2006), no una restricción de la plataforma: no debe colapsarse a 4-5 clases para "subir la métrica", porque la métrica oficial se calcula sobre las 6 etiquetas entregadas. Sí conviene un análisis de confusión adyacente (25-34 vs 35-49; 50-64 vs 65-xx) porque es donde la literatura sitúa la mayor parte del error.

## 5. Anexo — Estado de verificación de las referencias solicitadas

Referencias pedidas por el orquestador que **NO pudieron verificarse** con fuentes web en esta sesión
(CORREGIDO ≠ descartado; simplemente no comprobado):

| Referencia solicitada | Estado | Nota |
|---|---|---|
| Rosenthal & McKeown 2011, *Age Prediction in Blogs* | **NO VERIFICADO** | Sin DOI ni URL confirmada en Crossref/ACL Anthology; probable informe técnico de Columbia University. Se incluye en el CSV marcada explícitamente como no verificada |
| Liao et al. 2014, age/gender desde texto | **NO LOCALIZADO** | No se encontró una publicación de 2014 con ese autor sobre perfilado de edad; se sustituyó por Verhoeven & Daelemans (LREC 2014, CLiPS CSI corpus) verificada, y por Liu, Singh & Mneimneh (DeLTA 2021) |
| "Li et al. 2010" sobre optimización de F1 multiclase | **NO LOCALIZADO** | No se identificó el trabajo con esa clave; la corriente (d) se cubre con Parambath et al. 2014, Waegeman et al. 2014 (GFM), Narasimhan et al. 2015 y Lipton et al. 2014, todos verificados |
| Dembczynski et al. 2011 (NIPS, GFM) | **VERIFICADO INDIRECTAMENTE** | Citado como "(Dembczynski et al. 2011)" en los trabajos verificados arXiv:1310.4849 y arXiv:1604.07759; no se abrió el PDF original |
| Jansche 2005 (maximum expected F-measure) | **NO VERIFICADO** | No se intentó la verificación en la fuente (presupuesto de consultas); no incluido en el CSV |
| Wang & Manning 2012 (NBSVM) | **NO VERIFICADO** | No localizado en Crossref ni en el índice de ACL Anthology con las consultas realizadas; su idea de log-count ratio queda cubierta por Monroe et al. 2008 y Mukherjee & Liu 2010 (ambos verificados) |

Correcciones respecto a la lista de referencias solicitada (verificadas en la fuente):
- **Flekova & Gurevych** es de **PAN@CLEF 2013** (CEUR Vol-1179), no de 2015.
- **Sap et al. 2014** es **D14-1121** (`10.3115/v1/D14-1121`), no D14-1201 (ese ID corresponde a otro artículo).
- **Hovy & Fornaciari 2018** es **D18-1070** (`10.18653/v1/D18-1070`), no D18-1069 (verificado en Crossref).
- **Argamon et al. 2007** usa el DOI `10.5210/fm.v12i9.2003` (la página del artículo migró en el sitio de First Monday).
- **Mukherjee & Liu 2010** = D10-1021 y **Sapkota et al. 2015** = N15-1010: ambos IDs confirmados por su entrada BibTeX.
- El PDF del **Blog Authorship Corpus** (`u.cs.biu.ac.il/~koppel/BlogCorpus.htm`) no respondió (timeout) en dos intentos: las cifras de ese corpus quedan marcadas como no verificadas.
