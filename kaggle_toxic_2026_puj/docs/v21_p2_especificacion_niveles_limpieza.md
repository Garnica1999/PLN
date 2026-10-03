# SPEC P2 - Niveles de limpieza del corpus: RAW, LIGHT y MEDIUM

Proyecto: `kaggle_toxic_2026_pUJ` / `KAGGLE_TOXIC_2026_PUJ`
Fase: V21, fase P2. Rama `v21`, worktree `/home/carlos/Git/PLN-v21`.
Fecha: 2026-09-27. Estado: P0 cerrado, P1 en curso, **P2 especificado, nada ejecutado**.
Formato: ASCII puro, espanol sin acentos. Verificado byte a byte.
Presupuesto: 3 de 8 unidades de la fase. Coste de P2: 3 unidades = 3 entrenamientos.

---

## 1. Objective

**Medir el efecto del nivel de limpieza del corpus** sobre la aportacion complementaria de
la representacion densa entrenada desde cero, comparando tres niveles construidos desde
el **mismo dump**, con la **misma arquitectura**, la **misma downstream** y la **misma
parte task**.

La hipotesis ya recortada por P0 **no** es "el corpus estaba sucio": P0 demostro que V5
implemento `limpiar_marcado` + `normalizar_emb_normal` v5.0, deduplicacion por sha1,
filtro de 20 tokens y auditoria de fuga. La hipotesis viva es mas estrecha:

- **H-P2 (a favor)**: el nivel de limpieza es una variable no medida. Existe una sola
  variante (EMB-NORMAL), asi que nunca se ha comprobado si **menos** limpieza mejora la
  geometria que la tarea lee, ni si **mas** limpieza la mejora.
- **H0-P2 (en contra)**: la diferencia de nivel queda por debajo del ruido del protocolo.
  Es la hipotesis nula por defecto y es un resultado valido y documentado.

Success criteria de la fase, en una frase: **las tres condiciones quedan medidas con la
misma maquinaria, la misma referencia y los mismos controles, y P2 produce un veredicto
explicito a favor o en contra de H-P2, con la covariable de escala registrada.**

Lo que P2 **no** es: no es "aplicar limpieza" (ya se aplica), no es "buscar un corpus
mejor" (eso es la hipotesis de registro social, que pertenece a otra fase), no es una
barrida de hiperparametros de fastText (cerrada en v5.1 con 14 configuraciones), y no es
una sustitucion de V03.

---

## 2. Inputs

Resultados de la fase anterior que P2 consume, todos ya en disco y sin recalcular:

| Input | Ruta | Uso en P2 |
|---|---|---|
| P0 inventario y auditoria | `reports/v21_p0_00_inventario_v5_v5_1.md` | delimita la hipotesis, fija que no se repite la auditoria V5/V5.1 |
| P1 revision de estado del arte | `reports/v21_p1_01_embeddings_desde_cero.md` | declara agotadas las tres lineas que P2 no debe reabrir |
| Pipeline de corpus V5 | `experiments/embedding_01_corpus_wikipedia_v5.py` | `limpiar_marcado` (L307), `normalizar_emb_normal` (L350), `tokenizar` (L410), `construir_corpus` (L422), `concatenar_corpus` (L530), `hash_ngramas` (L556), `auditar_fuga` (L575), `vocabulario_toxic` (L627), `GRUPOS_RELEVANTES` (L642), `metricas_grupo` (L690), `clasificar_grupos` (L724), `medir_cobertura` (L752) |
| Maquinaria de familia V5.1 | `experiments/embedding_13_window_v5_1.py` | las tres evaluaciones downstream por variante, con el mismo punto de operacion |
| Fusion y complementariedad | `experiments/embedding_03_fusion_complementariedad_v5.py` | `vectores_documento` (L146), `matrices_tfidf` (L171), `evaluar_variante` (L278), `matriz_complementariedad` (L480), `rebanadas` (L508), `metricas_por_rebanada` (L559) |
| Downstream base | `experiments/embedding_02_fasttext_baseline_v5.py` | `evaluar_downstream`, `ARQUITECTURA` |
| Identidad de entrenamiento | `experiments/embedding_11_auditoria_firmas_v5_1.py` | `firma_entrenamiento` (L318) |
| Utilidades | `src/embedding_utils.py` | `entrenar_fasttext` (L212), `normalizar_emb_normal`, `tokenizar`, `cargar_matriz_vec` |
| Splits y metricas | `src/splits.py`, `src/evaluation.py` | `holdout_v2`, `pliegues_corregidos`, `grupos_fusionados`, `metrica_macro`, `bootstrap_diferencia_pareada`, `_f1_por_celda` |
| OOF canonico de V03 | `features/oof_v03_final.npz` | referencia congelada de la comparacion pareada |
| Manifiesto de corpus | `data/external/corpora/corpus_manifest.json` | `processed_sha256` de E-CORP-01, que es el valor con el que se verifica la reproduccion |
| Presupuesto de V5.1 | `reports/embedding_budget_v5_1.json` | tabla de firmas existentes contra las que se comprueba colision o no colision |
| Registro de experimentos | `experiments.csv` | filas `EBD-*` que P2 continua |

Valores de referencia que P2 debe reproducir, tomados del manifiesto y del informe de
V5, no de la memoria del implementador:

| Referencia | Valor | Origen |
|---|---|---|
| V03 `corrected_cv` | `0.7772114525165721` | OOF canonico |
| Umbrales congelados de V03 | HS 0.44, TR 0.51, AG 0.46 | D-039 |
| FUSION-WS-3 | `0.782959` | EBD-19 |
| `emb_solo` WS-3 | `0.6694` | EBD-19 |
| E-CORP-01 `processed_sha256` | `00b839da6fbc377ee5375acc9a236e831e46ecf92cc544188b81aef1a826e871` | manifiesto |
| E-CORP-01 documentos / tokens / tipos | 2.847 / 10.002.400 / 271.209 | manifiesto |
| Cobertura TOXIC `train+test` de E-CORP-01 | tipo `0.6839`, token `0.8833` | manifiesto |
| Cobertura TOXIC `train` de E-CORP-01 | tipo `0.7042`, token `0.8831` | manifiesto |
| Tipos de train consultados | 6.059 | correccion metodologica de EBD-18 |
| Corpus LIGHT: fraccion de OOV de train resoluble por esqueleto vocalico | 0.6741 | manifiesto |

---

## 3. Delimitacion frente a P0: lo que P2 no repite

P0 ya cerro por evidencia, y P2 no reabre ninguna de estas lineas:

- **No** se repite la auditoria de V5/V5.1. No se reescribe `limpiar_marcado` ni
  `normalizar_emb_normal`. Se **importan** y se **(re)usan**.
- **No** se varyan `dim`, `ws` mas alla del valor fijado, `minCount`, `minn/maxn` ni
  `neg`. La arquitectura es la de la seccion 6 y solo cambia el corpus.
- **No** se toca la normalizacion L2 por documento. EBD-08 midio que estandarizar el
  bloque denso cuesta -0.0871. Es la convencion fijada de la fase.
- **No** se ajustan umbrales por condicion. Los tres usan 0.44/0.51/0.46.
- **No** se estandariza, apila ni pondera el bloque denso de otra forma.
- **No** se gasta ninguna submission. 4 disponibles, 19 de 21 historicas quedaron abajo.

---

## 4. Separacion obligatoria: corpus preprocessing frente a task preprocessing

Esta seccion responde al punto 6 del encargo y es la que garantiza que la comparacion
mida **el corpus** y no dos cambios a la vez. Es normativa: el implementador no puede
fusionar los dos planos.

### 4.1 Las dos transformaciones, nombradas por separado

| Plano | Que se transforma | Donde se aplica | Cambia entre RAW / LIGHT / MEDIUM |
|---|---|---|---|
| **CORPUS** | el texto del dump MediaWiki que se da a fastText | antes de escribir `tokens_10M.txt` | **SI, es la variable del experimento** |
| **TASK** | el texto de los 1.731 tweets de train y 1.154 de test, hasta llegar a los vectores de tipo | despues de entrenar, antes de `vectores_documento` | **NO, es identico en las tres condiciones** |

### 4.2 Definicion congelada del plano TASK (fija, sin excepcion)

Se aplica **exactamente** la funcion ya validada en v5, importada de
`src.embedding_utils`, sin reimplementarla:

```
texto_task = normalizar_emb_normal(texto_del_tweet)   # minusculas conservando tildes,
                                                     # url -> url, #tag -> hashtag,
                                                     # @user -> usuario, digitos -> numero,
                                                     # espaciado colapsado. Sin corrector.
tokens    = tokenizar(texto_task)                    # RE_TOKEN identico al del corpus
utiles    = [t for t in tokens if t in vocabulario_del_embedding]
vector    = mean(vectores[t] for t in utiles)        # pooling A, media simple
vector    = vector / ||vector||                      # normalizacion L2 por documento
```

Reglas fijas del plano TASK, iguales en las tres condiciones:

1. `limpiar_marcado` **nunca** se aplica a los tweets: no hay marcado MediaWiki en ellos.
   Aplicarlo seria un error de categoria.
2. El patron de los marcadores es el de `normalizar_emb_normal`, no el de la condicion
   del corpus. Un tweet se tokeniza igual se entrene con RAW, LIGHT o MEDIUM.
3. Los tokens sin vector se **omiten** del promedio (regla de `vectores_documento`); si
   un documento se queda sin tokens utiles, su vector es de ceros. Se registra
   `fraccion_tokens_sin_vector` por condicion como diagnostico.
4. La fusion es `hstack(TFIDF_V03, bloque_denso_L2)` y el receptor es
   `LogisticRegression(C=2.0, class_weight="balanced", solver="liblinear")` con los tres
   umbrales congelados. Sin `StandardScaler`, sin `Ridge`, sin apilamiento.
5. La normalizacion L2 se aplica **por documento**, nunca por caracteristica.

### 4.3 Por que los dos planos tienen que diferir

- **Asimetria de funcion**: la transformacion del corpus solo tiene que dar forma a la
  distribucion de contextos que la representacion aprende. La transformacion de la tarea
  tiene que ser **inofensiva para la senal** y **compatible** con las formas de token que
  la representacion sabe(["]representar.
- **Compatibilidad de vocabulario**: `normalizar_emb_normal` produce los tokens `url`,
  `hashtag`, `usuario` y `numero`. En el corpus LIGHT y MEDIUM esos cuatro tokens existen
  y son frecuentes, luego tienen vector bien estimado. En el corpus RAW **no existen**.
  La unica forma de dar a las tres condiciones una interfaz comparable es mantener el
  lado task con EMB-NORMAL. El coste es que el vocabulario de RAW no cubre los
  marcadores, y **ese coste es precisamente una de las cosas que RAW mide**, no un handicap
  impuesto desde fuera.
- **La asimetria es la de un pipeline real**: en un despliegue real el corpus de
  entrenamiento y los textos de entrada casi nunca pasan por el mismo filtro. Fijar el
  lado task es lo que hace que la comparacion mida la calidad del corpus y no la
  artefactibilidad de la tuberia.
- **Sin fuga y sin simetria rota**: la funcion del lado task se define sobre las
  **entradas** de train y de test, nunca sobre etiquetas, y se aplica de forma identica en
  ajuste y en inferencia. No se ajusta nada con el holdout ni con test.

### 4.4 Prediccion cualitativa pre-registrada (falsable)

En el corpus RAW los grupos `hashtags`, `menciones` y `marcadores_web` de la tabla de
cobertura deben dar `type_coverage = 0.0000`, porque los tokens `hashtag`, `usuario`, `url`
y `numero` no existen en un corpus sin `normalizar_emb_normal`. En LIGHT deben dar
`1.0000` como da el manifiesto. **Si RAW tambien da 1.0000, el corpus RAW no es RAW** y
eso es un fallo de construccion, no un resultado. Se registra como asercion.

---

## 5. Definicion exacta de las tres condiciones

### 5.1 Seleccion de articulo: IDENTICA en las tres condiciones

Se reutiliza el predicado de `_articulo_de` (L278) tal cual: `ns == "0"`, titulo no vacio
y sin `:`, no redirect. Lo que se sustituye es **solo** la funcion de transformacion del
cuerpo del articulo. Si la seleccion de articulo cambiara, RAW compararia otra cosa.

### 5.2 RAW

Transformacion unica del cuerpo, aplicada al campo `<text>` crudo:

```
t = marcado.lower()          # minusculas conservando tildes
t = RE_ESPACIOS.sub(" ", t).strip()   # colapsa saltos de linea: una linea = un documento
```

Despues, **identico** en las tres condiciones: `huella = sha1(t)`, `tokens = tokenizar(t)`
con `RE_TOKEN = \w+(?:['\u2019]\w+)*` y `re.UNICODE`, descarte si `len(tokens) < 20`, y
escritura de `" ".join(tokens) + "\n"`.

Lo que RAW **conserva** y es el objeto del experimento: comentarios, referencias,
plantillas, tablas, enlaces con sus parametros, entidades HTML sin resolver, marcas de
edicion y todo el marcado inline. Lo que RAW **no** tiene: los marcadores `url`,
`hashtag`, `usuario`, `numero`.

Nota tecnica que P2 debe medir y no asumir: el tokenizador `RE_TOKEN` solo captura
`\w`, de modo que la **puntuacion** del marcado (`{{`, `|`, `}}`, `<`, `>`) desaparece de
todos modos. Lo que sobrevive al tokenizador es el **contenido alfabetico** del marcado:
nombres y valores de parametros de plantilla, texto de celdas de tabla, autores de
referencia, y entidades no resueltas. Esa es la magnitud real del efecto y por eso el
step 0 la mide antes de gastar una unidad.

### 5.3 LIGHT

Exactamente el pipeline de V5, sin cambios, reentrenado desde cero:
`limpiar_marcado` (L307) + `normalizar_emb_normal` (L350) + dedup por sha1 + filtro de 20
tokens, con `preprocessing_version = "v5.0-EMB-NORMAL"`.

Se reentrena **desde cero** aunque el `.vec` de V5 este en disco. Motivo declarado: el
objeto de P2 es medir el efecto del corpus, y reutilizar el `.vec` cacheado convertiria la
fila LIGHT en una relectura de EBD-19 en vez de una medicion. El reentrenamiento cuesta
una unidad de presupuesto y aporta dos cosas que ninguna relectura puede dar: el control
de cordura de reproduccion y **la primera estimacion empirica de la banda de
no-determinismo del protocolo a configuracion fija**, que es el suelo de ruido contra el
que se leen los tres niveles. El `.vec` cacheado (`fasttext_wiki_10m_ws_3.vec`) **no se
carga** en el pipeline de P2; su `model_sha256` se registra solo como referencia.

### 5.4 MEDIUM

LIGHT mas dos transformaciones, ambas con hipotesis y prediccion refutables, detalladas en
la seccion 7: **T1** marcado residual y **T2** normalizacion Unicode NFC. MEDIUM conserva
todo lo demas de LIGHT: marcadores, whitelist, dedup, filtro de 20 tokens, tokenizer,
seleccion de articulo, y las tres prohibiciones (no elimina negaciones, no elimina
puntuacion emocional, no elimina mayusculas, no colapsa elongaciones).

---

## 6. Process

Numerado y ejecutable. Cada step es un script independiente; el implementador puede
parar entre steps y el estado queda en disco.

### 6.0 Convenciones obligatorias de codigo y artefactos

- Interprete: `/home/carlos/venv/bin/python`. Threads de BLAS: 32.
- **P2 es CPU.** No se usa la GPU ROCm: fastText C++ no la usa y el resto es
  `scikit-learn` sobre matrices dispersas.
- Convencion de nombres, siguiendo la existente de v5/v5.1
  (`embedding_NN_<descripcion>_vX.py`), con `NN` correlativo desde 23:
  - `experiments/embedding_23_preflight_marcado_v21.py` (step 0, 0 unidades)
  - `experiments/embedding_24_corpus_niveles_v21.py` (steps 1 a 4, 0 unidades)
  - `experiments/embedding_25_niveles_downstream_v21.py` (steps 5 a 8, 3 unidades)
- **Un script completo por experimento, sin versionar y sin borrar codigo previo.**
- Prohibido reimplementar: `limpiar_marcado`, `normalizar_emb_normal`, `tokenizar`,
  `hash_ngramas`, `auditar_fuga`, `clasificar_grupos`, `metricas_grupo`,
  `medir_cobertura`, `firma_entrenamiento`, `vectores_documento`, `evaluar_variante`,
  `matriz_complementariedad`, `rebanadas`, `metricas_por_rebanada`, `evaluar_downstream`.
  Se cargan con `importlib.util.spec_from_file_location`, como ya hace
  `embedding_03_fusion_complementariedad_v5.py` con `embedding_01`.
- Docstrings completos en cada funcion (estilo del proyecto). Sin emojis ni acentos en el
  codigo. Constantes con nombre en mayusculas al principio del modulo.
- Rutas de salida:
  - corpus: `data/external/corpora/wikipedia_es/p2_raw/`,
    `.../p2_light/`, `.../p2_medium/`, cada uno con `tokens_shard_XX.txt`,
    `tokens_10M.txt` y `vocabulario.txt`.
  - modelos: `data/external/embeddings/fasttext_p2_{raw,light,medium}.vec` y `.bin`.
  - sifts: `data/external/embeddings/fasttext-p2-{raw,light,medium}_sifts.txt` y
    `..._sifts.entrada.txt`.
  - informes: `reports/v21_p2_01_niveles_limpieza.md`,
    `reports/v21_p2_01_niveles_limpieza.json`,
    `reports/v21_p2_01_niveles_limpieza.csv`,
    `reports/v21_p2_01_niveles_limpieza_rebanadas.csv`,
    `reports/figuras/fig_58_niveles_limpieza_v21.png`.

### 6.1 Step 0 - Pre-flight de marcado (0 unidades, ~2 min)

Se ejecuta **antes** de construir nada. Su unico proposito es medir si la comparacion
puede discriminar, y evitar gastar unidades en una comparacion vacia.

Sobre las primeras 5.000 paginas validas del mismo dump multistream
(`data/external/corpora/raw/eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2`,
`raw_sha256` `3248f107590e6a00e458fdce09616bb22726d78fe3f40106f5a1a7b05d50dabc`),
aplicando las tres funciones de transformacion en memoria, **sin escribir corpus**,
calcular y volcar en `reports/v21_p2_00_preflight.json`:

1. `fraccion_caracteres_boilerplate`: caracteres que caen dentro de `<!--...-->`,
   `<ref...>...</ref>`, `{{...}}`, `{|...|}` o `[[...]]`, dividido por el total. Se
   recorre linea a linea con los mismos regex de `limpiar_marcado`.
2. `fraccion_caracteres_nfd`: caracteres cuyo NFC difiere de si mismos, es decir
   caracteres en forma descompuesta. Se calcula con
   `unicodedata.normalize("NFD", c) != c`.
3. `fraccion_documentos_con_marcado_residual` por condicion: documentos cuya salida de
   `limpiar_marcado` (LIGHT) o de la etapa T1 (MEDIUM) sigue conteniendo `{{`, `}}`,
   `{|`, `|}` o `<ref`. Para RAW se mide sobre el texto crudo con los mismos markers.
4. `fraccion_lineas_campo_plantilla`: lineas que casan con `^\w{2,24}\s*=\s*\S` despues de
   LIGHT. Justifica o refute la segunda parte de T1.
5. `top30_tokens_ganados_por_LIGHT` y `top30_tokens_perdidos_por_LIGHT`: tipos que solo
   aparecen con una de las dos transformaciones, con su frecuencia. Da nombre al ruido de
   plantilla en vez de suponerlo.

Regla de decision del step 0 (ver seccion "Decision de parada", criterio D1): si
`fraccion_documentos_con_marcado_residual` de LIGHT es menor que 0.02 **y**
`fraccion_caracteres_nfd` es menor que 0.001, P2 se detiene y se reporta que, con este
tokenizador, los tres niveles no son separables. No se construye ni entrena nada.

### 6.2 Step 1 - Construccion de los tres corpus (0 unidades, ~3 min)

Reutilizar `construir_corpus` (L422) con estas adaptaciones minimas y declaradas:

- `CORPUS_DIR` pasa a ser un **parametro** `corpus_dir`, con un valor por condicion. No se
  escribe nunca en `data/external/corpora/wikipedia_es/emb_normal/`.
- La funcion de transformacion del cuerpo se pasa como parametro `transformar`, con tres
  implementaciones: `transformar_raw` (seccion 5.2), `transformar_light` (identica a la de
  V5), `transformar_medium` (LIGHT + T1 + T2, seccion 7).
- Todo lo demas es identico: `TAMAO_MINIMO_TOKENS_DOC = 20`, dedup por sha1 del texto
  **antes** de la normalizacion, escritura de una linea por documento, `vocabulario.txt`.
- Tras el bucle, `concatenar_corpus` (L530) con `objetivo_tokens = 10_000_000`, identico
  en las tres condiciones: se escriben lineas completas hasta que `tokens >= 10_000_000`,
  que es el mecanismo que produjo los 10.002.400 tokens de E-CORP-01.

Por condicion se registra en el manifiesto y en el JSON: `documentos`, `tokens`, `tipos`,
`articulos_leidos`, `duplicados`, `cortos`, `sin_texto_util`, `segundos`,
`processed_sha256` (sha256 del `tokens_10M.txt` concatenado), `bytes` y
`preprocessing_version`.

Ids de corpus propuestos, que no colisionan con E-CORP-01 a E-CORP-09:
`E-CORP-10` = wikipedia_es RAW 10M (`v21.0-P2-RAW`),
`E-CORP-11` = wikipedia_es MEDIUM 10M (`v21.0-P2-MEDIUM`).
LIGHT **no** crea id nuevo: reutiliza `E-CORP-01` con `v5.0-EMB-NORMAL`, y su
`processed_sha256` debe coincidir exactamente con el del manifiesto.

### 6.3 Step 2 - Auditoria de fuga por hash de 8-gramas (0 unidades, ~6 min)

`auditar_fuga` (L575) **sin modificar**, ejecutada sobre el `tokens_10M.txt` final de cada
condicion, con `NGRAMAS_FUGA = 8` y la misma multiplicacion `SEMILLA_Z`. Se ejecuta
contra **train y test** y se vuelca `n_gramas`, `n_gramas_unicos_corpus`,
`textos_con_coincidencia` y `detalle` por condicion.

Asercion dura: **`textos_con_coincidencia == 0` en las tres condiciones**. Un corpus con
cualquier coincidencia se descarta y no se registra. RAW es el de mayor riesgo, porque el
texto de plantilla repetido genera 8-gramas muy comunes.

Se mantiene la semetrica existente: el lado TOXIC se tokeniza con `normalizar_emb_normal`
(la funcion del plano TASK) y se busca en el indice del corpus. Esa asimetria es
intencional y es la de V5; cambiarla haria la auditoria incomparable con la historica.

### 6.4 Step 3 - Cobertura del vocabulario TOXIC (0 unidades, ~3 min)

`medir_cobertura` (L752) con `clasificar_grupos` (L724) y `GRUPOS_RELEVANTES` (L642)
**verbatim**, sobre el `vocabulario.txt` de cada condicion. Se reportan:

- Global, para `train`, `test` y `train+test`: `n_types`, `n_tokens`, `type_coverage`,
  `token_coverage`, `oov_rate_types`, `oov_rate_tokens`, `n_word_oov`.
- Por los 11 grupos: `n_types`, `n_tokens`, `type_coverage`, `token_cobertura`,
  `oov_rate`, `frecuencia_media_corpus`.
- Proyecciones de resolubilidad: `n_word_oov_resolubles_por_ascii` con su tasa,
  `n_word_oov_resolubles_por_esqueleto` con su tasa, y `type_coverage` proyectada por cada
  via.
- Diagnostico de marcadores: si `url`, `hashtag`, `usuario` y `numero` estan en el
  vocabulario de cada corpus (prediccion de la seccion 4.4).
- Diagnostico de acento: numero de tipos del corpus que contienen al menos una vocal
  acentuada, por condicion. Este es el readout directo de T2.

**Asercion dura**: LIGHT debe dar `type_coverage = 0.6839` y `token_coverage = 0.8833` en
`train+test`, y `0.7042` / `0.8831` en `train`, con las 4 decimales del manifiesto. Esta
asercion es **determinista**: no interviene fastText, asi que cualquier desviacion indica
un fallo de construccion del corpus, no ruido.

`representation_oov_rate` (que requiere volcar los vectores de los tipos consultados con
el modelo ya entrenado) se calcula en el step 5, para las tres condiciones, sobre los
**6.059 tipos de train** (correccion metodologica de EBD-18: nunca `train+test`). Se
registra como medicion obligatoria, pero **no** es puerta de exito: no puede bloquear la
fase.

### 6.5 Step 4 - Control de cordura de reproduccion (0 unidades, gate)

Se ejecuta **antes** de entrenar RAW y MEDIUM. Si falla, P2 para. Son tres aserciones
sobre LIGHT, en este orden:

1. **A - Identidad del corpus (exacta)**: `sha256(tokens_10M.txt)` de LIGHT debe ser igual
   al `processed_sha256` de E-CORP-01 en el manifiesto. Lee el valor del manifiesto; no lo
   escribas a mano en el codigo. Es la asercion mas fuerte y la mas barata.
2. **B - Identidad de cobertura (exacta)**: los valores de la seccion 6.4.
3. **C - Identidad de metrica (con tolerancia)**: ver seccion 6.7, tras entrenar LIGHT.

Orden obligatorio en disco: construir LIGHT, verificar A y B, **entrenar LIGHT**,
verificar C, y solo entonces construir/entrenar los otros dos. Si A o B fallan, se para
antes de gastar la unidad.

### 6.6 Step 5 - Entrenamiento de los tres modelos (3 unidades, ~6 min)

Arquitectura **congelada**, identica en las tres condiciones y sin un solo cambio:

```
FastText C++ commit 1142dc4c4ecb4 (MIT), modo no supervisado, subcomando `skipgram`
dim = 200 | ws = 3 | minCount = 3 | minn = 3 | maxn = 6 | neg = 10
epoch = 5 | alpha = 0.05 (defecto del binario, NO se pasa -lr)
sample = 1e-4 (defecto del binario, NO se pasa -sample)
bucket = 200000 | seed = 42 | thread = 16
```

`ws = 3` porque es el mejor medido en v5.1 (EBD-19) y es el punto de operacion contra el
que estan los dos valores de referencia ancla. La llamada es
`entrenar_fasttext` (L212) con `dim=200, ventana=3, min_count=3, negativos=10, epocas=5,
hilos=16, semilla=42, minn=3, maxn=6, bucket=200000`. Se guarda la cadena de comando
exacta que devuelve (`resultado["comando"]`) y se **aserta** que no contiene `-lr` ni
`-sample`, para que quede escrito que esos dos valores son los del binario. Se registra
tambien el `sha256` del ejecutable `tools/fastText/build/fasttext`.

Por modelo: `training_time_s`, `vocabulary_size` (del log del binario), `bytes` y
`sha256` de `.vec` y de `.bin`. Los dos se conservan: hay 434 GB libres y el `.bin` es la
evidencia de que el entrenamiento ocurrio.

Descarta el `.vec` viejo de V5 para WS-3: **no se carga**.

Volcado de sifts por modelo (los 6.059 tipos de train) para el
`representation_oov_rate` del step 6.4.

### 6.7 Step 6 - Asercion C: reproduccion de la metrica de referencia

Con LIGHT ya entrenado, se ejecutan sus tres evaluaciones downstream y se compara:

| Magnitud | Referencia | Asercion |
|---|---|---|
| `corrected_cv` de FUSION-LIGHT | `0.782959` | `abs(d - 0.782959) <= 0.0015` |
| `corrected_cv` de EMB-LIGHT-LR | `0.6694` | `abs(d - 0.6694) <= 0.0050` |
| `corrected_cv` del control V03 | `0.7772114525165721` | exacto a 6 decimales |

Tolerancias pre-registradas: 0.0015 en la fusion y 0.0050 en el embedding aislado.
**No son arbitrarias**: 0.0015 es la mitad del `DISPERSION_MAXIMA_FAMILIA = 0.005` que
v5.1 declaro, y por debajo de la dispersion interna observada de la familia window
(0.0024). La del embedding aislado es mas ancha porque la fusion es la magnitud que decide
y porque el rango de `emb_solo` es 4.8 veces el de la fusion.

El numero `epsilon_determinismo = abs(corrected_cv_FUSION_LIGHT - 0.782959)` **se registra
como producto secundario de primera clase**: es la primera estimacion empirica de cuanto
mueve el protocolo reentrenar la misma configuracion con la misma semilla, algo que el
historial nunca midio porque las familias reevaluaban `.vec` cacheados. Si
`epsilon_determinismo` resulta mayor que 0.0015, la asercion C falla y P2 para
(criterio D2 de la Decision de parada).

### 6.8 Step 7 - Las nueve evaluaciones downstream (0 unidades, ~6 min)

Se replica la maquinaria de `embedding_13_window_v5_1.py` **importandola**, no
reescribiendola. Por cada una de las tres condiciones, exactamente tres evaluaciones:

1. `EMB-<COND>-LR`: bloque denso solo, `evaluar_downstream(nombre, embeddings, y,
   pliegues, escalador=True)`, metricas con umbrales congelados.
2. `EMB-<COND>-SVC`: lo mismo con `LinearSVC`, control.
3. `FUSION-<COND>`: `evaluar_variante(nombre, textos, y, pliegues, embeddings,
   modo_embedding="l2")`, es decir `hstack` de las 27.431 TF-IDF de V03 con las 200d en
   L2 por documento, y `LogisticRegression(C=2.0, balanced, liblinear)`.

Es decir 9 evaluaciones. Para cada una se registra: `corrected_cv` con umbrales congelados,
`corrected_cv` con umbral propio de cada etiqueta (diagnostico, nunca usado para decidir),
`delta_vs_V03`, `std_pliegues_0.5`, y para las tres de fusion el IC95 pareado.

Sobre la comparacion con V03:
- `bootstrap_diferencia_pareada(y, pred_fusion, pred_v03, 10_000, 20260925)`. Se **reutiliza
  la semilla de bootstrap de v5.1** a proposito, para que los IC95 de P2 sean comparables
 termine a termino con la tabla de v5.1 y no con una nueva banda de ruido.
- `matriz_complementariedad(y, pred_fusion, pred_v03)` (L480): `celdas`, `ambas_correctas`,
  `solo_candidato`, `solo_referencia`, `ninguna_correcta`. La unidad es la **celda**, y su
  conteo de acierto usa `_f1_por_celda >= 0.999` (correccion de vocabulario de EBD-24).
- `rebanadas(desarrollo, y)` (L508): las 11 rebanadas, identicas en las tres condiciones
  porque se definen con la normalizacion del lado task. De ahi salen los deltas por
  rebanada de cada condicion frente a V03.
- Guardas de integridad: los ids del OOF canonico deben coincidir con el desarrollo
  (asercion de `embedding_13`, L~230), y se comprueban 1.483 filas de desarrollo y 248 de
  holdout.

### 6.9 Step 8 - Firma de entrenamiento, informe y registro (0 unidades)

**Firmas**: `firma_entrenamiento` (L318 de `embedding_11`) con los **13 campos**, en este
orden fijo:

```
processed_sha256 | corpus_version | dim | ventana | min_count | minn | maxn
| negativos | muestreo | epocas | sg | semilla | bucket
```

Aserciones:
- La firma de **LIGHT colisiona, por diseno**, con la identidad WS-3 ya registrada en
  `reports/embedding_budget_v5_1.json`. Que colisione es la prueba de que la condicion
  LIGHT es identica, bit a bit, a la que ya midio 0.7830.
- Las firmas de **RAW** y de **MEDIUM** son distintas entre si y **no colisionan** con
  ninguna de las firmas de `embedding_budget_v5_1.json` ni con ninguno de los 19 `.vec` ya
  en `data/external/embeddings/`. Se comprueba leyendo esos artefactos, no de memoria.
- `hilos` queda fuera de la firma a proposito: no cambia el modelo, solo el orden no
  determinista de actualizacion, que es justo lo que la asercion C mide.

**Manifiesto**: `data/external/corpora/corpus_manifest.json` se actualiza **anadiendo**
entradas E-CORP-10 y E-CORP-11 y los `processed_sha256` ya existentes. Prohibido alterar
ningun valor historico.

**Informe**: `reports/v21_p2_01_niveles_limpieza.md` con, en este orden: tabla de las tres
condiciones (tokens, documentos, tipos, segundos, sha256, cobertura global y por grupo,
fuga), tabla de las nueve evaluaciones, matriz 2x2, IC95, las 11 rebanadas, el
`epsilon_determinismo`, y el veredicto con el criterio que decide. Figura
`fig_58_niveles_limpieza_v21.png`: barras de `corrected_cv` de las nueve evaluaciones con
la linea de V03 y el IC95 de cada fusion, mas un panel con la cobertura por tipo y por
token de las tres condiciones.

**`experiments.csv`**: ver seccion 11.

---

## 7. MEDIUM: catalogo de transformaciones, con hipotesis y prediccion

Regla de la fase: **cada transformacion entra con su hipotesis escrita y una prediccion
refutable**. Ninguna entra "a ciegas". Y hay una restriccion dura de contenido:

> MEDIUM **no** elimina negaciones, **no** elimina puntuacion emocional, **no** elimina
> mayusculas y **no** colapsa elongaciones. En toxicidad esos cuatro fenomenos pueden
> contener senal, y hay evidencia directa de que la senal esta donde la mayoria de
> transformaciones la destruyen: EBD-07 midio que la fusion mejora **+0.0368** en la
> rebanada de elongaciones, la mayor de las once.

### 7.1 Candidatos evaluados (8 decisiones, 2 implementadas)

| # | Transformacion candidata | Hipotesis | Decision | Motivo |
|---|---|---|---|---|
| **T1** | Tratamiento del marcado residual: segunda pasada de `html.unescape`, 32 pasadas de plantilla en vez de 8, y descarte de lineas que son campos de plantilla | El marcado codificado por entidades evade los pases literales de LIGHT: `{{Infobox}}` escrito como `&#123;&#123;Infobox&#125;&#125;` no casa con `RE_PLANTILLA` y solo se materializa en el paso 10 de `limpiar_marcado`, **despues** del cual no queda ningun pase de eliminacion. Ademas, 8 pasadas no bastan para plantillas anidadas profundas. El dano no es la puntuacion, que el tokenizador ya elimina, sino los **nombres y valores de campo** que se cuelan como lexico de esquema (`imagedesc`, `pie de foto`) y consumen frecuencia y `bucket` | **IN** | Mecanismo identificado y verificable, y el step 0 lo cuantifica antes de gastar unidades |
| **T2** | Normalizacion de variantes Unicode (NFC) | **Hallazgo verificado en esta fase**: `RE_CARACTERES_PERMITIDOS = [^\w\s']` con `re.UNICODE` **no** incluye las marcas combinantes (categoria `Mn`), de modo que **borra en silencio los acentos de todo caracter en forma descompuesta**. Comprobado: `unicodedata.normalize("NFD", "espanol con enye")` pasa por el filtro de LIGHT y sale como `espan ol nin o`, es decir **un token partido y el acento perdido**. LIGHT declara conservar las tildes, pero solo lo hace si la entrada llega en NFC. El corpus y la tarea disagree entonces sobre la identidad de los tipos acentuados, y los tipos afectados se parten en dos | **IN** | Defecto real, mecanismo medido, y la transformacion es sin perdida: NFC preserva el significado y no toca negaciones, puntuacion, mayusculas ni elongaciones |
| **T3** | Restauracion de entidades HTML **como texto literal** (`&eacute;` se queda escrito) | Meteria secuencias `&...;` en el vocabulario | **OUT** | Inyectaria ruido de schema en el corpus. Lo aprovechable de las entidades es la resolucion, que es lo que hace T1, no su forma literal |
| **T4** | Colapso de repeticiones de caracteres (elongaciones) | Fusionar `callaaate` con `callate` sube el conteo del tipo fusionado y mejora su estimacion | **OUT** | El coste es mayor que el beneficio: la elongacion es la rebanada que mas aporta (+0.0368 en EBD-07) y en un corpus enciclopedico la elongacion es rara, luego la transformacion cambiaria muy poco del corpus y destruiria la geometria que la tarea lee. Ademas `sample = 1e-4` esta practicamente desactivado, asi que el efecto de aplanar frecuencias es casi nulo y la hipotesis mecanica no se activa |
| **T5** | Eliminacion de boilerplate al estilo C4 (lineas repetidas, lineas cortas sin digitos, documentos con mas del 30 % de URL) | El corpus tiene ruido de plantilla | **OUT** | Por construccion del experimento: las tres condiciones salen del **mismo dump de Wikipedia**, y la fuente se mantiene fija a proposito para que la unica variable sea el nivel de limpieza. El boilerplate que justifico esa limpieza era el de mC4, no el de Wikipedia. Meterlo aqui seria cambiar de fuente y de limpieza a la vez |
| **T6** | Documentos a nivel de parrafo en vez de a nivel de articulo | Ventanas de skipgram que no cruzan parrafos darian contextos mas homogeneos | **OUT** | Con `ws = 3` el uncontesto de ventana abarca 7 tokens, luego una frontera de parrafo cae dentro del uncontesto menos de 3 veces por articulo. El efecto es despreciable frente al ruido, y ademas es otro eje (estructura de documentos, no nivel de limpieza) |
| **T7** | Plegado a ASCII o reconstruccion de tildes | P0 ya lo refuto: el plegado ASCII cubre solo el 8.5 % de los OOV | **OUT** | Evidencia previa y directa. MEDIUM no repite un experimento cerrado |
| **T8** | Corrector ortografico, o eliminacion de stopwords, o `StandardScaler` sobre el denso | Varios | **OUT** | El corrector esta prohibido por la convencion `v5.0-EMB-NORMAL` e inyectaria conocimiento lexico externo. Las stopwords cambian la distribucion que aprende el skipgram y son otro eje. El `StandardScaler` esta descartado por EBD-08 (-0.0871) |

### 7.2 Implementacion exacta de T1 y T2

```python
RE_CAMPO_PLANTILLA = re.compile(r"^\w{2,24}\s*=\s*\S", re.UNICODE)

def transformar_medium(marcado: str) -> str:
    texto = limpiar_marcado(marcado)                     # etapa LIGHT, sin cambios
    texto = html.unescape(html.unescape(texto))          # T1: segunda pasada de entidades
    texto = unicodedata.normalize("NFC", texto)          # T2: NFC antes de volver a limpiar
    texto = RE_COMENTARIO.sub(" ", texto)                # T1: segundo ciclo de eliminacion
    texto = RE_REF.sub(" ", texto)
    texto = RE_ETIQUETA_HTML.sub(" ", texto)
    texto = RE_TABLA.sub(" ", texto)
    for _ in range(32):                                  # T1: 32 pases en vez de 8
        nuevo = RE_PLANTILLA.sub(" ", texto)
        if nuevo == texto:
            break
        texto = nuevo
    texto = RE_ENLACE_ARCHIVO.sub(" ", texto)
    texto = RE_URL_SIMPLE.sub(" url ", texto)
    lineas = []
    for linea in texto.splitlines():
        limpia = linea.strip()
        if not limpia or limpia.startswith("="):
            continue
        limpia = limpia.lstrip("*#:; ")
        if RE_CAMPO_PLANTILLA.match(limpia):             # T1: descarte de campo de plantilla
            continue
        limpia = RE_CARACTERES_PERMITIDOS.sub(" ", limpia)
        limpia = unicodedata.normalize("NFC", limpia)    # T2: NFC tras la whitelist
        limpia = RE_ESPACIOS.sub(" ", limpia).strip()
        if limpia:
            lineas.append(limpia)
    return "\n".join(lineas)
```

Notas de justificacion que el implementador debe respetar alpie:

1. **Por que T2 se aplica dos veces.** NFC es idempotente, y la segunda llamada, **despues**
   de la whitelist, cubre un caso concreto: al borrar un span de plantilla puede quedar
   una letra base pegada a una marca combinante que la whitelist ya habia eliminado. Es
   una defensa, no una segunda transformacion distinta.
2. **Por que T1 repite el ciclo.** `html.unescape` se ejecuta en el paso 10 de
   `limpiar_marcado` y **no hay ningun paso de eliminacion de plantillas despues**. Ese es
   el hueco que T1 tapa.
3. **Riesgo aceptado y declarado de T1**: `RE_CAMPO_PLANTILLA` tambien puede descartar
   una linea legitima de pseudocodigo o de formula que empiece por `clave =`. Es aceptable
   porque el step 0 mide `fraccion_lineas_campo_plantilla` y porque el step 3 mide el
   numero de tipos con acento y la cobertura, de modo que el dano, si lo hay, queda
   registrado y no es invisible.
4. **Lo que MEDIUM conserva de LIGHT, sin tocar**: los marcadores `url`, `hashtag`,
   `usuario` y `numero`; la whitelist; el dedup por sha1; el minimo de 20 tokens;
   `RE_TOKEN`; la seleccion de articulo; y las cuatro prohibiciones de la seccion 7.

---

## 8. Igualdad de escala y covariables que se registran

**Regla**: los tres corpus se truncan a la misma cantidad de tokens con el mismo
mecanismo (lineas completas hasta superar 10.000.000), de modo que la comparacion no
mezcle limpieza con tamano.

**El problema real**: RAW infla el recuento de tokens por articulo, porque dentro del
marcado hay palabras que LIGHT borra. Para alcanzar 10M tokens, **RAW consumira menos
articulos** que LIGHT. Igualar tokens e igualar articulos son cosas incompatibles aqui.

**Decision**:

1. La unidad de emparejamiento es el **token**, fijada en 10.000.000 para las tres.
2. La covariable que se registra y se reporta como concomitantemente es
   `n_articulos_consumidos` (con `documentos`, `tipos`, `tokens_por_documento` y
   `articulos_leidos`). Se llama `covariable_documentos` en el informe y en las notas de
   `experiments.csv`, y se escribe explicitamente que **es una consecuencia del nivel de
   limpieza y no una variable controlada**.
3. Se registra tambien `lineas_duplicadas_residuales`: numero de lineas identicas que
   quedan en el `tokens_10M.txt` final. El dedup de V5 actua **antes** de la normalizacion,
   luego dos textos distintos que normalizan al mismo se escapan; es un defecto latente de
   LIGHT que P2 **mide** en las tres condiciones en vez de cambiarlo, porque cambiar la
   regla de dedup seria una quinta variable. Si el numero es distinto entre condiciones,
   se reporta como candidata a fase posterior.
4. **No** se construye un cuarto corpus a igualdad de articulos. Costaria una unidad mas
   del presupuesto y mezclaria dos preguntas. Se declara como `Open Question` 2.

---

## 9. Controles obligatorios: tabla resumen

| # | Control | Referencia | Asercion | Bloquea |
|---|---|---|---|---|
| C1 | `processed_sha256` de LIGHT | E-CORP-01 en el manifiesto | **exacto** | todo lo demas |
| C2 | Cobertura global LIGHT `train+test` | tipo 0.6839, token 0.8833 | **exacto, 4 decimales** | todo lo demas |
| C3 | Fuga de 8-gramas, train y test | 0 textos con coincidencia | **exacto, 0 en las 3** | uso de ese corpus |
| C4 | FUSION-LIGHT | 0.782959 | `abs <= 0.0015` | comparar RAW y MEDIUM |
| C5 | EMB-LIGHT-LR | 0.6694 | `abs <= 0.0050` | comparar, con diagnostico |
| C6 | Control V03 | 0.7772114525165721 | exacto a 6 decimales | todos los deltas |
| C7 | Firma de LIGHT | colisiona con la WS-3 registrada | **debe colisionar** | interpretacion de identidad |
| C8 | Firmas de RAW y MEDIUM | 0 colisiones | **exacto** | unicidad de la comparacion |
| C9 | Marcadores `url`/`hashtag`/`usuario`/`numero` | RAW 0.0000, LIGHT 1.0000 | exacto | interpretacion de RAW |
| C10 | Tipos de train con vector | 6.059 de 6.059 en las 3 | exacto | nada, es diagnostico |
| C11 | Ids del OOF canonico = ids del desarrollo | igualdad exacta | exacto | comparacion pareada |
| C12 | Desarrollo / holdout | 1.483 / 248 | exacto | nada, es integridad |
| C13 | `-lr` y `-sample` ausentes de la linea de comando | ausentes | exacto | trazabilidad de la arquitectura |
| C14 | `epsilon_determinismo` | se registra siempre | `> 0` implicito | se usa como suelo de ruido |

---

## 10. Metrica, criterio de decision y como se decide

### 10.1 Metrica

`corrected_cv` **v2**: macro-F1 sobre las tres clases positivas, calculada sobre las
predicciones fuera de pliegue del desarrollo (1.483 filas) con `pliegues_corregidos` y
`holdout_v2`, y **umbrales congelados de V03** (0.44 / 0.51 / 0.46). Es la unica
metrica que se reporta para decidir, y la razon es que es la que se ha usado en las 20
filas de embeddings, de modo que los tres niveles son comparables con el historico sin
traducir metricas. El `corrected_cv` con umbral propio de cada etiqueta se reporta como
diagnostico y **nunca** decide: si se optimizara por condicion, la comparacion mediria
optimismo y no limpieza.

### 10.2 Los dos criterios, declarados los dos, y cual decide

**Criterio de V5 (D-039, pre-registrado, dos niveles)**: `delta_vs_V03 >= +0.005` **Y** IC95
que excluya el 0. Bajo este criterio, las 14 configuraciones de v5 quedaron fuera.

**Criterio de V21 (seccion 14 del encargo)**: una mejora de leaderboard valida es
suficiente para considerar adopcion; no se exige que el IC95 excluya el 0 como condicion
absoluta.

**P2 mide en local, no en el leaderboard**, y no gasta submissions. Por eso el criterio
que **decide en P2** se declara de forma explicita y conservadora, y es este:

Una condicion se declara **CANDIDATE** solo si se cumplen **las tres** condiciones:

- **(G1) magnitud**: `delta_vs_V03 >= +0.005`, el mismo suelo que v5 fijo. Se declara que
  ese suelo esta por debajo de 1 error estandar del leaderboard (SE del LB con 1.154
  filas = 0.01219), o sea que **G1 es necesario y no suficiente**, y que un `deltaOOF` de
  +0.005 esta dentro del ruido del LB por un factor de 2.4.
- **(G2) complementariedad**: `solo_candidato > solo_referencia` en la matriz 2x2 frente a
  V03, o sea reparacion neta de celdas positiva. Es la exigencia que ya aplico EBD-07 y
  que separa una mejora real de un aflorar de hallucinations.
- **(G3) por encima del ruido medido**: `delta_vs_V03 > epsilon_determinismo`, el suelo que
  midio el reentrenamiento de LIGHT. Sin este tercer criterio, cualquier diferencia entre niveles
  podria ser no-determinismo de fastText con 16 hilos.

**El IC95 se reporta siempre, en las nueve evaluaciones, y se usa como desempate y como
medida de solidez de la evidencia; deja de ser puerta.** Eso es exactamente lo que cambio
la seccion 14, y queda escrito para que nadie lo lea al reves.

**Lo que P2 no puede hacer**: autorizar una submission. Aunque una condicion sea
CANDIDATE, la decision de enviar pertenece a otra conversacion y requiere: (i) las cuatro
unidades de submission intactas, (ii) el recuento de celdas favorable, y (iii) el
antecedente de 19 de 21 submissions por debajo de V03 y 7 de 7 a la baja en V16-V20. El
resultado maximo de P2 es `CANDIDATE, no adopted`, igual que las 20 filas anteriores.

**Ademas**, para que "el mejor nivel" sea una afirmacion con sentido, se exige que el
ganador supere al mejor nivel ya conocido (LIGHT, 0.7830) por
`max(epsilon_determinismo, 0.0015)`. Un nivel que solo mejora a V03 pero no a LIGHT no es
unaaportacion de la limpieza: es ruido.

---

## 11. Registro en `experiments.csv`

Formato: el archivo tiene **23 columnas** y su encabezado es el de
`experiments.csv` actual. **No se anade, no se quita y no se reordena ninguna columna**, y
no se escribe ningun valor con acentos. Verificar con `head -1 experiments.csv` antes de
escribir y comparar el numero de campos de cada fila con el del encabezado.

Convencion de `experiment_id` que siguen las filas `EBD-*`: `EBD-` mas un numero de dos
digitos con cero a la izquierda, correlativo, y un sufijo en mayusculas con guion solo
cuando la fila es un andamiaje y no un experimento (`EBD-00-SCAFFOLD`). La serie `EBD-`
llega hasta `EBD-26`. P2 abre con tres filas:

| Fila | Contenido | `script` |
|---|---|---|
| `EBD-27` | Question: `existe una diferencia medible entre niveles de limpieza del corpus?` Hypothesis: `H-P2`. `preprocessing`: los tres niveles, con su `preprocessing_version` y su `processed_sha256`. `validation`: auditoria de fuga de 8-gramas y cobertura por tipo, token y grupo. `result`: tabla de los tres corpus y sus controles C1, C2, C3, C9. `decision`: si el pre-flight pasa o si P2 para por D1 | `experiments/embedding_24_corpus_niveles_v21.py` |
| `EBD-28` | Question: `cual nivel de limpieza maximize la aportacion complementaria?` Hypothesis: `H-P2 frente a H0-P2`. `feature_groups`: 27.431 TF-IDF de V03 mas 200d en L2 por documento. `representation`: los tres `.vec` con su `sha256`. `model`: `LogisticRegression(C=2.0, balanced, liblinear)`. `hyperparameters`: los 13 campos de la firma. `validation`: `corrected_cv` v2 con umbrales congelados, IC95 pareado de 10.000 remuestreos, matriz 2x2 y 11 rebanadas. `training_time`: suma de los tres tiempos de entrenamiento en segundos. `result`: las nueve evaluaciones con `epsilon_determinismo`. `decision`: veredicto con G1, G2, G3 y el criterio que decide | `experiments/embedding_25_niveles_downstream_v21.py` |
| `EBD-29` | Question: `la limpieza es la palanca, o el nivel es inerte?` Question de cierre. `result`: consolidacion de las tres filas anteriores mas el readout de las 11 rebanadas. `decision`: `H0-P2` o `H-P2`, y si procede el cierre de la linea de limpieza para el resto de V21 | `experiments/embedding_25_niveles_downstream_v21.py` |

Ademas del registro en la tabla: el pre-flight se registra como artefacto
`reports/v21_p2_00_preflight.json` y su numero aparece en las notas de `EBD-27`. Si el
pre-flight detiene la fase, **se escriben igualmente `EBD-27` y `EBD-29`**, con `EBD-28`
en `NO EJECUTADO` y el motivo del criterio de parada. Una fase detenida se registra igual
que una fase ejecutada.

---

## 12. Reproducibilidad: que se escribe

`reports/v21_p2_01_niveles_limpieza.json` contiene, por condicion y de forma plana:

- `semilla_fasttext = 42`, `hilos = 16`.
- `preprocessing_version`, `processed_sha256`, `bytes`, `tokens`, `documentos`, `tipos`.
- `tokenizer`: el **literal** de `RE_TOKEN` y de cada regex usado, copiado, no referenciado.
- `arquitectura`: los 13 campos de la firma, mas `alpha = 0.05` y `sample = 1e-4`
  marcados como `defecto_del_binario: true`, mas el `sha256` del ejecutable fastText.
- `linea_comando`: la cadena exacta devuelta por `entrenar_fasttext`.
- `downstream`: `LogisticRegression(C=2.0, class_weight="balanced", solver="liblinear",
  max_iter=1000, random_state=42)`, `dimension = 200`, `ventana = 3`, `epocas = 5`,
  `min_count = 3`, `subpalabras = (3, 6)`, `negativos = 10`, `bucket = 200000`.
- `protocolo`: `pliegues` de `pliegues_corregidos`, `holdout` de `holdout_v2`,
  `n_desarrollo = 1483`, `n_holdout = 248`, `umbrales = [0.44, 0.51, 0.46]`,
  `metric = "corrected_cv v2"`.
- `bootstrap`: `repeticiones = 10000`, `semilla = 20260925`.
- `artefactos`: `sha256` y `bytes` de `.vec` y de `.bin` por condicion, mas el `sha256` del
  `.vec` de WS-3 de V5 como referencia no utilizada.
- `controles`: el resultado booleano y el valor de cada uno de C1 a C14.
- `epsilon_determinismo`.
- `entorno`: version de Python, de scikit-learn, de numpy y de pandas, y la del binario.

---

## 13. Costes y presupuesto

| Paso | Coste esperado | Observacion |
|---|---|---|
| Step 0, pre-flight | 10 a 20 s | 5.000 paginas en memoria, sin escritura |
| Step 1, tres corpus | 15 s por corpus, unos 2 a 3 min en total | E-CORP-01 tardo **13.8 s** en construir sus shards; el dump ya esta descargado y verificado |
| Step 2, fuga | 2 a 3 min por corpus, 6 a 9 min en total | 10M hashes de 8-gramas, `np.unique` y `searchsorted` sobre 2.885 textos |
| Step 3, cobertura | 30 a 60 s por condicion | las proyecciones ASCII y esqueleto son conjuntos precalculados |
| Step 4, control A y B | segundos | sin coste |
| Step 5, tres entrenamientos | **83 s** por modelo con `ws=3` segun EBD-19, unos 4 a 5 min en total | el registro de v5 da 43 s a 5M y 577 s a 50M, luego 10M con ventana 3 esta en ese orden de magnitud |
| Step 5, sifts | 10 a 20 s por modelo | volcado de vectores de 6.059 tipos |
| Step 7, nueve evaluaciones | 3 a 6 min | 9 x (5 pliegues x 3 cabezas); domina el reajuste del TF-IDF |
| sha256 de 6 artefactos grandes | 5 a 10 s | 3 `.vec` de ~175 MB y 3 `.bin` de ~322 MB |
| **Total de pared** | **20 a 30 min**, con margen declarado de 45 min | |
| **Espacio en disco** | unos 1.6 GB mas: 3 corpus (~200 MB), 3 `.vec` (~525 MB), 3 `.bin` (~970 MB) | 434 GB libres, no hay que borrar nada |
| **Presupuesto de la fase** | **3 de 8 unidades** | una unidad por entrenamiento; no hay una cuarta unidad reservada |

El gasto es exactamente 3 unidades y no se puede gastar una cuarta: si tras LIGHT, RAW y
MEDIUM hiciera falta una cuarta variante, P2 se cierra con lo que tenga. No hay unidad de
reserva en P2.

---

## Boundaries

### Always

- Importar y reutilizar las funciones existentes; nunca reimplementarlas. Las tres
  condiciones se construyen **parametrizando** el pipeline de V5, no copiandolo.
- Mantener el lado task **literalmente identico** en las tres condiciones: solo
  `normalizar_emb_normal` + `tokenizar` + mean pooling + L2 por documento.
- Mantener la arquitectura de la seccion 6.6 exactamente, en las tres condiciones.
- Mantener los umbrales congelados de V03 y `corrected_cv` v2 como unica metrica que decide.
- Ejecutar el step 0 antes de construir, y el step 4 antes de entrenar los otros dos.
- Registrar los tres `processed_sha256`, los tres `sha256` de `.vec` y los tres
  `model_sha256`, y las 13 firmas.
- Reportar las once rebanadas y la matriz 2x2 aunque la condicion pierda.
- Escribir el informe de la fase **tambien cuando la fase se detiene**, con el motivo.
- Actualizar el manifiesto de corpus **anadiendo** campos y entradas, sin alterar ningun
  valor historico.

### Ask first

- **Cualquier gasto de submission.** P2 no propone enviar; si una condicion sale
  CANDIDATE, la decision de enviar se consulta aparte, con el recuento de celdas a la
  vista y sin consumir ninguna de las 4 unidades por iniciativa propia.
- **Cualquier cuarto entrenamiento** o cualquier variante adicional de MEDIUM, aunque
  el protocolo diga "por si acaso". El presupuesto son 3 unidades.
- **Cambiar la fuente del corpus** de Wikipedia ES a cualquier otra (registro social, mC4,
  mixto). Eso es otra hipotesis y otra fase.
- **Cambiar la arquitectura** de la seccion 6.6, incluido `ws`. El 3 esta congelado porque
  es el punto de operacion de los valores ancla.
- **Tocar `emb_normal/`**, el `.vec` de V5, el OOF canonico de V03 o cualquier
  `submissions/`.
- **Eliminar artefactos** para recuperar disco, incluidos los `.bin`. Hay 434 GB libres.

### Never

- **Nunca** escribir en `data/external/corpora/wikipedia_es/emb_normal/`. Es el corpus de
  referencia del que dependen 6 modelos ya medidos.
- **Nunca** cargar `fasttext_wiki_10m_ws_3.vec` ni ningun otro `.vec` cacheado de V5 en el
  pipeline de P2. Se puede **leer su `sha256`** como referencia y nada mas.
- **Nunca** usar test para entrenar, para ajustar umbrales, para seleccionar el nivel
  ganador ni para ninguna decision. Test solo se lee en la auditoria de fuga, y solo como
  conjunto de busqueda, nunca como fuente de aprendizaje.
- **Nunca** tocar los umbrales 0.44 / 0.51 / 0.46, ni ajustarlos por condicion, ni
  reportarlos como "ajustados" en la metrica que decide.
- **Nunca** aplicar `StandardScaler`, `Ridge`, `SGD`, apilamiento ni ponderacion al bloque
  denso. EBD-08 y EBD-09 cerraron las tres lineas.
- **Nunca** eliminar negaciones, puntuacion emocional, mayusculas ni elongaciones del
  corpus en ninguna condicion, ni anadir un corrector ortografico, ni usar una lexicon
  externa de insultos (ambiguedad de D-030 que se asume en el sentido mas restrictivo).
- **Nunca** usar transformers, LLM, RoBERTa, BERT ni ningun embedding preentrenado.
- **Nunca** reescribir ni versionar scripts previos, ni borrar resultados previos.
- **Nunca** declarar un resultado sin su `sha256` de corpus y su firma de 13 campos.
- **Nunca** presentar un resultado sin su control de cordura: si C1 a C6 no estan
  verificados, el resultado no existe.

---

## Success Criteria

Medibles, con umbral. Todos deben cumplirse para que P2 se de por ejecutada.

| # | Criterio | Umbral |
|---|---|---|
| SC1 | Corpus construidos | 3 archivos `tokens_10M.txt`, cada uno con tokens en `[10.000.000, 10.010.000]`, con `sha256` y `preprocessing_version` registrados |
| SC2 | Reproduccion del corpus LIGHT | `sha256` identico al `processed_sha256` de E-CORP-01 (C1) |
| SC3 | Reproduccion de la cobertura de LIGHT | `type_coverage = 0.6839` y `token_coverage = 0.8833` en `train+test`, a 4 decimales (C2) |
| SC4 | Auditoria de fuga | `textos_con_coincidencia == 0` para train y test en las 3 condiciones (C3) |
| SC5 | Control de reproduccion de metrica | FUSION-LIGHT con `abs(d - 0.782959) <= 0.0015` y EMB-LIGHT-LR con `abs(d - 0.6694) <= 0.0050`, y `epsilon_determinismo` registrado (C4, C5) |
| SC6 | Control V03 | `corrected_cv` del control V03 = `0.7772114525165721` a 6 decimales (C6) |
| SC7 | Firmas | la firma de LIGHT colisiona con la WS-3 registrada; las de RAW y MEDIUM son distintas entre si y no colisionan con ninguna firma ni `.vec` existentes (C7, C8) |
| SC8 | Marcadores | `url`, `hashtag`, `usuario` y `numero` ausentes del vocabulario de RAW y presentes en el de LIGHT y MEDIUM (C9) |
| SC9 | Tipos de train con vector | 6.059 de 6.059 en las 3 condiciones (C10) |
| SC10 | Evaluaciones downstream | 9 evaluaciones completadas, cada una con `corrected_cv` congelado, `delta_vs_V03`, `std_pliegues_0.5`; las 3 de fusion con IC95 pareado de 10.000, matriz 2x2 con sus 4 conteos y deltas en las 11 rebanadas |
| SC11 | Covariables | `covariable_documentos`, `tokens_por_documento` y `lineas_duplicadas_residuales` reportadas por condicion, con la advertencia de que la primera es consecuencia y no variable |
| SC12 | Integridad | 1.483 filas de desarrollo, 248 de holdout, ids del OOF canonico iguales a los del desarrollo, `-lr` y `-sample` ausentes de la linea de comando |
| SC13 | Artefactos | informe `.md`, `.json`, `.csv` de resultados, `.csv` de rebanadas, figura `fig_58` y 3 filas nuevas en `experiments.csv` con las 23 columnas |
| SC14 | Presupuesto | gasto exacto de 3 unidades, registrado en el JSON |
| SC15 | Veredicto | una afirmacion explicita a favor o en contra de H-P2, con el criterio que decide nombrado, o la declaracion de parada por uno de los criterios D1 a D4 |

Criterio de calidad, ademas: **las tres condiciones deben medirse con el mismo codigo de
downstream**. Si algun numero viene de un camino de codigo distinto, la comparacion queda
anulada aunque los valores parezcan coherentes.

---

## Open Questions

1. **Cuanta masa de marcado sobrevive al tokenizador.** Es la pregunta de mayor
   incertidumbre de P2 y la responde el step 0, pero su respuesta determina si P2 tiene
   margen para discriminar. Si `fraccion_documentos_con_marcado_residual` de LIGHT fuera
   menor que 0.02, la hipotesis estaria muerta antes de gastar una unidad. Esta es la
   fuente principal de riesgo de la fase.
2. **Que pasa si se empareja por articulos en vez de por tokens.** Concretamente: cuanto
   valdria un cuarto corpus RAW a igualdad de articulos con LIGHT, que tendria mas de
   10M tokens. Queda fuera del presupuesto de 3 unidades y se registra como pregunta
   abierta, no como trabajo silencioso.
3. **Cuanto mueve el protocolo reentrenar sin cambiar nada.** `epsilon_determinismo` sale
   de LIGHT, pero es una sola muestra con 16 hilos. Si sale grande, todas las familias de
   v5.1 estan mas cerca del ruido de lo que sugieren sus dispersiones internas, y eso
   seria un hallazgo de la fase, no un inconveniente.
4. **Si MEDIUM resulta indistinguible de LIGHT.** Habria que decidir si es porque T1 y T2
   no tienen masa en este dump, o porque la masa existe pero no aporta. El step 0 y el
   step 3 dan los dos datos; la decision de si abrir un P2-bis con transformaciones
   distintas se toma despues, con el informe delante.
5. **Como se traduce "mejor nivel" a una submission.** No lo responde P2. Depende del
   recuento de celdas, del numero de unidades de submission y del estado de las otras fases
   de V21.

---

## Decision de parada

P2 se cierra **sin llegar a conclusion**, y con resultado valido y documentado, si se
cumple cualquiera de estos cuatro criterios. Todos son decididos por datos, no por
juicio.

**D1 - Separabilidad nula (pre-flight).** Si en el step 0
`fraccion_documentos_con_marcado_residual` de LIGHT es menor que 0.02 **y**
`fraccion_caracteres_nfd` es menor que 0.001, entonces, con este tokenizador, los tres
niveles de limpieza no son separables: la mayor parte del marcado ya la elimina `RE_TOKEN`
y la whitelist, no `limpiar_marcado`. **Se para antes de construir y de entrenar.** El
resultado es una refutacion de la premisa de la seccion 5 del encargo, con su causa
identificada, y se escribe igualmente `EBD-27` y `EBD-29` con `EBD-28` en NO EJECUTADO.

**D2 - El ancla no reproduce (control de cordura).** Si falla cualquiera de C1 a C6, la
comparacion **no se hace**. Se para, se investigan las aserciones fallidas y se reporta el
fallo con su evidencia. No hay un "a pesar del control": una comparacion de niveles de
limpieza sin ancla reproducible mide la deriva del protocolo, no la limpieza.

**D3 - Fuga en un corpus.** Si alguna condicion da `textos_con_coincidencia > 0`, ese
corpus se descarta. Si el afectado es RAW o MEDIUM, la comparacion de tres niveles se
queda incompleta y P2 se cierra como **inconcluyente**: comparar dos niveles no responde a
la pregunta de tres. Si el afectado es LIGHT, aplica D2.

**D4 - Ruido por encima de la dispersion.** Si **ninguna** de las tres condiciones supera a
LIGHT por `max(epsilon_determinismo, 0.0015)`, y ademas los tres `delta_vs_V03` estan
separados entre si por menos de `epsilon_determinismo`, P2 cierra como `H0-P2`: el nivel
de limpieza **no es la palanca**, la variable esta dentro del ruido, y la linea de
limpieza se da por cerrada para el resto de V21 sin gastar una unidad mas. Este es el
resultado mas probable segun todo lo que P0 y v5.1 han medido, y por eso esta escrito
antes de ejecutar y no despues.

**Ademas, y sin que sea un criterio:** si se agota el presupuesto de 3 unidades y la
comparacion esta a medias, P2 publica lo que tenga, con la parte no ejecutada marcada como
tal, y la siguiente fase decide si completa el triangulo.
