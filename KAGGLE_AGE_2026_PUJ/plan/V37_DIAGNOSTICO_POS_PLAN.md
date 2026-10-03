# SPEC V37 · DIAGNOSTICO `P(POS | edad)`

- **Fecha**: 2026-10-01
- **Fase**: V37-DIAG-POS (fase CPU pura, D-007)
- **Estado**: ESPECIFICACION. Nada implementado. Nada ejecutado.
- **Intencion**: `plan/00_intent_v37.md` (confirmada por Carlos el 2026-10-01)
- **Campeon congelado**: V30 + plug-in, `corrected CV macro-F1 = 0.6159732` (verificado de nuevo en
  esta spec: `0.6159732487657944`, ver seccion 1.3). Pesos `[1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]`.
- **Decision de alcance ya tomada por esta spec**: el paso (d) **NO se ejecuta en su parte predictiva**.
  Veredicto de la auditoria de novedad en la seccion 6. Decision por defecto: **conservar el campeon**.

---

## 1. Objective

### 1.1 Que se construye

Una cadena **exclusivamente descriptiva y de CPU** que responde una sola pregunta, sin tocar el campeon:

> ¿La distribucion condicional `P(tag POS | edad)` contiene un patron **estable** y **no redundante**
> en este corpus, y cuanto vale ese patron como informacion aislada?

Cinco entregados, en orden gateado:

1. **A** Tabla `P(tag | edad)` de 6 clases x 18 etiquetas UPOS, con soporte, cobertura, control de
   longitud, y **cota superior de potencia discriminativa** por clasificador plug-in bayesiano cerrado.
2. **B** Estabilidad de esa tabla entre los 5 folds canonicos, entre `md` y `lg`, con intervalos por
   grupos de duplicados.
3. **C** Contraste con lo ya medido (D-026, V14, V31, P1a, P2-P5): ¿es nuevo o es redescripcion?
4. **D** Auditoria de novedad, honesta y argumentada, del canal de transiciones POS de primer orden
   condicionado por edad `P(POS_{i+1} | POS_i, edad)`. Incluye la medicion **descriptiva y gratis**
   de la divergencia entre clases de esa matriz, que es la cantidad que decide la hipotesis.
5. **E** Gate de decision antes de cualquier corrida predictiva, con veredicto escrito.

### 1.2 Por que

Carlos lo pide textualmente: *"estudiar las probabilidades condicionales de cada tipo de palabra, es
lo que se me ocurre, para hallar algun patron"*. La pregunta es si el patron existe y si es rentable.
Un diagnostico que solo dijera "existe un patron" no seria accionable; por eso el paso (a) incluye
obligatoriamente la cota de potencia discriminativa (seccion 4.3). Sin ella, `P(POS|edad)` es una
tablanice para un informe, no una decision.

### 1.3 Metrica de exito del DIAGNOSTICO (no es metrica de leaderboard)

Veredicto con evidencia, uno de dos:

| Veredicto | Significado |
|---|---|
| `PATRON_ESTABLE_NO_REDUNDANTE` | Hay diferencias por clase que sobreviven el ruido muestral, la variacion entre folds, el cambio de anotador y el control de longitud; y la cota de potencia no queda por debajo de lo que P2 ya exploto |
| `DIAGNOSTICO_CERRADO` | Cualquiera de los criterios de la seccion 8.2 falla. Se cierra la familia "POS marginal" con numero, se conserva el campeon y se escribe acta |

**Correccion de referencia obligatoria.** El proyecto ha citado siempre *"suelo de azar 0,1667"*
(D-026, D-041, D-044). Con las clases casi perfectamente balanceadas de este corpus
(28.863 / 28.676 / 28.347 / 28.230 / 28.060 / 27.160; ratio max/min = **1,0627**), el suelo
**estricto** es el de la clase mayoritaria, y es otro:

| Referencia | macro-F1 | verificado |
|---|---|---|
| Azar uniforme (6 clases equiprobables) | 0,166667 | analitico |
| **Clase mayoritaria constante (suelo estricto)** | **0,0485421** | calculado de `y_true` |
| Campeon V30 + plug-in | 0,6159732 | recomputado de `v30_prob` x pesos |

Toda cifra POS que se cite en el informe debe bring su suelo. Re-expresado contra el suelo estricto:

| Referencia previa | macro-F1 | + sobre 0,0485 | + sobre 0,1667 |
|---|---|---|---|
| P2 canal `pos_seq` solo (md) | 0,2750921 | +0,2266 | +0,1084 |
| P2 canal `pos_seq` solo (lg) | 0,2725490 | +0,2240 | +0,1059 |
| P1a `P(clase \| token)` bloque solo | 0,3997000 | +0,3512 | +0,2330 |
| Campeon | 0,6159732 | +0,5674 | +0,4493 |

**Rebalanceo: NOAPLICA.** Ratio max/min = 1,0627, muy por debajo de cualquier umbral de desbalanceo
(skill `data-rebalancing`). Ni `class_weight`, ni SMOTE, ni submuestreo. Se declara explicitamente para
que no se introduzca rebalanceo por inercia.

### 1.4 Success criteria de la fase completa

- La cadena corre de punta a punta en **menos de 15 minutos** de reloj en 32 hilos / 31 GB.
- El autonumerico de arranque del proyecto pasa (`src.verificacion_numerica.autotest_numerico`).
- Los 5 artefactos de la seccion 10 existen y son reproducibles bit a bit dada la misma semilla.
- El informe contiene el veredicto, la tabla, los intervalos y la auditoria de novedad. Sin celdas
  abiertas en `plan/V32_REAPERTURA_PLAN.md` tocadas.
- `git diff DECISIONS.md` solo contiene el acta nueva. `v30_prob`, los pesos, la representacion, la
  semilla y los hiperparametros del campeon intactos (verificado por hash en el acta).

---

## 2. Inputs

### 2.1 Artefactos POS congelados (ya existen, NO se reanotan)

| Ruta | Forma | Columnas |
|---|---|---|
| `features/linguistic/anotacion_md_train.parquet` | (169336, 8) | `fila, pos_seq, lemma_pos_seq, morph_seq, dep_seq, ent_seq, n_tokens, n_sents` |
| `features/linguistic/anotacion_lg_train.parquet` | (169336, 8) | idem |

Verificado en esta spec:

- `fila` es el indice de fila: `fila == arange(169336)`, exacto en ambos parquets.
- `len(pos_seq.split()) == n_tokens` en **el 100 %** de las filas, en ambos parquets.
- **18 etiquetas UPOS distintas** en ambos: `ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN
  PUNCT SCONJ SPACE SYM VERB X`.
- Tokens totales: **2.590.877** en md y en lg (identico). Media 15,30 tokens/tweet, mediana 15,
  min 0, max 71.
- **104 filas con `pos_seq` vacio** (`n_tokens == 0`, 0,061 %).
- **`SPACE` esta dentro de la secuencia**: 32.865 ocurrencias en 25.232 filas (14,9 % de las filas).
  Esto **contradice la definicion de D-026**, que excluia `token.is_space`
  (`scripts/phase10_dos_vias.py:113`). La fase C debe resolver y declarar esta discrepancia de
  denominador (seccion 7.4).
- spaCy `es_core_news_md` / `_lg`: autorizado **solo para etiquetas**, nunca vectores (D-076). En V37
  **no se ejecuta spaCy en absoluto**: solo se leen los parquets.

### 2.2 Canonicos (particion y campeon)

`artifacts/p33_scores/todos/`: `y_true.npy`, `fold_idx.npy`, `v30_prob.npy`, `cand_prob.npy`,
`cand_pred.npy`, `ids.npy`, `manifest.json`.

Verificado en esta spec:

- `y_true` (169336,) con `bincount = [28863 28676 28347 28230 28060 27160]`, y coincide exactamente
  con el entero de `age_range` en el orden de `data/raw/age_train.csv` (`ids == train['id']`).
- `fold_idx` con `bincount = [33868 33868 33867 33866 33867]`: 5 folds balanceados, ya agrupados.
- `v30_prob` (169336, 6), filas normalizadas a 1.
- **Ancla verificada**: `argmax(v30_prob * [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128])` da macro-F1
  `0.6159732487657944` == **0.6159732**. El baseline primario del proyecto esta bien y el script
  debe reconfirmarlo en runtime con `[STOP] exit 2` si no reproduce (patron `v36_01_transiciones_clase.py`).
- Referencia anidada 0,6154643 (no se usa en V37: V37 no fusiona nada).

### 2.3 Grupos de duplicados (para los intervalos)

`features/dup_clusters.parquet` (282228, 3): `norm, sig, cluster` (train + test concatenados).

Verificado en esta spec:

- `dup_clusters['cluster'][:169336]` alineado con las filas de train.
- **164.018 clusters** sobre 169.336 filas (5.318 filas duplicadas, 3,14 %).
- **Reproduccion exacta del canonico**: `StratifiedGroupKFold(n_splits=5, shuffle=True,
  random_state=42)` sobre `y_true` y esos clusters reproduce `fold_idx` con coincidencia **1,000000**.
  V37 **no recrea los folds**: los lee del `.npy`. El cluster se usa solo como unidad de bootstrap.

### 2.4 Lo ya medido que esta fase debe CONTRASTAR, no repetir

| Acta | Que midio | Numero |
|---|---|---|
| D-026 | 26 features POS con `es_core_news_md`: 18 proporciones UPOS + `pos_unique` + `pos_entropy` + diversidad de bigramas + de trigramas + `content_ratio` + `function_ratio` + `verb_noun` + `adj_noun` + `pos_tokens` | integracion **-0,0032** |
| D-044 / V14 | 37 ratios agregados (morfologia verbal y nominal, lemas, entidades, sintaxis) | LogReg aislado **0,2409**, LinearSVC **0,2270**; fusion **-0,0015**, 1/5 folds |
| V31 | canales regex sin tagger: pronombres 1P/2P/3P/plural, palabras funcionales, tiempo verbal, negacion, densidad | banco **+0,006820**, integracion **+0,0000642** |
| D-041 / P1a | `P(clase \| token)` con suavizado empirico-bayesiano y cross-fit interno | bloque solo **0,3997**; fusion **-0,0002**, 3/5 folds; **controles permutados empatados** (0,6000 / 0,6003 / 0,6000) |
| V13 P17 / P20 / P19 | pronombres / modalidad / arboles de conjunciones | 0,1918 / 0,1270 / 0,1539 (todos <= azar) |
| D-077 / P2 | `pos_seq` -> TfidfVectorizer(word, 1-3 grams, min_df=5, sublinear_tf) + LogReg C=1.0 por fold | md solo **0,2750921**, lg solo **0,2725490**; integracion **-0,0086357** (md) / **-0,0084168** (lg), **0/5 folds**; control barajado_filas -0,0039528 (md), barajado_intra -0,0080310 |
| D-078 / D-079 / D-080 | NER estructural, lemma_POS, texto tipado | cerrados; 0 rescates de 9780 errores de alta confianza |
| D-088 / V36-E1 | bloque de transiciones de clase de caracter (C1-A), senal real y especifica por documento | cerrada: *"el campeon ya la expresa mejor"* |
| D-087 | techo alcanzable LB **0,642** | cerrado |

**Tesis del plan**: si `P(tag|edad)` resulta ser la misma tabla que D-026 features como ratios, y su
cota de potencia no supera lo que P2 ya exploto como n-gramas, entonces V37 **no abre nada**: cierra.
Ese cierre, con numero, es un resultado valido y es el resultado mas probable.

---

## 3. Fases y gates

```
FASE A  Diagnostico descriptivo  P(tag|edad) + cota de potencia      -> GATE G1
FASE B  Estabilidad: folds + md/lg + longitud + IC por grupos        -> GATE G2
FASE C  Contraste con lo ya medido (D-026, V14, V31, P1a, P2-P5)      -> GATE G3
FASE D  Auditoria de novedad del canal de transiciones (+ medicion    -> GATE G4
        descriptiva de la divergencia entre clases, NO predictiva)
FASE E  Gate de decision antes de cualquier corrida predictiva       -> ACTA
```

Reglas de transicion:
- Una fase no empieza hasta que el artefacto JSON de la anterior esta escrito y el gate evaluado.
- Un gate fallido **no** se reintenta con ajustes. Se documenta y se pasa a la siguiente fase con el
  veredicto que implica.
- Ninguna fase toca `v30_prob`, los pesos del plug-in, ni produce submission.

---

## 4. Process

### 4.0 Pre-registro congelado (antes de calcular nada)

Script `scripts/age_linguistic_13_preregistro_v37.py` escribe
`experiments/age_linguistic_13_preregistro_v37.json` con:

- SHA256 de los 2 parquets (anclados a los valores de P2: `dacbc3055ccde0d9a53f335f65b63df666b2ac1a4eee4a94e5b46fd7cd8c5745`
  md, `81287d541c082ea888ad1b5d06b83680bc5078c624c2da32f0473139d67a739c` lg) y de los 6 `.npy` canonicos
  (anclados al `manifest.json`: `y_true b1c928f9...`, `fold_idx 74d18330...`, `v30_prob 3d773524...`).
- SHA256 de `features/dup_clusters.parquet`.
- `etiquetas_pos`: las 18 UPOS, **declaradas a priori** desde la union observada. No hay seleccion.
- `vocab_seleccion`: `"ninguna"`. No hay `min_df`, no hay seleccion de caracteristicas.
- `suavizado`: `{"tipo": "laplace", "alpha": 1.0, "nota": "valor fijo declarado a priori; no hay grilla"}`
- `priores`: `"frecuencia de clase empirica del train de cada fold"`.
- `modelo_cota`: `"naive bayes multinomial sobre proporciones por token; sin ajustar, sin tuneo, sin grilla"`.
- `semillas`: `{"bootstrap_cluster": 20261010, "permute_control": 20261010, "hash_agrupador": 0}`
- `umbrales`: los de la seccion 8.2, literales.
- `referencias_previas`: la tabla de la seccion 2.4, copiada literal.
- `piso_estricto_macro_f1`: 0,0485421. `azar_uniforme_macro_f1`: 0,166667.
- `baseline_primario`: 0,6159732. `pesos_plugin`: los 6 congelados.
- `no_hacer`: la lista completa de la seccion 8, Boundaries/Never.

### 4.1 FASE A · `P(tag | edad)` descriptivo

**A1 · Perfil por clase.** Para cada clase `c` en 0..5 y cada etiqueta `t` en las 18:

```
n[c,t] = suma de ocurrencias de t en pos_seq de las filas con y == c
p[c,t] = n[c,t] / sum_t' n[c,t']
```

Calculado de tres formas, todas declaradas a priori y **las tres se reportan** (no se elige una):

1. **Crudo**: sobre todas las filas de la clase.
2. **Estandarizado por longitud**: se estandariza la distribucion de longitudes de cada clase a la
   distribucion de longitudes del corpus. Concretamente, se estratifica `n_tokens` en **quintiles**
   cortando en los percentiles 20/40/60/80 calculados sobre la particion canonica (decision de diseno
   declarada, no un hiperparametro buscado), se calcula `p_q[c,t]` dentro de cada estrato `q`, y se
   promedia con pesos `w_q` = frecuencia del estrato `q` en el corpus completo:
   `p_est[c,t] = sum_q w_q * p_q[c,t]`.
   Las 5 filas `p_q[c,t]` se publican tambien, para que el lector vea el perfil entero y no un
   estrato elegido.
3. **Excluyendo SPACE**: identico a (1) pero con el denominador y el numerador sin `SPACE`, para
  gehacer comparables con la definicion de D-026 (`if not token.is_space`). Se reporta como linea de
   sensibilidad declarada, no como alternativa seleccionada.

**Soporte y cobertura** por celda `(c,t)`:
- `n_tokens[c,t]` (conteo absoluto), `n_filas[c,t]` (filas que contienen al menos una `t`),
  `cobertura = n_filas[c,t] / n_filas[c]` (proporcion de filas de la clase que contienen la etiqueta).
- Etiquetas de soporte insuficiente se marcan `soporte_insuficiente = (n_tokens[c,t] < 1000)` y se
  sombrean en las figuras. No se eliminan: se muestran y se marcan.

**Diferencia por clase** contra la referencia pooled:

```
ref[t] = n[:,t].sum() / n.sum()          (pooled sobre las 6 clases)
delta[c,t] = p[c,t] - ref[t]
SE[c,t]    = sqrt(p*(1-p)/n_tokens[c,t] + ref*(1-ref)/n_tokens[:,t].sum())   (pooled, conservador)
z[c,t]     = delta[c,t] / SE[c,t]
```

`z` se corrige con Bonferroni sobre las 18 pruebas por clase (alfa 0,05). `z_corregido` se publica.

**A2 · Descriptivo de apoyo.** Para el informe y las figuras: distribucion de `n_tokens` por clase,
entropia de Shannon del perfil POS por clase, `n_sents` por clase, y el perfil agregado (pooled). Es
contexto, no criterio.

**A3 · COTA DE POTENCIA DISCRITIVA (obligatoria, ver seccion 5).**

**A4 · Salidas.** `reports/tables/v37_pos_por_edad.csv` (formato largo, columnas `anotador,
variante, q_estrat, edad, tag, n_tokens, n_filas, cobertura, p, ref, delta, se, z, z_bonferroni,
soporte_insuficiente`; 1728 filas segun el desglose de TA.1) y
`reports/figures/v37_pos_01_perfil_por_clase.png`, `v37_pos_02_perfil_md_vs_lg.png`.

**GATE G1** (seccion 8.1).

### 4.2 FASE B · Estabilidad

**B1 · Variacion entre folds, leave-one-fold-out.** Para cada `k` en 0..4:

```
p_k[c,t] se estima SOLO con las filas fold_idx != k, y se aplica a las filas fold_idx == k
dispersion[c,t] = max_k p_k[c,t] - min_k p_k[c,t]
```

Esto es el mismo esquema que la cota de potencia: si la tabla se estimara sobre el fold entero, la
"estabilidad entre folds" seria circular. Se exige que el numero de filas por clase dentro de cada
train de fold sea >= 20.000 (verificado: ~28.000), y se publica `n_train_fold`.

**B2 · Acuerdo md vs lg.** Por celda `(c,t)`: `acuerdo[c,t] = |p_k^md[c,t] - p_k^lg[c,t]|`
maximizado sobre `k`. Se publica `max_dc_md_lg[c,t]` y la etiqueta que lo maximiza.
Contexto: md y lg son modelos spaCy **independientes** (mismo pipeline, distinto tamano). Un acuerdo
alto entre ellos indica que el rasgo es del texto; un desacuerdo alto indica artefacto del anotador.

**B3 · IC95 por grupos de duplicados.** Unidad de remuestreo = **cluster de duplicado**
(164.018 grupos), semilla **20261010**, **2000 iteraciones**, IC95 por percentiles 2,5 / 97,5.
Para que el coste entre en minutos, la tabla deJp se implementa con estadisticos suficientes:

```
S[grupo, celda] con celda = clase*19 + t   (18 tags + 1 total de tokens)  -> (164018, 115)
key[fila] = cluster[fila]*115 + celda[fila]
por iteracion:  tot = np.bincount(key[idx], minlength=n_grupos*115)   # idx = remuestreo de grupos
                p_b = reshape(tot, (n_grupos, 115))[:, :18] / tot[:, 18:19]
```

Coste: 2000 x ~169.336 incrementos de `bincount` ~ 3,4e8, del orden de 30-60 s. Aceptable.
Se reporta `ic95_delta[c,t]`, `delta_fuera_de_ic` (el pooled `delta` cae fuera del IC de la diferencia)
y el **control de cobertura de grupos**: se comprueba que cada iteracion remuestrea
`n_grupos` grupos y que el numero de filas efectivas por iteracion se mantiene dentro de
+-5 % de 169.336 (si no, se registra y se documenta; no se itera).

**B4 · Salidas.** `reports/tables/v37_pos_estabilidad.csv`, `reports/figures/v37_pos_03_estabilidad_folds.png`,
`v37_pos_04_ic_por_grupo.png`.

**GATE G2** (seccion 8.1).

### 4.3 FASE C · Contraste con lo ya medido

**C1 · Solape con D-026.** D-026 construyo `pos_{tag}` para las 18 UPOS como `counts/total` con
`total` **excluyendo `SPACE`**. V37 construye `p[tag|edad]`. **La tabla de V37 es la desagregacion
por clase de las 18 features de D-026.** El contraste se hace calculando, para las 18 celdas
`(tag, ratio)`, la correlacion de Pearson y de Spearman entre:
- vector A = las 18 proporciones UPOS por fila (definicion D-026, sin SPACE), y
- vector B = el perfil UPOS por fila reconstruido desde `p[edad|t]` de V37, con la edad verdadera.

Es decir: **se verifica si la tabla descriptiva de V37 es la misma señal que D-026 midio.** Si la
correlacion es alta (esperado, > 0,9), V37 es una **descomposicion legible** de D-026 y no un
hallazgo. Se reporta el valor y se declara.

**C2 · Solape con V14.** V14 midio 37 ratios agregados de morfologia, lemas, entidades y sintaxis
(aislado 0,2409; fusion -0,0015). V17 compara, etiqueta por etiqueta, el signo y el rango de la
diferencia por clase (`delta[c,t]` de V37) contra los 37 de V14 leidos de
`experiments/v14a_linguistico.json` y `experiments/v14b_fusion_linguistico.json`. Objetivo: separar
"etiqueta ya cubierta por un ratio agregado de V14" de "etiqueta cuyo perfil no aparece en V14".

**C3 · Solape con V31.** V31-03 midio pronombres por persona, palabras funcionales, tiempo verbal,
negacion y densidad lexical, con banco +0,006820 e integracion +0,0000642. V37 contrasta el bloque
`PRON` (y `PRON` desglosado no, porque V31 lo desgloso por forma lexical y V37 no tiene lemma de
persona disponible en UPOS) y `DET ADP AUX CCONJ SCONJ` (bloque funcion) contra el rendimiento de
V31. Se declara explicitamente que **V37 no puede desglosar pronombres por persona** porque UPOS no
carrya ese atributo: es una limitacion del artefacto, no un hallazgo.

**C4 · Contraste con P1a y P2.** Se contrastan las tres cifras medidas contra los dos suelos de la
seccion 1.3 y se responde por escrito a tres preguntas:
1. ¿La cota de potencia de V37 supera, iguala o queda por debajo de P2 (0,2750921)? Si queda por debajo,
   la informacion marginal es un **subconjunto estricto** de lo que P2 ya exploto como n-gramas 1-3.
2. ¿Queda por debajo de P1a (0,3997), que es la direccion dual `P(clase|token)` con cross-fit? Si es
   asi, confirma que el refinamiento por etiqueta no aporta sobre el refinamiento por token.
3. ¿La cota supera el suelo estricto 0,0485421 por un margen que justifique abrir una linea? Si no,
   la familia se cierra.

**C5 · Salidas.** `reports/tables/v37_pos_contraste.csv`, seccion del informe con la tabla 2.4
reexpresada contra los dos suelos.

**GATE G3** (seccion 8.1).

### 4.4 FASE D · Auditoria de novedad del canal de transiciones (NO predictiva)

Contenido integro en la seccion 6. La parte computable de esta fase es **una sola medicion
descriptiva y gratis**, mas la declaracion de la auditoria.

**D1 · Medicion de viabilidad (descriptiva, sin entrenar nada).** Para cada `k`, con filas
`fold_idx != k`, y para cada clase `c` y cada par `(a,b)` de etiquetas:

```
T_k[c,a,b] = P(POS_{i+1} = b | POS_i = a, edad = c)      (conteo de bigramas, suavizado Laplace alpha=1)
```

**Pregunta que decide la hipotesis.** El paso (d) solo puede contener informacion de clase si la
matriz de transiciones **difiere entre clases mas de lo que difiere cuando simplemente se reestima
sobre otra muestra del mismo tamano**. Eso se mide con dos distancias de variacion total, sin modelo:

Para cada fold `k`, sobre las filas `TRAIN_k = {fold_idx != k}`:

```
T_k[c,a,b]        = P(POS_{i+1}=b | POS_i=a, edad=c)   con Laplace alpha=1
T_k_pooled[a,b]   = P(POS_{i+1}=b | POS_i=a)          con Laplace alpha=1, sobre las mismas filas

div_interclase_k[c] = 0.5 * sum_{a,b} | T_k[c,a,b] - T_k_pooled[a,b] |

# RUIDO DE ESTIMACION: las filas de TRAIN_k se parten en dos mitades disjuntas por un hash
# DECLARADO del id de cluster (semilla 0, sin aleatoriedad de datos). Se estima T en cada mitad.
# 0.5*sum|T_A - T_B| es, por construccion, cuánto se mueve la matriz al reestimarla.
mitad(c, lado)     = filas de TRAIN_k cuyo hash(cluster_id, lado) cae en ese lado
div_ruido_k[c]     = 0.5 * sum_{a,b} | T_mitadA[c,a,b] - T_mitadB[c,a,b] |

div_interclase[c] = media_k( div_interclase_k[c] )
div_ruido[c]       = media_k( div_ruido_k[c] )
R                  = media_c(div_interclase[c]) / media_c(div_ruido[c])
```

**Decision pre-registrada, sin eleccion ni umbral buscado.** El contraste es
`D = media_c(div_interclase[c]) - media_c(div_ruido[c])`, con **IC95 pareado por bootstrap de grupos de
duplicados** (2000 iteraciones, semilla 20261010, unidad el cluster, misma maquinaria que B3):

- **IC95 de `D` contiene el 0** -> la variacion entre clases es indistinguible del ruido de estimacion.
  La hipotesis esta **muerta por ruido estadistico**. El canal predictivo no tiene justificacion y no
  se escala a nadie.
- **IC95 de `D` excluye el 0** -> la interaccion de orden tres **existe** en el corpus. La hipotesis
  queda **viva pero NO ejecutada**: se escala a Carlos con el numero, y sigue sin ejecutarse en V37.

`R` se publica como medida de efecto, pero **la decision la toma el IC95 de `D`**, no `R > 1`. Motivo:
`R > 1` con un denominador pequeno es un test de un solo lado, fragil y no distinguishible de ruido de
la propia estimacion del denominador; el IC95 pareado responde directamente a la pregunta y usa
maquinaria ya congelada en el pre-registro. **Ninguna de las dos ramas autoriza una corrida.**

Ademas, y esto es parte de la honestidad del diagnostico: se publica el **soporte por celda**
(`n_bigramas[c,a,b]`) y la **densidad de la matriz de transiciones** (numero de `(a,b)` con soporte
>= 100 frente al total de `18*18 = 324`). Con `min_df=5` en P2, el numero efectivo de bigramas POS
sobre `pos_seq` fue ya limitado; la cifra exacta aqui es la que permite afirmar si una matriz de
transiciones por clase es estimable en este corpus.

**D2 · Salidas.** `reports/tables/v37_pos_transiciones.csv` (con `T[c,a,b]`, `n_bigramas[c,a,b]`,
`div_interclase[c]`, `div_ruido[c]`, `R`, `D`, `ic95_D`, `excluye_cero`, y la densidad de la matriz),
`reports/figures/v37_pos_05_transiciones_divergencia.png`.

**GATE G4** (seccion 8.1). G4 no autoriza ninguna corrida predictiva.

### 4.5 FASE E · Gate de decision

Aplica el arbol de la seccion 8.3, escribe el veredicto en el informe, propone el acta, y **se detiene**.
V37 no produce ninguna corrida predictiva, ningun submission, y ningun cambio en el campeon.

---

## 5. Cota de potencia discriminativa: diseno exacto y sin fuga

### 5.1 Decision propia (explicita)

**Va dentro del diagnostico. Es obligatoria. Va en la fase A, no en la fase E.**

Razon en cuatro puntos:

1. **Es la evidencia decisiva.** `P(tag|edad)` sin una cota de rendimiento no responde a la pregunta
   accionable. Un patron puede existir y no valer nada; puede existir y valer mucho. La tabla sola no
   distingue.
2. **Es barata.** 20 pases de conteos sobre 135.000 filas x 18 celdas. Orden de segundos. El
   presupuesto de la fase es "minutos" y esta medicion no lo compromete.
3. **No toca el campeon.** No se reentrena V30, no se fusiona nada, no se buscan pesos. Es un
   clasificador **aislado**, evaluado en su propio OOF. Es el mismo tipo de medicion que ya existe en
   el proyecto (P2 "canal solo", V14a "control aislado", P1a "bloque solo"), y esos numeros son la
   vara con la que se compara.
4. **Es la unica forma de distinguir (i) de (ii)** en la auditoria de novedad: si la cota de la
   marginal esta claramente por debajo de la de P2, entonces el canal de transiciones solo puede
  , supera P2 por una interaccion de orden tres sobre una senal marginal ya agotada.

### 5.2 Modelo (cerrado, sin hiperparametros)

Clasificador **naive Bayes multinomial con normalizacion por longitud**, sobre proporciones de POS.

Para el fold `k`:

```
train_k = filas con fold_idx != k          (135.468 filas; 27.000-28.000 por clase)
val_k   = filas con fold_idx  == k          (33.868 filas)

pi_k[c]      = n_filas[c en train_k] / n_filas[train_k]                  (6 numeros)
theta_k[c,t] = (n_tokens[c,t en train_k] + 1.0)
               / (n_tokens[c en train_k] + 1.0 * 18)                     (6 x 18 numeros)

para fila i en val_k, con x[i,t] = n_tokens[i,t] / n_tokens[i]  (proporciones, suma 1):
log p(c | i) = log pi_k[c] + sum_t  x[i,t] * log theta_k[c,t]
p_i         = softmax(log p(. | i))
pred_i      = argmax_c p_i[c]
```

- **Longitud invariante**: `x` es una proporcion, no un conteo. El modelo no ve `n_tokens`. Es la
  lectura honesta de "distribucion condicional" y es la que hace que la comparacion con el control de
  longitud de la fase B tenga sentido.
- **Suavizado**: Laplace con `alpha = 1.0`, fijo, declarado a priori. **No hay grilla.**
- **Vocabulario**: las 18 UPOS declaradas. **No hay `min_df`, no hay seleccion, no hay drop.**
- **Sin ajustar**: no se tocan ni `alpha`, ni el modelo, ni el vocabulario. El modelo no se
  reentrena mas de una vez por fold. Nada se selecciona con los datos de validacion.
- **Estabilidad numerica**: `log theta` se calcula con `np.log` sobre probabilidades strictly
  positive (el Laplace lo garantiza); se recorta a `[-30, 0]` por seguridad y se registra si el recorte
  se activa. `log pi` con epsilon `1e-12`.

### 5.3 Por que NO hace falta CV anidada (justificacion explicita)

Las unicas cantidades estimadas son **6 priors + 108 probabilidades** = 114 estadisticos suficientes,
estimados de ~135.000 filas (~1.180 filas por celda de clase x etiqueta, ~430.000 tokens por clase).
**No hay ningun paso de seleccion de modelo**, luego no hay nada que pueda sobreajustarse al fold de
validacion. Elestimador es una tabla de lookup. Anadir un CV anidado seria Theater.

Ademas, y esto es lo que hace el protocolo limpio:
- La particion canonica **ya agrupa por duplicados** (`StratifiedGroupKFold` sobre `dup_clusters`), asi
  que ningun duplicado cruza la frontera ajuste/aplicacion. Se verifica en runtime que
  `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` sobre `cluster` reproduce
  `fold_idx` con coincidencia 1,000000; si no reproduce, `[STOP] exit 2` (se usa el `.npy` leido, no
  el reconstruido; la reconstruccion es solo una asercion).
- El estimador se ajusta **una vez por fold** y se aplica **solo a ese fold**. Cada fila de las
  169.336 recibe exactamente una prediccion, hecha con un modelo que no la vio.

### 5.4 Brazos (4, todos reportados, ninguno seleccionado)

| Brazo | Que es | Para que |
|---|---|---|
| **A0** | Prediccion constante de la clase mayoritaria | El suelo estricto, `0,0485421` |
| **A1** | NB sobre **proporciones** (el diseccion de 5.2), md y lg | **La cota principal** |
| **A2** | NB sobre **conteos crudos** (`x = n_tokens[i,t]`, sin dividir), md y lg | Cuantifica cuanto de A1 es longitud: el incremento es `A2 - A1` por construccion |
| **A_perm** | NB como A1 pero con las **etiquetas de clase permutadas** dentro de `train_k` (semilla 20261010) | **Detector de fuga**. Debe caer en A0 +-0,005 |

`A_perm` es la pieza que convierte la cota en una cota *honesta*: si el permutado no cae al suelo,
existe una via de fuga (orden de filas, desalineacion parquet/canonico, filtro de columnas) y el
resultado no vale nada. Si cae, la cota es creible.

### 5.5 Metricas de A

- `macro_f1` sobre las 6 clases (sklearn `f1_score(average="macro", labels=range(6))`).
- **IC95 por grupos de duplicados**, misma maquinaria que B3, semilla 20261010, 2000 iteraciones,
  unidad el cluster.
- Reportar siempre `A0`, `A1`, `A2`, `A_perm`, y las diferencias `A1 - A0`, `A2 - A1`, `A1 - P2`,
  `A1 - P1a`.
- **Re-expresar P2, P1a y el campeon contra A0 y contra 0,166667** (seccion 1.3).

### 5.6 Lo que NO se hace aqui

- No se fusiona A1 con el campeon. **Ninguna** integracion, ni neutral 0,5/0,5, ni con los pesos
  congelados, ni con busqueda de alfa.
- No se ajustan pesos del plug-in. No se genera submission. No se toca `v30_prob`.

---

## 6. Auditoria de novedad del paso (d) · `P(POS_{i+1} | POS_i, edad)`

### 6.1 Las tres opciones

**(i) misma familia ya cerrada** · **(ii) variante disfrazada** · **(iii) genuinamente distinto**.

### 6.2 Analisis

**Por que no es (i).** P2 (D-077) uso `TfidfVectorizer(analyzer="word", ngram_range=(1,3),
min_df=5, sublinear_tf=True)` sobre `pos_seq` y `LogisticRegression(C=1.0)`. Un modelo lineal sobre
indicadores de uni/bi/trigramas de POS representa **efectos aditivos**: una contribuciones por
unigrama, otra por bigrama, otra por trigrama, mas un sesgo por clase. La matriz de transiciones
condicionada por edad `T_c[a,b] = P(POS_{i+1}=b | POS_i=a, edad=c)` es un objeto **indexado por
clase**. Su contribucion al score de la clase `c` requiere el termino de **interaccion de orden tres**
`1{POS_i=a} x 1{POS_{i+1}=b} x 1{edad=c}`, y un modelo **lineal** no puede expresar una interaccion de
orden tres a partir de margenes de orden uno y dos. Hay una diferencia formal real.

**Por que no es (iii).** Cinco razones, y cada una remite a evidencia medida:

1. **La literatura en espanol recomienda precisamente lo que P2 ya midio, y lo midio peor de lo que
   uno esperaria.** Flekova y Gurevych (2013, PAN/CLEF 2013, CEUR Vol-1179) sobre el corpus PAN-2012
   (236.600 documentos en ingles, 75.900 en espanol; 3 grupos de edad) escriben: *"While in English the
   ratio of individual part-of-speech tags plays an important role, in Spanish **POS trigrams and
   quadrigrams are preferred**"*, y alcanzan edad 0,57 en espanol y 0,53 en ingles (mayoritario 0,33).
   Cruz, Haro y Ortega (2013, ITALICA, CEUR Vol-1179) construyen exactamente representaciones de
   n-gramas de POS `n = 1, 2, 3` con el tagset PAROLE simplificado, seleccion por chi-cuadrado, y
   reportan **edad 0,62 en espanol**. Ese es el patron (a)-(c) del paso (d) sin la indexacion por clase:
   es P2. P2 lo midio: canal solo `0,2750921` (md), integracion `-0,0086357`, **0/5 folds**.

2. **El refinamiento por clase ya se midio en la direccion dual y no tiene hueco.** P1a (D-041) estimo
   `P(clase | token)` con suavizado empirico-bayesiano y **cross-fit interno**, y dio **0,3997** de
   bloque solo: el mejor bloque aislado del proyecto. Su fusion con el campeon fue **`-0,0002` con
   3/5 folds, y sus tres controles permutados quedaron empatados con el bloque real
   (0,6000 / 0,6003 / 0,6000)**. Una matriz de transiciones por clase es un **refinamiento** de una
   distribucion por token por clase sobre el mismo alfabeto. P1a ya mostro que ese refinamiento, con
   cross-fit y a peso fijo, no mueve al campeon y es indistinguible de su propia permutacion.

3. **El mecanismo "canal de transiciones integrado con el campeon" ya fallo dos veces en este
   proyecto, por el mismo camino.** P2: bloque de secuencia POS, `-0,0086357`, 0/5 folds. V36-E1
   (D-088): bloque de transiciones de clase de caracter (C1-A), que dio *"señal REAL, ESPECÍFICA y POR
   DOCUMENTO"* y aun asi se cerro porque *"el campeon ya la expresa mejor"*. La leccion de D-088 es
   directa: en este corpus, un bloque de transiciones puede tener senal real y aun asi ser
   **redundante** frente a lo que el campeon ya extrae del texto.

4. **La senal marginal que domina esta familia ya esta medida y es debil.** El paso (d) explota la
   desviacion de `T_c` entre clases. Pero la mayor parte de la informacion POS por edad esta en las
   **proporciones marginales**, y esas ya tienen cuatro mediciones independientes: D-026 integracion
   `-0,0032`; V14 37 ratios, integracion `-0,0015` con 1/5 folds; V31 canales sin tagger, integracion
   `+0,0000642`; P2, integracion `-0,0086357`. Ninguna abrio nada. El paso (d) no puede extraer de la
   interaccion mas de lo que la marginal no dio, salvo que la interaccion sea desproporcionadamente
   grande, y esa es exactamente la pregunta que responde la medicion D1 de la fase D.

5. **La literatura reciente sugiere que para **edad** el efecto POS marginal es debil, y que el efecto
   fuerte es de genero, no de edad.** Johannsen, Hovy y Sogaard (2015, CoNLL 2015, ACL Anthology
   K15-1011) hacen el estudio multilingue mas riguroso que conozco de sintaxis por edad y genero
   (Trustpilot, dependencias universales, 11 idiomas, seleccion de estabilidad sobre regresion
   logistica L1, chi-cuadrado con correccion de Bonferroni). Su conclusion textual: *"For age,
   features vary a lot more than for gender, i.e., there is **less support** for each than there was for
   the gender features"*, y *"We did not find any evidence for **pervasive** age effects across
   languages"*. En contraste, para generoEncuentra tres treelets atomicos (etiquetas POS) significativos
   en los 11 idiomas. Es decir: la literatura apoya que **la edad tiene menos sintaxis discriminantiva
   que el genero**, y que la que tiene es dispersa y multilingue, no universal.

### 6.3 Veredicto

> **(ii) VARIANTE DISFRAZADA.**
>
> El paso (d) no es genuinamente nuevo. Su objeto formal (`T_c` indexado por clase) no esta en la
> clase de funciones que P2 midio, asi que no es (i) en sentido estricto. Pero el margen que aportaria
> esta acotado por dos mediciones previas que ya lo acotan: (a) el refinamiento por clase de la
> distribucion por token, cross-fit, a peso fijo, ya se midio en la direccion dual y dio `-0,0002` con
> controles permutados empatados (D-041); y (b) la integracion de un bloque de transiciones con el
> campeon ya fallo dos veces por el mismo mecanismo (D-077, D-088). La unica pieza genuinamente no
> medida es la **magnitud de la interaccion de orden tres**, y esa se mide **gratis y sin entrenar
> nada** en la fase D (D1).

### 6.4 Decision operativa

1. **El canal predictivo del paso (d) NO SE EJECUTA.** Decision por defecto: **conservar el campeon
   V30 + plug-in, 0,6159732**.
2. **Se ejecuta solo la medicion descriptiva D1** (divergencia entre clases vs ruido de estimacion,
   con IC95 pareado por grupos de duplicados, y soporte de la matriz de transiciones). Es gratis, no entrena nada, no es un canal predictivo, y es la
   cantidad que decide la hipotesis. Su resultado **no habilita ninguna corrida**: habilita, como
   maximo, escalar a Carlos.
3. **Para habilitar el canal predictivo se requiere autorizacion expresa de Carlos con una hipotesis
   nueva por escrito**, no una variante encadenada. Esto aplica la regla que ya documento D-077: *"una
   hipotesis negativa se cierra; no se encadenan variantes de POS sin una hipotesis nueva"*.
4. Si Carlos autoriza, la hipotesis nueva debe declarar por escrito cual es la diferencia mecanica
   respecto de P2 (la interaccion `1{POS_i} x 1{POS_{i+1}} x 1{edad}`) y cual es su cota teorica
   (idealmente, el resultado de D1), **antes** de escribir una sola linea.

**No se vende novedad que no existe.**

---

## 7. FASE C en detalle · el contraste que decide

### 7.1 Las cinco familias ya medidas y su tipo de variable

| Familia | Tipo de variable | Que la hace distinta de `P(tag\|edad)` |
|---|---|---|
| D-026 | 26 ratios por fila (18 UPOS + 8 derivados), sin SPACE | Es la **version no desagregada por clase** de la misma tabla |
| V14 | 37 ratios **agregados** de morfologia, lemas, entidades, sintaxis | Agrega tags y mezcla POS con morfologia y dependencias |
| V31 | Regex sin tagger: pronombres por persona, funcion, tiempo verbal, negacion, densidad | Usa **forma lexical**, no categoria gramatical |
| P1a | `P(clase\|token)`, cross-fit | Es la **direccion dual** de la misma tabla |
| P2 | n-gramas 1-3 sobre `pos_seq` | Es la **version con estructura de secuencia**, no condicionada por edad |

### 7.2 La pregunta que C responde

> ¿La tabla `P(tag|edad)` es una **descomposicion legible** de señales ya medidas, o es un patron
> nuevo? La respuesta honesta mas probable es la primera, y el plan esta escrito para que eso sea un
> **resultado aceptable y no un fracaso**.

### 7.3 Descomposicion de D-026 (26 = 18 + 8)

Las 8 derivadas que ya se midieron y que V37 no vuelve a medir:
`pos_unique`, `pos_entropy`, `pos_bigram_diversity`, `pos_trigram_diversity`, `pos_content_ratio`,
`pos_function_ratio`, `pos_verb_noun_ratio`, `pos_adj_noun_ratio`. Las 18 son `pos_{tag}`.

**Implicacion directa y honesta**: V37 no puede aportar ninguna señal que D-026 nomidiera como
*variable*. Lo unico que V37 puede aportar es **descomposicion por clase** (que D-026 no reporto),
**estabilidad** (que D-026 no midio) y **una cota de potencia de la marginal** (que D-026 no midio;
D-026 solo reporto el delta de integracion `-0,0032`). Ese es el valor real de V37.

### 7.4 Discrepancia de denominador que C debe declarar

D-026: `tags = [token.pos_ for token in document if not token.is_space]`.
V37 (parquet congelado): `pos_seq` **incluye `SPACE`** (32.865 ocurrencias en 25.232 filas, 14,9 %),
y `n_tokens` las cuenta (`len(pos_seq.split()) == n_tokens` en el 100 % de las filas).

Consecuencias que C debe medir y declarar, no ocultar:
1. Las proporciones de D-026 y las de V37 **no son la misma variable**. La de V37 esta desinflada en
   ~1,3 % por token.
2. `SPACE` es en realidad un indicador de **saltos deEspacio** del tokenizador, no una categoria
   gramatical. Es el unico "tag" con semantica de formato. Se reporta con su propia fila y con la
   linea de sensibilidad sin SPACE (A1.3), declarada a priori.
3. Los conteos de V37 incluyen ~1,3 % de tokens no gramaticales. Afecta al denominador de todos los
   IC. Se documenta con la cuenta exacta.

### 7.5 Salida de la fase C

`reports/tables/v37_pos_contraste.csv` + seccion del informe que responde las tres preguntas de C4.
Sin esta fase, V37 no puede cerrarse: cerrar sin contrastar seria repetir en otro formato.

---

## 8. Boundaries

### 8.1 Gates (medibles, evaluados por el script)

**G1 (fase A)** · pasa si el perfil por clase existe con soporte utilizable:
- `n_tokens[c,t] >= 1000` en al menos **12 de las 18 etiquetas** para al menos **5 de las 6 clases**
  (criterio de reportabilidad, no de senal).
- Las tres variantes (crudo, estandarizado por longitud, sin SPACE) estan calculadas y publicadas.
- La cota A1 esta calculada para md y lg, con IC95 por grupos, y `A_perm` cae en `A0 +- 0,005`.

**G2 (fase B)** · pasa si la tabla es estable:
- `dispersion[c,t] <= 0,005` para todas las etiquetas con `n_tokens[c,t] >= 5000` y
  `z_bonferroni` significativo.
- `max_dc_md_lg[c,t] <= 0,005` para esas mismas etiquetas.
- El IC95 por grupos esta completo para las 18 x 6 celdas.

**G3 (fase C)** · pasa si el contraste esta escrito y responde las 3 preguntas de C4 con numeros. No
tiene umbral de paso: es una obligation de forma. Su ausencia es un **STOP**.

**G4 (fase D)** · pasa si la auditoria de novedad esta escrita con las 5 razones de 6.2 y si la
medicion D1 esta completa. **G4 no autoriza ninguna corrida predictiva.**

### 8.2 Umbrales del veredicto (fase E, literales y pre-registrados)

Se evaluan **todos**. Se necesitan **los cinco** para `PATRON_ESTABLE_NO_REDUNDANTE`:

- **(V1) Diferencia real.** Existe `delta[c,t] >= 0,010` (diferencia absoluta de probabilidad de
  aparicion por token entre la clase `c` y el pooled) con `z_bonferroni` significativo (p < 0,05/18),
  para **al menos 3 etiquetas** y **al menos 4 de las 6 clases**.
  Justificacion de los dos floors: (a) `z` con Bonferroni protege del ruido muestral; con
  ~431.813 tokens por clase, para `p = 0,05` el error estandar es `sqrt(0,05*0,95/431813) = 3,3e-4`,
  luego `0,010` es ~30 veces el error estandar: el umbral estadistico es ~`3*SE = 1,0e-3`, muy por
  debajo, y `0,010` es un **suelo practico** (0,010 de probabilidad de aparicion son ~0,15 tokens en
  un tweet de 15 tokens: uno token extra cada 7 tweets). Se exigen los dos floors.
- **(V2) Estabilidad entre folds.** Esas mismas etiquetas cumplen `dispersion[c,t] <= 0,005`.
- **(V3) Acuerdo entre anotadores.** Esas mismas etiquetas cumplen `max_dc_md_lg[c,t] <= 0,005`.
- **(V4) No es longitud.** La version estandarizada por longitud conserva
  `|p_est[c,t] - ref[t]| >= 0,008` para esas mismas etiquetas (es decir, el efecto no desaparece al
  estandarizar por quintiles de `n_tokens`).
- **(V5) Potencia por encima de lo ya exploiterado.** `A1(md) >= 0,20` **y** `A1(lg) >= 0,20`
  **y** `A1(md) >= 0,2750921 - 0,02` **y** `A1(lg) >= 0,2725490 - 0,02`.
  Interpretacion: (a) `>= 0,20` esta ~4 veces el suelo estricto; (b) exigir `>= P2 - 0,02` impide que
  la fase A "descubra" una senal que el n-gramado de P2 ya habia agotado. Si `A1` cae por debajo de
  P2, la marginal es un subconjunto estricto de lo que P2 exploto, y la conclusion correcta es
  **redundante**, no nueva.
- **(V6) Control de fuga limpio.** `|A_perm - A0| <= 0,005` en md y lg.

`DIAGNOSTICO_CERRADO` si cualquiera de V1..V6 falla. **No hay veredicto intermedio.**

### 8.3 Arbol de decision de la fase E

```
G1 falla  -> STOP. Artefactos rotos o corpus no reportable. No se continua. Acta de incidente.
G2 falla  -> DIAGNOSTICO_CERRADO por inestabilidad. Campeon conservado.
G3 falla  -> STOP. No se puede cerrar sin contraste.
V1..V6 falla (G2 ok) -> DIAGNOSTICO_CERRADO. Campeon conservado. Se documenta que la senal
                          existe pero no es ACTIONABLE (redundante, pequena o inestable).
V1..V6 pasa -> PATRON_ESTABLE_NO_REDUNDANTE.
                  -> Y AUN ASI: NO se ejecuta ninguna corrida predictiva en V37.
                  -> Se escala a Carlos con el informe. La siguiente linea, si la quiere,
                     requiere hipotesis nueva escrita (misma regla que D-077).
G4 sin autorizacion de Carlos -> el canal de transiciones queda BLOQUEADO. Decision por defecto:
                     conservar el campeon.
```

### 8.4 Always

- Un experimento = un script nuevo. No se edita ni se borra ningun script ya ejecutado.
- Todo script en `scripts/` es **ASCII puro**: sin acentos, sin emojis, sin CJK. Los documentos si
  admiten acentos.
- Todo runner lleva `--verificar` con **autoejercicio estructural permanente** que ejercite **su propia
  ruta de salida** (leccion de D-086: un runner sin autoejercicio muere *despues* de entrenar). Ver
  seccion 11.
- Se leen los parquets y los `.npy` canonicos. **No se ejecuta spaCy.** **No se reanota nada.**
- Los folds se **leen** de `fold_idx.npy`. No se recrean. La particion es la canonica de V20 y V30.
- Toda cifra POS se cita con su suelo (0,0485421 y/o 0,166667).
- Docstrings completos, sin emojis en codigo, estructura de secciones por bloques.
- CPU pura (`n_jobs=-1` donde aplique, 32 hilos). Sin GPU en esta fase (D-007).

### 8.5 Ask first

- Cualquier cambio en los pesos del plug-in, en `v30_prob`, en la representacion o en la semilla.
- Cualquier intento de integrar **cualquier** resultado de V37 con el campeon.
- Autorizar el canal predictivo del paso (d) (requiere hipotesis nueva escrita).
- Ampliar el alcance a `lemma_POS`, `morph_seq` o `dep_seq` (fuera de esta spec: ya cerrados en
  D-078/D-079/D-080 y fuera del encargo de Carlos, que pidio `P(POS|edad)`).
- Cualquier rebalanceo de clases (esta spec lo declara NO APLICA, ratio 1,0627).
- Cambiar el pre-registro una vez congelado. Se versiona como `_v37b`, nunca se edita en sitio.

### 8.6 Never

- **Test, holdout, leaderboard, submission.** Nunca. Ni para diagnostico.
- **Transformers, LLM, embeddings**, preentrenados o entrenados. (spaCy `es_core_news_md/_lg` solo
  para etiquetas, D-076; en V37 ni se carga.)
- Datos o etiquetas externas. Ningun corpus, ningun lexicon de fuera.
- Metadatos. Ninguna feature de metadata, follower, perfil o timestamp.
- **Agregar por autor.** Un tweet por fila, siempre. El bootstrap agrupa por duplicado, no por autor.
- **Grillas de hiperparametros.** Cero. Ni `alpha`, ni `C`, ni `n_jobs` de busqueda, ni subconjuntos
  de etiquetas, ni subconjuntos de folds.
- Tocar `test`. Tocar `submissions/`.
- Reentrenar o reevaluar el campeon. V30 se lee, no se corre.
- Editar o borrar `scripts/age_linguistic_03_pos_secuencia.py` ni ningun otro ya ejecutado.
- Fabricar metricas. Si algo no se midio, se dice que no se midio.
- Declarar que el paso (d) es nuevo cuando la auditoria dice lo contrario.

---

## 9. Descomposicion en tareas

Orden por dependencia. Cada tarea cabe en una sesion enfocada. Todas las tareas de codigo las
ejecuta `@ml-coder` **solo tras** la aprobacion de esta spec por Carlos.

### FASE 0 · Pre-registro (bloqueante, antes de calcular nada)

- [ ] **T0.1 · Congelar el pre-registro de V37**
  - Acceptance: existe `experiments/age_linguistic_13_preregistro_v37.json` con los SHA256 de los 2
    parquets (iguales a los de P2), de los 6 `.npy` canonicos (iguales a los del `manifest.json`) y de
    `dup_clusters.parquet`; con las 18 UPOS declaradas; `alpha = 1.0`; las 4 semillas; los 6
    umbrales V1-V6 literales; los dos suelos; el baseline `0,6159732`; los 6 pesos del plug-in; la
    tabla de referencias previas de la seccion 2.4; y la lista `no_hacer` de 8.6.
  - Verify: `python -c "import json;d=json.load(open('experiments/age_linguistic_13_preregistro_v37.json'));print(d['hashes']['anotacion_md_parquet'][:8])"`
    imprime `dacbc305`; igual para lg con `81287d54`, `y_true` con `b1c928f9`, `fold_idx` con
    `74d18330`, `v30_prob` con `3d773524`. JSON valido y ASCII en las claves.
  - Files: `scripts/age_linguistic_13_preregistro_v37.py`, `experiments/age_linguistic_13_preregistro_v37.json`
  - Checkpoint: **G0** · el hash del pre-registro se imprime y se registra en el informe. Si un hash no
    coincide con el anclaje, `[STOP] exit 2`: los artefactos_POS cambiaron y la spec queda obsoleta.

- [ ] **T0.2 · Autoaserciones de integridad del entorno de datos**
  - Acceptance: un script de verificacion comprueba y afirma, en runtime: (a)
    `len(pos_seq.split()) == n_tokens` en el 100 % de las filas en md y lg; (b) `fila == arange`;
    (c) `n_tags == 18` en ambos; (d) `y_true` coincide con el entero de `age_range` de train; (e)
    `ids` coincide con `train['id']`; (f) `fold_idx` tiene 5 valores con conteos
    `[33868 33868 33867 33866 33867]`; (g) `dup_clusters['cluster'][:169336]` reproduce `fold_idx`
    con coincidencia 1,000000; (h) `argmax(v30_prob * pesos) == 0.6159732` con tolerancia 1e-9;
    (i) las filas de train de cada fold tienen >= 20.000 filas por clase.
  - Verify: sale con codigo 0 e imprime `OK 9/9`. Cualquier fallo: `[STOP] exit 2` con el mensaje.
  - Files: `scripts/age_linguistic_13_preregistro_v37.py` (funcion `verificar_integridad`)
  - Checkpoint: **G0b**

### FASE A · Diagnostico descriptivo + cota

- [ ] **TA.1 · Lectura de artefactos y perfil crudo por clase**
  - Acceptance: `reports/tables/v37_pos_por_edad.csv` con columnas
    `anotador, variante, q_estrat, edad, tag, n_tokens, n_filas, cobertura, p, ref, delta, se, z,
    z_bonferroni, soporte_insuficiente`, en formato largo y cubriendo:
    (a) `variante = "crudo"` con `q_estrat = "pooled"`, 18 tags x 6 clases;
    (b) `variante = "estandarizado"` con `q_estrat = "pooled"`, 18 x 6;
    (c) `variante = "sin_space"` con `q_estrat = "pooled"`, 18 x 6;
    (d) `variante = "crudo"` con `q_estrat = "q1".."q5"`, 5 x 18 x 6.
    Todo x 2 anotadores. Las 104 filas con `n_tokens == 0` quedan **excluidas de los denominadores** y
    su cuenta se declara en el JSON. `ref`, `delta`, `se`, `z` y `z_bonferroni` solo se rellenan en las
    filas `q_estrat = "pooled"`; en las de quintil van vacios (no se selecciona estrato).
  - Verify: el numero de filas del CSV es
    `2 anotadores x [3 variantes pooled x 18 tags x 6 clases + 5 quintiles x 18 tags x 6 clases]`
    `= 2 x (324 + 540) = 1728`; la suma de `p` por `(anotador, variante, q_estrat, edad)` es 1,0 con
    tolerancia 1e-9 en **todas** las combinaciones salvo `sin_space`, donde la suma es
    `1 - SPACE(token)` y por tanto < 1, con la cuenta de SPACE declarada.
  - Files: `scripts/age_linguistic_14_diagnostico_pos.py`, `reports/tables/v37_pos_por_edad.csv`
  - Checkpoint: **G1a**

- [ ] **TA.2 · Figuras del perfil**
  - Acceptance: `reports/figures/v37_pos_01_perfil_por_clase.png` (6 curvas `p[tag|edad]` por
    anotador,con soporte de etiquetas ordenadas por delta) y `v37_pos_02_perfil_md_vs_lg.png` (scatter
    `p^md` vs `p^lg` por celda con linea identidad). Las celdas con
    `soporte_insuficiente` sombreadas. Leyenda en espanol, sin acentos en los archivos de codigo
    (las figuras si admiten tildes en sus etiquetas).
  - Verify: ambos PNG existen, pesan > 20 KB y se abren.
  - Files: `reports/figures/v37_pos_01_perfil_por_clase.png`,
    `reports/figures/v37_pos_02_perfil_md_vs_lg.png`
  - Checkpoint: **G1b**

- [ ] **TA.3 · Cota de potencia discriminativa (A0, A1, A2, A_perm)**
  - Acceptance: `experiments/age_linguistic_14_diagnostico_pos.json` contiene, para md y lg,
    `A0`, `A1` (NB sobre proporciones), `A2` (NB sobre conteos), `A_perm` (NB con etiquetas
    permutadas), con `macro_f1`, `ic95_por_grupo`, `n_train_por_clase`, `n_triminos_activados` y
    `log_pi` y `log_theta` con sus hash. Se cumple `|A_perm - A0| <= 0,005`.
  - Verify:
    - `A0 == 0.0485421` con tolerancia 1e-6.
    - `argmax` de A1 por fold k usa **solo** filas `fold_idx != k` para estimar `theta` y `pi`
      (asercion explicita en el codigo, no solo en el comentario).
    - el numero total de filas predichas es exactamente 169.336, cada fila predicha por exactamente
      un fold (asercion `np.bincount(fold_idx) == bincount(predichas_por_fold)`).
    - `A_perm` cae en `A0 +- 0,005`.
  - Files: `scripts/age_linguistic_14_diagnostico_pos.py`,
    `experiments/age_linguistic_14_diagnostico_pos.json`
  - Checkpoint: **G1c** · **es el checkpoint mas importante de la fase**. Si `A_perm` no cae al suelo,
    STOP por fuga suspected, sin continuar.

- [ ] **TA.4 · IC95 por grupos de duplicados de A1**
  - Acceptance: la misma maquinaria de bootstrap por cluster (2000 iteraciones, semilla 20261010)
    aplicada al macro-F1 de A1 y A2, en md y lg. `n_iter = 2000`, unidad = cluster.
  - Verify: tiempo de la fase A con bootstrap **< 5 minutos** medido e imprimeo en el JSON; la suma de
    `n_grupos` por iteracion esta dentro de +-5 % de 169.336 en el 100 % de las iteraciones.
  - Files: `scripts/age_linguistic_14_diagnostico_pos.py`
  - Checkpoint: **G1d**

- [ ] **TA.5 · Gate G1**
  - Acceptance: el JSON incluye `gate_G1: {rreportable: bool, n_celdas_soporte_ok: int,
    a_perm_ok: bool, passed: bool}` evaluado con los criterios de 8.1 G1.
  - Verify: `passed` coincide con la evaluacion manual de los tres criterios. Si `passed == false`, la
    cadena se detiene aqui y se documenta.
  - Files: `scripts/age_linguistic_14_diagnostico_pos.py`
  - Checkpoint: **G1**

### FASE B · Estabilidad

- [ ] **TB.1 · Dispersion entre folds (leave-one-fold-out)**
  - Acceptance: `reports/tables/v37_pos_estabilidad.csv` con `anotador, edad, tag,
    n_tokens_media, p_min, p_max, dispersion, max_dc_md_lg, argmax_fold`.
    `dispersion = max_k p_k - min_k p_k`.
  - Verify: para un tag con soporte alto (PROPN, NOUN), `dispersion` esta en el orden de 1e-3 a 1e-2;
    si `dispersion > 0,05` para todos los tags, suspecto de bug (no de senal) y se investiga antes de
    seguir.
  - Files: `scripts/age_linguistic_14_diagnostico_pos.py`, `reports/tables/v37_pos_estabilidad.csv`
  - Checkpoint: **G2a**

- [ ] **TB.2 · Acuerdo md vs lg e IC95 por grupo**
  - Acceptance: columnas `max_dc_md_lg` y su etiqueta; mas `ic95_delta[c,t]`, `delta_fuera_de_ic`
    en `reports/tables/v37_pos_ic_por_grupo.csv`, calculados con bootstrap por cluster.
  - Verify: el numero de celdas con `delta_fuera_de_ic == True` y su `delta` minimo absoluto se
    imprimen. Si `delta_fuera_de_ic` es mas del 30 % de las celdas con soporte, suspecto de bug.
  - Files: `scripts/age_linguistic_14_diagnostico_pos.py`,
    `reports/tables/v37_pos_ic_por_grupo.csv`
  - Checkpoint: **G2b**

- [ ] **TB.3 · Control de longitud quintilado y figuras de estabilidad**
  - Acceptance: `v37_pos_03_estabilidad_folds.png` (boxplot de `p_k` por (tag, clase) para las
    etiquetas de soporte alto) y `v37_pos_04_ic_por_grupo.png` (delta con IC95 por clase, ordenadas
    por delta). Las 5 filas de quintil visibles en el informe.
  - Verify: figuras existen, pesan > 20 KB.
  - Files: `reports/figures/v37_pos_03_estabilidad_folds.png`,
    `reports/figures/v37_pos_04_ic_por_grupo.png`
  - Checkpoint: **G2c**

- [ ] **TB.4 · Gate G2**
  - Acceptance: `gate_G2` en el JSON con `n_celdas_estables`, `n_celdas_acuerdo_ok`,
    `passed`. Se evaluan solo las celdas con `n_tokens >= 5000` y `z_bonferroni` significativo.
  - Verify: el numero de celdas evaluadas se imprime (tiene que ser > 0; si es 0, el gate se
    declara NO EVALUABLE y se escala a Carlos).
  - Checkpoint: **G2**

### FASE C · Contraste

- [ ] **TC.1 · Reconstruccion de D-026 y correlacion**
  - Acceptance: se calculan las 18 proporciones UPOS por fila con la definicion **literal de D-026**
    (`if not token.is_space`, denominador = tokens sin SPACE) y el perfil UPOS por fila desde
    `p[edad|t]` de V37 con la edad verdadera. Se reportan Pearson y Spearman de los dos vectores de
    18 componentes sobre las 169.336 filas (muestreo de 20.000 filas, semilla declarada, por coste).
    Ademas la cuenta deSPACE y su porcentaje de filas.
  - Verify: la correlacion reportada es **> 0,9** (esperado). Si sale < 0,7 hay un bug de
    alineacion: STOP.
  - Files: `scripts/age_linguistic_14_diagnostico_pos.py`
  - Checkpoint: **G3a**

- [ ] **TC.2 · Contraste con V14, V31, P1a y P2**
  - Acceptance: `reports/tables/v37_pos_contraste.csv` con una fila por referencia previa
    (D-026, V14, V31, P1a, P2-md, P2-lg) y columnas `macro_f1`, `suelo_estricto`, `azar_uniforme`,
    `delta_vs_suelo_estricto`, `delta_vs_azar`, `que_midio`, `se_reexpresa_como`. Mas las respuestas
    escritas a las 3 preguntas de C4.
  - Verify: las 6 filas estan; `suelo_estricto == 0.0485421` en todas; los valores de P2 y P1a
    coinciden con los de la seccion 2.4 con tolerancia 1e-7.
  - Files: `reports/tables/v37_pos_contraste.csv`
  - Checkpoint: **G3b**

- [ ] **TC.3 · Declaracion de la discrepancia SPACE**
  - Acceptance: el informe declara explicitamente la diferencia de denominador entre D-026 y el
    parquet congelado, con el numero exacto (32.865 ocurrencias, 25.232 filas, 14,9 %) y las tres
    consecuencias de 7.4.
  - Verify: la seccion esta en el informe con esas cifras literales.
  - Checkpoint: **G3c**

- [ ] **TC.4 · Gate G3**
  - Acceptance: `gate_G3.passed == true` solo si TC.1, TC.2 y TC.3 estan completos.
  - Verify: sin las tres, STOP. No se puede cerrar sin contraste.
  - Checkpoint: **G3**

### FASE D · Auditoria de novedad (descriptiva)

- [ ] **TD.1 · Matriz de transiciones por clase y divergencias**
  - Acceptance: `reports/tables/v37_pos_transiciones.csv` con, para cada clase `c` y cada par
    `(a,b)`: `n_bigramas[c,a,b]`, `T_k[c,a,b]` (leave-one-fold-out, Laplace alpha=1),
    `T_k_pooled[a,b]`, `div_interclase[c]`, `div_ruido[c]`, `R`, `D = media_c(div_interclase) -
    media_c(div_ruido)`, `ic95_D`, `excluye_cero`, y la densidad de la matriz (pares con soporte
    >= 100 sobre 324). Las dos mitades se construyen por hash declarado del id de cluster.
  - Verify: `D` y su IC95 se imprimen; `excluye_cero` se declara; `n_bigramas` suma coincide con
    `sum(len(pos_seq.split()) - 1)` para los no vacios; la densidad y el numero de mitades (2) se
    publican; el bootstrap usa 2000 iteraciones y semilla 20261010.
  - Files: `scripts/age_linguistic_14_diagnostico_pos.py`,
    `reports/tables/v37_pos_transiciones.csv`
  - Checkpoint: **G4a**

- [ ] **TD.2 · Figura de divergencia de transiciones**
  - Acceptance: `reports/figures/v37_pos_05_transiciones_divergencia.png` con, por clase, las dos
    divergencias (`interclase` y `ruido`) en barras, la linea de igualdad y el marcador de `R`.
  - Verify: figura existe, pesa > 20 KB.
  - Checkpoint: **G4b**

- [ ] **TD.3 · Seccion de auditoria de novedad en el informe**
  - Acceptance: el informe contiene la seccion 6 de esta spec **verbatim** (las 5 razones de 6.2, el
    veredicto (ii) y las 4 decisiones de 6.4), mas el resultado de TD.1 y su lectura contra
    `R <= 1` / `R > 1`.
  - Verify: el informe contiene las cadenas `(i) misma familia`, `(ii) VARIANTE DISFRAZADA`,
    `(iii) genuinamente distinto`, y `NO SE EJECUTA`.
  - Files: `reports/v37_diagnostico_pos.md`
  - Checkpoint: **G4**

### FASE E · Gate de decision

- [ ] **TE.1 · Evaluacion de V1..V6 y veredicto**
  - Acceptance: `experiments/age_linguistic_14_diagnostico_pos.json` incluye
    `veredicto: PATRON_ESTABLE_NO_REDUNDANTE | DIAGNOSTICO_CERRADO`,
    `criterios: {V1..V6: {cumple: bool, valor: ..., umbral: ...}}`,
    `arbol: "rama_de_la_seccion_8.3_ejecutada"`.
  - Verify: los 6 criterios estan evaluados con valores y umbrales literales. `veredicto` es
    coherente con `criterios` por la regla de 8.2 (los 6 true -> el primero; cualquiera false -> el
    segundo).
  - Checkpoint: **E1**

- [ ] **TE.2 · Informe final**
  - Acceptance: `reports/v37_diagnostico_pos.md` con: resumen ejecutivo; tabla `P(tag|edad)` (las 6
    clases x 18 tags); estabilidad; cota de potencia con los dos suelos; contraste con la seccion 2.4
    reexpresada; auditoria de novedad; veredicto; **y una declaracion explicita de que ninguna
    corrida predictiva se ejecuto y que el campeon no se toco**. Sin celdas abiertas.
  - Verify: el informe nombra los 5 artefactos de la seccion 10; contiene `0,6159732` como baseline no
    tocado; contiene las dos referencias al azar (0,0485421 y 0,166667).
  - Files: `reports/v37_diagnostico_pos.md`
  - Checkpoint: **E2**

- [ ] **TE.3 · Acta propuesta**
  - Acceptance: redaccion de la entrada de `DECISIONS.md` (siguiente numero libre tras D-088) con
    Observation / Experiment / Result / Decision / Reason / Consequences / Evidence, respetando el
    formato de las actas existentes. **Se propone, no se aplica sin Carlos.**
  - Verify: `@ml-reviewer` aprueba el texto; `git diff DECISIONS.md` vacio hasta que Carlos lo ordene.
  - Checkpoint: **E3**

- [ ] **TE.4 · Handoff**
  - Acceptance: `plan/contexto.md` actualizado con el veredicto de V37 y el estado del ciclo; si
    procede, `plan/V32_REAPERTURA_PLAN.md` sin cajas abiertas nuevas.
  - Verify: handoff legible en < 20 lineas.
  - Checkpoint: **E4**

---

## 10. Artefactos (nombres fijos)

### 10.1 Codigo (`scripts/`, ASCII puro)

| Ruta | Que |
|---|---|
| `scripts/age_linguistic_13_preregistro_v37.py` | Escribe el pre-registro congelado. `--verificar` = aserciones de integridad (T0.2) |
| `scripts/age_linguistic_14_diagnostico_pos.py` | Runner de las fases A-E. `--verificar` = autoejercicio de 5 caminos (seccion 11). `--humo` = corrida minima |

### 10.2 Resultados (`experiments/`)

| Ruta | Que |
|---|---|
| `experiments/age_linguistic_13_preregistro_v37.json` | Pre-registro congelado. **Inmutable** una vez escrito |
| `experiments/age_linguistic_14_diagnostico_pos.json` | Resultado completo: tabla, estabilidad, cota, contraste, transiciones, gates G1-G4, V1-V6, veredicto, `segundos`, hash de todo lo escrito |

### 10.3 Tablas (`reports/tables/`)

| Ruta | Que |
|---|---|
| `reports/tables/v37_pos_por_edad.csv` | La tabla `P(tag|edad)` completa, **1728** filas en formato largo |
| `reports/tables/v37_pos_estabilidad.csv` | Dispersion entre folds y acuerdo md/lg |
| `reports/tables/v37_pos_ic_por_grupo.csv` | IC95 por grupos de duplicados |
| `reports/tables/v37_pos_contraste.csv` | Contraste con D-026, V14, V31, P1a, P2 |
| `reports/tables/v37_pos_transiciones.csv` | `T[c,a,b]`, divergencias, `R`, densidad |

### 10.4 Informe (`reports/`)

| Ruta | Que |
|---|---|
| `reports/v37_diagnostico_pos.md` | Informe final con veredicto |

### 10.5 Figuras

Convention del proyecto: `src/config.py` define `FIGURES = REPORTS / "figures"`. **No existe un
directorio `figuras/` en la raiz** (verificado: 106 PNG en `reports/figures/`). Por coherencia con el
proyecto, las figuras van a `reports/figures/`, no a un `figuras/` nuevo.

| Ruta | Que |
|---|---|
| `reports/figures/v37_pos_01_perfil_por_clase.png` | `p[tag|edad]`, 6 curvas por anotador |
| `reports/figures/v37_pos_02_perfil_md_vs_lg.png` | Scatter md vs lg con identidad |
| `reports/figures/v37_pos_03_estabilidad_folds.png` | Boxplot de `p_k` por (tag, clase) |
| `reports/figures/v37_pos_04_ic_por_grupo.png` | Delta con IC95 por clase |
| `reports/figures/v37_pos_05_transiciones_divergencia.png` | Divergencia entre clases vs ruido de estimacion, por clase |

### 10.6 Plan (este documento)

| Ruta | Que |
|---|---|
| `plan/V37_DIAGNOSTICO_POS_PLAN.md` | Esta spec |
| `plan/00_intent_v37.md` | Intencion (ya existe, no se modifica) |
| `plan/contexto.md` | Handoff (se actualiza en TE.4) |

---

## 11. Autoejercicio estructural de `--verificar` (leccion de D-086)

D-086: un runner sin autoejercicio en `--verificar` y `--humo` **muere despues de entrenar**, cuando ya
ha gastado el presupuesto. V35 llevo esa leccion a la practica con **cinco caminos**. V37 replica el
patron, con funciones reales de punta a punta:

1. **Camino 1 — perfil**: perfil por clase sobre un subconjunto sintetico de 200 filas; verifica
   suma 1,0 y que `ref` coincide con el promedio ponderado.
2. **Camino 2 — cota**: `naive_bayes_pos()` sobre las mismas 200 filas con las 18 etiquetas; verifica
   que devuelve (n, 6), filas normalizadas, `pred` en rango, y que `A_perm` cae cerca de `A0` en el
   sintetico.
3. **Camino 3 — bootstrap por grupo**: 20 iteraciones sobre 30 grupos sinteticos; verifica la forma
   del IC y que `n_grupos` por iteracion coincide con el numero de grupos.
4. **Camino 4 — escritura**: escribe los 5 CSV, el JSON y las 5 figuras **en un directorio temporal**,
   los vuelve a leer, comprueba que los CSV parsean y tienen las columnas esperadas, y que las
   figuras pesan > 0. **Esta es la parte que D-086 dice que faltaba**: ejercitar la propia ruta de
   salida, no solo la computacion.
5. **Camino 5 — integridad**: corre `verificar_integridad()` completo (las 9 aserciones de T0.2).

`--verificar` corre los cinco caminos en un directorio temporal y sale con 0. No escribe nada en
`experiments/`, `reports/` ni `figures/`. Si un camino falla, sale con 2 y nombra el camino.

`--humo` corre la cadena completa sobre 5.000 filas (subconjunto estratificado, semilla declarada) y
escribe en un directorio temporal. No escribe artefactos reales. Objetivo: < 60 s.

---

## 12. Estado del arte (contexto, no justificacion)

Revision breve para **contextualizar** el resultado. **No justifica ninguna tecnica de esta spec**:
la unica justificacion de las fases A-C es la medicion.

| # | Referencia | Dataset | Metrica reportada | Lectura para V37 |
|---|---|---|---|---|
| 1 | **Flekova, L.; Gurevych, I.** (2013). *Can We Hide in the Web? Large Scale Simultaneous Age and Gender Author Profiling in Social Media*. CLEF 2013 / PAN, CEUR-WS Vol-1179. `ceur-ws.org/Vol-1179/CLEF2013wn-PAN-FlekovaEt2013.pdf` | PAN-2012 AP: 236.600 docs EN, 75.900 ES; 3 grupos de edad; TreeTagger | Edad accuracy **0,53 EN / 0,57 ES** (mayoritario 0,33); sistema final conjunto 0,29 EN / 0,38 ES | **Clave para el paso (d).** *"While in English the ratio of individual part-of-speech tags plays an important role, in Spanish **POS trigrams and quadrigrams are preferred**"*. Tambien: los jovenes usan mas pronombres y menos sustantivos y articulos |
| 2 | **Cruz, F.L.; Haro, R.; Ortega, F.J.** (2013). *ITALICA at PAN 2013: An Ensemble Learning Approach to Author Profiling*. CLEF 2013 / PAN, CEUR-WS Vol-1179. `ceur-ws.org/Vol-1179/CLEF2013wn-PAN-CruzEt2013.pdf` | PAN-2013 AP: 236K EN + 76K ES documentos; Freeling + tagset PAROLE (simplificado y completo) | **3er puesto ES, 10 EN**; ES: conjunto 0,39, genero 0,61, **edad 0,62**; EN: conjunto 0,31, genero 0,54, edad 0,59 | Construye **n-gramas de POS n=1,2,3**, seleccion chi-cuadrado, LibLINEAR, y concluye que *"the use of a more complex set of tags for the POS-tagging in Spanish"* fue clave. Es exactamente el patron de P2 |
| 3 | **Posadas-Duran, J.P.; Markov, I.; Gomez-Adorno, H.; Sidorov, G.; Batyrshin, I.; Gelbukh, A.; Pichardo-Lagunas, O.** (2015). *Syntactic N-grams as Features for the Author Profiling Task*. CLEF 2015 / PAN, CEUR-WS Vol-1391, paper 136. `ceur-ws.org/Vol-1391/136-CR.pdf` | PAN-2015 AP: tweets EN/ES/DU, 100 tweets por autor; CoreNLP/FreeLing/Alpino; SVM | Cifras por idioma en la tabla 1 del paper (columnas de gender/age/personality **no legibles en el texto extraido**: no se citan aqui para no inventar). La metrica relevante es la conclusion textual | **Negativo para edad.** *"their usage is **not that successful for the age and gender**"* (si para personalidad, en RMSE). Es la contraparte honesta de la fila 1 |
| 4 | **Modaresi, P.; Liebeck, M.; Conrad, S.** (2016). *Exploring the Effects of Cross-Genre Machine Learning for Author Profiling in PAN 2016*. CLEF 2016, CEUR-WS Vol-1609, paper 970, pp. 970-977. `ceur-ws.org/Vol-1609/16090970.pdf` | PAN-2016 AP: EN/ES/DU, cross-genre; 436 docs EN, 250 ES; LogReg one-vs-rest, `C = 1e-3` | **1er puesto en genero EN (0,7564), 1er empate en ES**; edad **0,5179 ES** | **Negativo directo sobre el objeto de la fase A.** *"we also tried punctuation n-grams and used polyglot to retrieve **L2-normalized POS-Tag distributions** of UTS tags. Unfortunately, both attempts only **worsened** our results"*; y *"the POS distribution turned out **not to be a good genre-independent feature**"*. Es el antecedente mas cercano a la fase A y es negativo |
| 5 | **Johannsen, A.; Hovy, D.; Sogaard, A.** (2015). *Cross-lingual syntactic variation over age and gender*. CoNLL 2015, pp. 103-112. ACL Anthology `K15-1011`, DOI `10.18653/v1/K15-1011`. `aclanthology.org/K15-1011.pdf` | Trustpilot, 11 idiomas, dependencias universales; seleccion de estabilidad sobre LogReg L1 + chi-cuadrado con Bonferroni | Sin metrica de clasificacion (estudio de variacion); 5 treelets significativos en 10 idiomas para edad | **El estudio mas riguroso que conozco de sintaxis por edad.** *"For age, features vary a lot more than for gender, i.e., there is **less support** for each"*; *"We did not find any evidence for **pervasive** age effects across languages"*. Para genero, 3 treelets atomicos (etiquetas POS) significativos en los 11 idiomas. **Apoya que la edad tiene menos sintaxis discriminantiva que el genero** |
| 6 | **Flekova, L.; Preotiuc-Pietro, D.; Ungar, L.** (2016). *Exploring Stylistic Variation with Age and Income on Twitter*. ACL 2016 (Vol. 2: Short Papers), pp. 313-319. ACL Anthology `P16-2051`. `aclanthology.org/P16-2051.pdf` | 5.191 usuarios con ingreso (10,8M tweets) y **4.279 usuarios con edad (574.095 tweets)**; ArkTweet POS, **universal tagset**; Elastic Net y SVR-RBF, Pearson en CV de 10 folds | Correlacion de Pearson por rasgo (tabla 1 y figura 2); correlaciones por rasgo, no una cifra unica | **El antecedente mas proximo a `P(POS|edad)`.** *"we measure the **ratio of each POS** using the universal tagset"*. Hallazgo: *"increased use of **nouns, determiners and adjectives** is correlated higher with **age** as opposed to income, while a high ratio of **pronouns and interjections** is a good predictor of lower income but, **only to a lesser extent, younger age**, with which it is traditionally associated"*. Es decir: el signo clasico de los pronombres con la edad es **debil**. Unit = usuario, no tweet |
| 7 | **Autores NO verificados** (el fichero fetched no expuso la linea de autores). *Tweets Classification Using Corpus Dependent Tags, Character and POS N-grams*. PAN 2015. `downloads.webis.de/pan/publications/papers/gonzalezgallardo_2015.pdf` | PAN-2015 AP: ES/EN/IT/DU; Freeling con etiquetas dependientes de corpus; LinearSVC | Ranking global **83,46 %** | *"With POS N-grams, in Spanish and English we were able to capture the most representative series of **two and three** grammatical elements"*. Confirma que el patron n-grama de POS es el estandar de la literatura en espanol |
| 8 | **Morgan-Lopez, A.A.; Kim, A.E.; Chew, R.F.; Ruddle, P.** (2017). *Predicting age groups of Twitter users based on language and metadata features*. PLoS ONE 12(8):e0183537. DOI `10.1371/journal.pone.0183537` | 3.184 usuarios (1.036 de 13-17, 1.634 de 18-24, 514 de 25+), hasta 200 tweets por usuario; LogReg L1 one-vs-rest, CV 10 folds + 80/20 | Dummy 38 %; solo-lenguaje **72 % F1**; solo-metadatos 57 % F1; WWBP words 67 % F1; mejor modelo **73,9 % F1** | **El limite de la evidencia publicada.** El mejor resultado con rasgos linguisticos y 3 clases esta en el rango 0,72-0,74 F1, pero **por USUARIO con 200 tweets agregados**. En nuestro caso, un tweet por fila: la unidad amortiza el ruido. Es la razon por la que el techo de 0,616 con 6 clases y 1 tweet por fila no es ni Comparable ni contradice estas cifras |
| 9 | **Schwartz, H.A.; et al.** (2013). *Personality, Gender, and Age in the Language of Social Media: The Open-Vocabulary Approach*. PLoS ONE 8(9):e73791. DOI `10.1371/journal.pone.73791` | World Well-Being Project, ~75.000 participantes de Facebook | correlacion deocabulario con edad/genero; sin metrica por usuario | La plantilla metodologica de `P(unidad|edad)` en grande. No POS. Base de las citas posteriores |
| 10 | **Autores NO verificados** (2024). *Methods and Annotated Data Sets Used to Predict the Gender and Age of Twitter Users: Scoping Review*. J Med Internet Res, `www.jmir.org/2024/1/e47923` | Revision de 74 estudios (8 solo edad) | Edad: F1 **0,31 a 0,94**; accuracy **0,43 a 0,86** | El rango publicado es enorme y los metodos heterogeneos. **El genero se predice mejor que la edad** en la mayoria de los estudios. Sin sintaxis sintetica cuantificable |

### 12.1 Que se sigue de esta revision, con honestidad

1. **Hay evidencia solida de que `P(POS|edad)` existe como patron en espanol**, y de que en espanol
   el patron mas fuerte es **secuencial** (trigramas y cuatrigramas) mas que marginal (fila 1, fila 7,
   fila 2). Esto es un argumento a favor de que la hipotesis no es absurda. **No** es un argumento a
   favor de que el canal predictivo funcione en nuestro corpus.
2. **Hay evidencia publicada contraria y directa.** Filas 3, 4 y 5: los n-gramas sintacticos no
   funcionan para edad (fila 3), la distribucion L2-normalizada de POS empeoro los resultados
   (fila 4), y el estudio multilingue mas riguroso encuentra **menos soporte** para edad que para
   genero (fila 5). Filas 3 y 4 son los dos antecedentes mas cercanos a P2 y a la fase A, y ambos son
   negativos.
3. **El antecedente mas proximo a la fase A (fila 6) reporta un patron con signo mixto**: nouns,
   determiners y adjectives suben con la edad; los pronombies, con el signo clasico pero **debil**.
   Es decir, incluso el trabajo de referencia no sostiene que los pronombres marcen edad. Eso es
   relevante porque V31 gasto su mejor banco (+0,006820) justamente en pronombres.
4. **No he encontrado evidencia de que un clasificador cerrado de Bayes sobre proporciones POS sea la
   cota de potencia de nada.** La fila 6 usa regresion (Elastic Net, SVR), no un plug-in bayesiano.
   La cota de la seccion 5 es una **construccion propia de este plan**, elegida por ser
   cerrada, sin hiperparametros y honesta. No se presenta como replicar literatura.
5. **Las metricas publicadas no son comparables con las nuestras**: distintos idiomas, distintas
   clases de edad (3 frente a 6), distinta unidad (autor con 100-200 tweets frente a un tweet) y
   distintos generos textuales (blogs y cross-genre frente a tweets). Se dice explicitamente para
   que la tabla 12 no se lea como un ranking.

**No hay evidencia solida de que el paso (d) sea una lineaDslGANADORA. Mi veredicto de 6.3 no cambia
por la literatura: cambia por las mediciones internas D-041, D-077 y D-088, que son las que aplican a
este corpus.**

---

## 13. Success Criteria (consolidado, medible)

| # | Criterio | Umbral |
|---|---|---|
| S1 | Integridad del entorno de datos | Las 9 aserciones de T0.2 pasan |
| S2 | Ancla del campeon reproducida | `argmax(v30_prob*pesos)` = `0.6159732` con tolerancia 1e-9 |
| S3 | Anclas de artefactos reproducidas | SHA256 de parquets y `.npy` iguales a los de P2 y al `manifest.json` |
| S4 | Tabla completa | **1728** filas en `v37_pos_por_edad.csv`; `p` suma 1,0 por (anotador, `variante`, `q_estrat`, edad), salvo `sin_space` |
| S5 | Cota sin fuga | `\|A_perm - A0\| <= 0,005` en md y lg |
| S6 | Cota reportada | `A0`, `A1`, `A2` con IC95 por grupos, md y lg, reexpresados contra los dos suelos |
| S7 | Control de fuga | `A1` ajustado solo con `fold_idx != k`; 169.336 filas predichas exactamente una vez |
| S8 | Estabilidad | `dispersion <= 0,005` y `max_dc_md_lg <= 0,005` en las celdas con soporte y significancia |
| S9 | Contraste escrito | Las 3 preguntas de C4 respondidas con numeros; correlacion con D-026 reportada |
| S10 | Auditoria de novedad escrita | Las 5 razones, el veredicto `(ii)` y las 4 decisiones de 6.4 en el informe |
| S11 | Divergencia de transiciones medida | `D = media_c(div_interclase) - media_c(div_ruido)` con IC95 por grupos de duplicados (2000, semilla 20261010), `R` reportado como medida de efecto, y la densidad de la matriz de transiciones publicada |
| S12 | Veredicto coherente | `veredicto` implicado por V1..V6 por la regla de 8.2 |
| S13 | Tiempos | Fase completa **< 15 min**; `--verificar` < 60 s; `--humo` < 60 s |
| S14 | Campeon intacto | `v30_prob`, los 6 pesos, la semilla y los hiperparametros sin cambios (hash) |
| S15 | Sin produccionillycuda | Cero submission, cero train/holdout/leaderboard, cero test |
| S16 | ASCII en codigo | `scripts/` sin acentos, sin emojis, sin CJK (verificado con grep) |
| S17 | Autoejercicio | `--verificar` corre 5 caminos con funciones reales, incluido el camino 4 de escritura en directorio temporal |

---

## 14. Open Questions

1. **Autorizacion del paso (d).** ¿Autoriza Carlos, con hipotesis nueva escrita, un canal predictivo
   de transiciones POS condicionado por edad, si la medicion D1 da `R > 1`? *Respuesta por defecto
   en esta spec: NO, no se ejecuta.*
2. **Que hacer si G1 no pasa** por soporte insuficiente (poco probable: el tag mas raro, `PART`, tiene
   448 ocurrencias en md y hay 18 tags). ¿Se acepta un umbral de soporte mas laxo (100 en vez de
   1000) para las etiquetas de cola, marcados como no reportables? *Propuesta: si, con la marca
   `soporte_insuficiente`, sin cambiar la regla de decision.*
3. **El suelo de 0,0485421 cambia la lectura de las actas previas.** ¿Quiere Carlos que el acta de V37
  Include una nota de re-expresion de P2 y P1a contra el suelo estricto, dado que el proyecto siempre
   ha citado 0,1667? *Propuesta: si, en la seccion 1.3 del informe, sin reescribir las actas
   anteriores.*
4. **Si V37 cierra, ¿que queda abierto?** Segun D-087 el techo es 0,642 y el campeon esta en 0,6159732.
   El margen teorico es de 0,026. ¿Vale la pena otro ciclo, o se declara la exhausting del proyecto?
   *No es decision de esta spec; se registra como pregunta para Carlos.*
5. **`P(tag|edad)` con `lemma_POS` o `morph_seq`.** Cerrados en D-078/D-079/D-080 y fuera del encargo.
   ¿Se mantienen cerrados si V37 dice que la marginal POS no aporta? *No se reopen sin peticion
   explicita.*
6. **Coste de la estandarizacion por longitud.** Quintiles de `n_tokens` con cortes en los
   percentiles 20/40/60/80 es una decision de diseno declarada, no un hyperparameter buscado. ¿La
   acepta, o prefiere deciles (mas finos, mas ruido) o el pooled simple (mas grosero)? *Propuesta:
   quintiles, y ademas publicar las 5 filas para que no dependa de la eleccion.*

---

## 15. Lo que esta spec NO hace

- No entrena ni modifica el campeon. No cambia `v30_prob`, los pesos, la representacion ni la semilla.
- No genera submission. No toca `test`, `holdout` ni `leaderboard`.
- No reanota con spaCy. No usa vectores de spaCy.
- No reentrena nada que no sea el clasificador de Bayes cerrado de la seccion 5, que es el objeto
  medido, no un candidato.
- No ejecuta el canal de transiciones del paso (d). Veredicto `(ii)` en la seccion 6.
- No reabre D-026, V14, V31, D-041, D-077, D-078, D-079, D-080, D-086, D-087 ni D-088.
- No edita ni borra ningun script ya ejecutado.
- No crea un directorio `figuras/` en la raiz: usa `reports/figures/`, que es la convencion de
  `src/config.py` y donde ya viven las 106 figuras del proyecto.