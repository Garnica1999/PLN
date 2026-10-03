# V37 - DIAGNOSTICO `P(POS | edad)` - informe final

**Fecha**: 2026-10-01 | **Fase**: V37-DIAG-POS (CPU pura, D-007) | **Runner**: `scripts/age_linguistic_14_diagnostico_pos.py` | **Pre-registro congelado**: `experiments/age_linguistic_13_preregistro_v37.json`

## Veredicto

**DIAGNOSTICO_CERRADO**

- Rama del arbol de la seccion 8.3 ejecutada: `G2 falla -> DIAGNOSTICO_CERRADO por inestabilidad. Campeon conservado.`
- Campeon: **NO TOCADO**. V30 + plug-in sigue en 0.6159732 (macro-F1 CV corregida), con los seis pesos congelados. Ni `v30_prob` ni los pesos se han escrito.
- Corrida predictiva ejecutada: **ninguna**. No hay submission, no se ha tocado test, holdout ni leaderboard, y no se ha usado transformers, LLM ni embeddings.

## 0. Instrumento y suelos

| Referencia | macro-F1 |
|---|---|
| Azar uniforme (6 clases equiprobables) | 0.166667 |
| Clase mayoritaria constante (suelo estricto) | 0.0485421 |
| Campeon V30 + plug-in (leido, no reentrenado) | 0.6159732 |

Recuento de clases: [28863, 28676, 28347, 28230, 28060, 27160]; ratio max/min = 1.0627. **Rebalanceo: NO APLICA**.

## 1. FASE A - perfil `P(tag | edad)`

Perfil crudo del anotador primario (md), en proporcion de aparicion por token. `SPACE` esta **dentro** de la secuencia: 18 etiquetas, no 17.

| tag | 13-17 | 18-24 | 25-34 | 35-49 | 50-64 | 65-xx | pooled | delta min | delta max |
|---|---|---|---|---|---|---|---|---|---|
| ADJ | 0.0531 | 0.0541 | 0.0567 | 0.0603 | 0.0656 | 0.0620 | 0.0591 | -0.0060 | +0.0066 |
| ADP | 0.0748 | 0.0822 | 0.0963 | 0.1038 | 0.1134 | 0.1139 | 0.0989 | -0.0240 | +0.0151 |
| ADV | 0.0356 | 0.0379 | 0.0332 | 0.0301 | 0.0278 | 0.0259 | 0.0313 | -0.0054 | +0.0066 |
| AUX | 0.0283 | 0.0313 | 0.0282 | 0.0268 | 0.0257 | 0.0247 | 0.0273 | -0.0026 | +0.0040 |
| CCONJ | 0.0244 | 0.0234 | 0.0231 | 0.0227 | 0.0240 | 0.0224 | 0.0233 | -0.0009 | +0.0011 |
| DET | 0.0648 | 0.0734 | 0.0779 | 0.0817 | 0.0899 | 0.0823 | 0.0790 | -0.0142 | +0.0109 |
| INTJ | 0.0058 | 0.0056 | 0.0042 | 0.0033 | 0.0023 | 0.0035 | 0.0040 | -0.0017 | +0.0018 |
| NOUN | 0.1235 | 0.1289 | 0.1377 | 0.1479 | 0.1537 | 0.1530 | 0.1419 | -0.0184 | +0.0118 |
| NUM | 0.0220 | 0.0174 | 0.0192 | 0.0197 | 0.0191 | 0.0197 | 0.0195 | -0.0022 | +0.0025 |
| PART | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | -0.0001 | +0.0000 |
| PRON | 0.0632 | 0.0653 | 0.0525 | 0.0447 | 0.0405 | 0.0371 | 0.0495 | -0.0124 | +0.0158 |
| PROPN | 0.2305 | 0.2066 | 0.2141 | 0.2151 | 0.2062 | 0.2149 | 0.2143 | -0.0081 | +0.0162 |
| PUNCT | 0.1213 | 0.1201 | 0.1198 | 0.1134 | 0.1021 | 0.1164 | 0.1151 | -0.0129 | +0.0062 |
| SCONJ | 0.0245 | 0.0280 | 0.0226 | 0.0202 | 0.0171 | 0.0160 | 0.0210 | -0.0050 | +0.0070 |
| SPACE | 0.0196 | 0.0127 | 0.0104 | 0.0091 | 0.0149 | 0.0105 | 0.0127 | -0.0036 | +0.0069 |
| SYM | 0.0030 | 0.0029 | 0.0030 | 0.0027 | 0.0023 | 0.0031 | 0.0028 | -0.0005 | +0.0003 |
| VERB | 0.1054 | 0.1098 | 0.1006 | 0.0982 | 0.0950 | 0.0942 | 0.1000 | -0.0058 | +0.0098 |
| X | 0.0002 | 0.0002 | 0.0001 | 0.0002 | 0.0001 | 0.0003 | 0.0002 | -0.0000 | +0.0001 |

- Tokens totales: 2,590,877; media 15.30 tokens/tweet; mediana 15.
- Filas con `pos_seq` vacio (`n_tokens == 0`): **104** (0.061 %). Regla E5: `x = 0` y luego `pred = argmax_c pi_c`; se incluyen y se cuentan.
- `SPACE`: 32,865 ocurrencias en 25,232 filas. Cuota por clase (token): 13-17 1.956 %, 18-24 1.269 %, 25-34 1.040 %, 35-49 0.911 %, 50-64 1.486 %, 65-xx 1.049 %.
- La cuota de `SPACE` **dentro** de la estandarizacion por quintiles de `n_tokens` (E10), que es donde queda realmente confundida con la longitud:

| quintil | 13-17 | 18-24 | 25-34 | 35-49 | 50-64 | 65-xx |
|---|---|---|---|---|---|---|
| q1 | 2.482 % | 1.687 % | 1.589 % | 1.136 % | 1.351 % | 1.005 % |
| q2 | 2.178 % | 1.126 % | 1.192 % | 0.975 % | 1.357 % | 0.948 % |
| q3 | 1.938 % | 1.256 % | 1.053 % | 0.907 % | 1.424 % | 1.272 % |
| q4 | 1.737 % | 1.233 % | 0.960 % | 0.903 % | 1.499 % | 1.059 % |
| q5 | 1.761 % | 1.280 % | 0.888 % | 0.861 % | 1.564 % | 0.956 % |

- `SPACE` es el unico 'tag' confundido con la longitud por construccion (viene de `token.is_space`) y el mas variable entre clases: el cociente entre la cuota maxima y la minima entre clases es 2.15x. Se declara como limite.
- Media de proporciones por tweet (E12) frente a la razon de sumas que usa `p`: correlacion de Pearson 0.995558 y desviacion maxima absoluta 0.021680. `p[c,t] = n_tokens[c,t] / n_tokens[c]` NO es la media de proporciones por tweet: es una razon de sumas ponderada por token, y por tanto confunde preferencia POS con longitud. Se publica en paralelo para que el problema de mezcla a nivel de documento quede visible.

## 2. FASE A - cota de potencia discriminativa

| Brazo | md (pooled) | md (media por fold) | lg (pooled) | lg (media por fold) |
|---|---|---|---|---|
| A0 (mayoritaria constante) | 0.0485421 | - | 0.0485421 | - |
| A1 (NB sobre proporciones) | 0.1720532 | 0.1720444 | 0.1739055 | 0.1738838 |
| A2 (NB sobre conteos crudos) | 0.2052270 | 0.2052195 | 0.2063921 | 0.2063651 |
| A_perm (control de fuga) | 0.0799411 | nan | 0.0806730 | nan |
| A_perm_val (control de evaluacion) | 0.0670040 | nan | 0.0673039 | nan |

IC95 por grupos de duplicados (2000 iteraciones, unidad = cluster, 164,018 grupos, semilla 20261010):

| Brazo | IC95 md | IC95 lg |
|---|---|---|
| A1 (md) | [0.1703007, 0.1738377] | |
| A1 (lg) | [0.1721453, 0.1756758] | |
| A2 (md) | [0.2031891, 0.2072232] | |
| A2 (lg) | [0.2042258, 0.2084497] | |
| A_perm (md) | [0.0788982, 0.0809877] | |
| A_perm (lg) | [0.0796719, 0.0816878] | |

- Control de cobertura de grupos: filas efectivas por iteracion entre 168,311 y 170,445 (referencia 169,336); 100.00 % de las iteraciones dentro de +-5 %.
- **E1**: el unico criterio de `A_perm` es unilateral `A_perm <= 0.176667` (azar + 0.010). La banda `|A_perm - A0| <= 0.005` queda **eliminada** en los cinco sitios de la spec. Medido: md 0.0799411, lg 0.0806730; el permutado **degenera hacia la clase mayoritaria** (predice la clase 0 en 116,439 de 169,336 filas en md), que es lo que E1 explica y lo que impide tratarlo como un control que 'cae al azar'.
- **E1, instrumento de inferencia**: IC95 pareado de `A1 - A_perm` = md [0.0900353, 0.0942188], lg [0.091205, 0.0953618]; excluye cero: md True, lg True.
- **E2**: `A2` NO es identico a `A1`. Medido `A2 - A1` = +0.0331738 en md y +0.0324865 en lg; 48,440 filas cambian de prediccion en md. La invariancia solo se cumple sin prior; con `log pi_c` sin escalar, el factor `n_i` cambia el balance prior/verosimilitud fila a fila. `A2` es el **brazo de sensibilidad a la longitud**, no una cuantificacion de 'cuanto de A1 es longitud'.
- **E3**: `A_perm_val` (se ajusta con las etiquetas reales y se puntua contra `y` permutado a nivel de cluster) = md 0.0670040, lg 0.0673039; criterio `<= 0.176667`: True.
- **E6**: A1 se reporta en las dos convenciones (pooled sobre las 169,336 filas y media de macro-F1 por fold). Las diferencias contra P2 y P1a son **estimaciones puntuales** con el IC de cada lado en el JSON; no se afirma significancia porque son estimadores y convenciones distintas.

## 3. FASE B - estabilidad

- Anotador md: `dispersion[c,t] = max_k p_k - min_k p_k`; maximo global 0.003479 sobre las celdas con soporte alto.
- Anotador lg: `dispersion[c,t] = max_k p_k - min_k p_k`; maximo global 0.003630 sobre las celdas con soporte alto.
- Maximo cruzado entre anotadores (criterio conservador de V2 y G2, no una columna de la tabla): 0.003630.
- `max_dc_md_lg`: maximo global 0.013349.
- **E13**: la coincidencia de secuencia fila a fila entre md y lg es 0.1635 y el desajuste de presencia es 0.00 %. Los parquets congelados guardan **solo UPOS**, luego el tag especifico del modelo nunca se congelo: ambos anotadores proyectan sobre el mismo inventario de 18 por construccion y **no son anotadores independientes** (mismo pipeline, mismo tokenizer). B2 queda como sanity check de la proyeccion universal y el 16.3 % se publica como **limitacion del conjunto de datos**, no como resultado.
- IC95 por grupos completo para las 6x18 celdas: md 83 celdas cuyo IC95 excluye cero y 0 con el delta fuera del IC; lg 89 y 0.

## 4. FASE C - contraste con lo ya medido

- **TC.1**: correlacion de Pearson por fila (cifra literal de TC.1) 0.576471, Spearman 0.622534 sobre 20,000 filas muestreadas (semilla 20261012). Pearson agregado sobre las 18 etiquetas: 0.998798 (Spearman 0.977296).
  la cifra literal de TC.1 compara un vector de 18 componentes MEDIANTE EL TWEET con un vector que solo toma 6 valores distintos (el perfil de su clase), de modo que su techo es una correlacion atenuada por construccion y no es comparable con el umbral 0,9 de la spec, que esta pensado para dos series de 18 puntos. La cifra que responde a la pregunta de C ('es la misma senal que D-026?') es la agregada y la por clase, y se publica la literal por trazabilidad

### Las tres preguntas de C4, respondidas con numeros

- **C4.1 - ?la cota supera, iguala o queda por debajo de P2 (0.2750921, canal pos_seq md)?** queda **por debajo**: A1(md) = 0.1720532 con IC95 [0.1703007, 0.1738377], diferencia puntual -0.1030389. La informacion marginal de `P(POS|edad)` es un **subconjunto estricto** de lo que P2 ya exploto como n-gramas 1-3, luego la conclusion correcta es redundante, no nueva. No se afirma significancia: son estimadores y convenciones distintas (E6).
- **C4.2 - ?queda por debajo de P1a (0.3997), la direccion dual `P(clase|token)` con cross-fit?** queda **muy por debajo**: A1(md) = 0.1720532, diferencia -0.2276468. Confirma que el refinamiento por etiqueta no aporta sobre el refinamiento por token, que es justo lo que ya habia mostrado P1a con integracion -0.0002 y controles permutados empatados.
- **C4.3 - ?la cota supera el suelo estricto 0.0485421 por un margen que justifique abrir una linea?** A1(md) = 0.1720532 esta +0.1235111 sobre el suelo estricto y +0.0053862 sobre el azar uniforme. Supera el suelo estricto por un margen amplio, pero **supera el azar por menos de 0.01** (0.0054), luego la senal por si sola no es accionable y la familia se cierra. Correlacion agregada con las 18 proporciones de D-026: 0.998798 (Pearson) y 0.977296 (Spearman), luego V37 es una descomposicion legible de D-026 y no un hallazgo. A1(lg) = 0.1739055.

### Discrepancia de denominador `SPACE` (TC.3)

- D-026 usaba `tags = [token.pos_ for token in document if not token.is_space]`; el parquet congelado **incluye** `SPACE`: 32,865 ocurrencias en 25,232 filas (14.9 % de las filas).
- Consecuencia 1: las proporciones de D-026 y las de V37 **no son la misma variable**; la de V37 esta desinflada en un 1.268 % por token.
- Consecuencia 2: `SPACE` es un indicador de saltos de espacio del tokenizador, no una categoria gramatical; es el unico 'tag' con semantica de formato y el unico confundido con la longitud por construccion.
- Consecuencia 3: los conteos de V37 incluyen 32,865 tokens no gramaticales (1.268 % de 2,590,877), lo que afecta al denominador de todos los IC.

## 5. FASE D - auditoria de novedad del canal de transiciones

Las tres opciones son: (i) misma familia ya cerrada; (ii) variante disfrazada; (iii) genuinamente distinto. El veredicto de la seccion 6.3 de la spec es **(ii) VARIANTE DISFAZADA**, y la medicion de abajo no habilita ninguna corrida.

1. **Por que no es (i).** P2 uso `TfidfVectorizer(1-3)` + `LogReg` sobre `pos_seq`, que representa efectos aditivos. `T_c[a,b] = P(POS_(i+1)=b | POS_i=a, edad=c)` exige el termino de interaccion de orden tres `1{POS_i=a} x 1{POS_(i+1)=b} x 1{edad=c}`, que un modelo lineal no puede expresar a partir de margenes de orden uno y dos.
2. **Por que no es (iii)**, con evidencia medida: la literatura en espanol recomienda precisamente lo que P2 ya midio (Flekova y Gurevych 2013: *in Spanish POS trigrams and quadrigrams are preferred*; Cruz, Haro y Ortega 2013: n-gramas POS 1-3). Corregido segun E14, el 0,57 ES / 0,53 EN de Flekova y Gurevych son del sistema **solo-edad de 3 grupos** (mayoritario 0,33), no del conjunto de 6 clases, y el 0,62 de Cruz es el accuracy del **ensemble apilado completo**, no del bloque POS; el estudio multilingue mas riguroso (Johannsen, Hovy y Sogaard 2015) encuentra **menos soporte** para edad que para genero, y reporta genero sobre clases lexicas BabelNet, no sobre treelets POS.
3. El refinamiento por clase ya se midio en la direccion dual: P1a (D-041) dio 0.3997 de bloque solo, integracion -0.0002 con 3/5 folds y controles permutados **empatados**.
4. El mecanismo 'canal de transiciones integrado con el campeon' ya fallo dos veces por el mismo camino: P2 -0.0086357 con 0/5 folds y V36-E1 (D-088), senal real y especifica por documento y aun asi cerrada porque *el campeon ya la expresa mejor*.
5. La senal marginal que domina esta familia ya esta medida y es debil: D-026 -0.0032, V14 -0.0015, V31 +0.0000642, P2 -0.0086357. Ninguna abrio nada.

**Medicion D1** (descriptiva, sin entrenar nada): 2,421,645 bigramas, celdas (fold, clase, mitad) = 60, mitad por `splitmix64(cluster_id)` con semilla declarada 0 (sin aleatoriedad de datos).

- `media_c(div_interclase)` = 1.104614
- `media_c(div_ruido)` = 0.535050
- `D` = 0.569564, IC95 pareado [0.379760, 0.684608], **excluye cero = True**
- `R` = 2.0645 (IC95 [1.5748, 2.3354]), publicado como medida de efecto; la decision la toma el IC95 de `D`, no `R > 1`.
- Densidad de la matriz: 18 de 324 pares `(a,b)` con soporte >= 100 bigramas; por clase [16, 16, 16, 16, 18, 17].

**Ninguna de las dos ramas autoriza una corrida predictiva.**

**E15**: no queda ninguna familia legal no medida con potencial. Los subtipos UD de `dep_seq` (22 ratios, sonda 0.2147) y la morfologia completa (23 ratios, sonda 0.2064) caen en la misma banda 0.19-0.24 que las seis familias ya cerradas, muy por debajo de P2 (0.2751), y con fusion siempre <= 0. `SPACE` aislado da 0.04854, exactamente la prediccion constante de la mayoritaria: su aporte es renormalizacion, no senal propia. La decision de conservar el campeon no es austeridad sino el cierre de un espacio ya agotado.

## 6. Gates y criterios

| Gate | Criterio | Resultado |
|---|---|---|
| G1 | n_tokens[c,t] >= 1000 en al menos 12 de las 18 etiquetas para al menos 5 de las 6 clases; las tres variantes calculadas y publicadas; cota A1 para md y lg con IC95 por grupos y A_perm <= 0.176667 (E1) | **pasa: 6/6 clases con soporte; fuga limpia: True** |
| G2 | dispersion[c,t] <= 0.005 y max_dc_md_lg[c,t] <= 0.005 en las celdas con n_tokens >= 5000 y z_bonferroni significativo; IC95 por grupos completo para las 6x18 celdas | **NO PASA: 27/27 estables, 20/27 de acuerdo** |
| G3 | TC.1 + TC.2 + TC.3 completos: sin las tres hay STOP | **pasa: contraste escrito (obligacion de forma, sin umbral de paso)** |
| G4 | auditoria de novedad con las 5 razones de 6.2 y medicion D1 completa; G4 NO autoriza ninguna corrida predictiva | **pasa: auditoria escrita + D1 completa; el canal predictivo NO se ejecuta** |

| Criterio | Umbral | Valor | Cumple |
|---|---|---|---|
| V1 | existe delta[c,t] >= 0.010 con z_bonferroni significativo para al menos 3 etiquetas y al menos 4 de las 6 clases | {"n_etiquetas": 5, "etiquetas": ["ADP", "DET", "NOUN", "PRON", "PROPN"], "n_clases": 4, "clases": ["13-17", "18-24", "50-64", "65-xx"], "n_celdas": 8} | **True** |
| V2 | esas mismas etiquetas cumplen dispersion[c,t] <= 0.005 | {"n_celdas_evaluadas": 20, "n_celdas_que_cumplen": 20, "dispersion_max_sobre_esas_celdas": 0.00363} | **True** |
| V3 | esas mismas etiquetas cumplen max_dc_md_lg[c,t] <= 0.005 | {"n_celdas_evaluadas": 20, "n_celdas_que_cumplen": 14, "max_dc_md_lg_max_sobre_esas_etiquetas": 0.013349} | **False** |
| V4 | la version estandarizada por longitud conserva |p_est[c,t] - ref[t]| >= 0.008 para esas mismas etiquetas | {"n_celdas_evaluadas": 20, "n_celdas_que_cumplen": 16, "delta_estandarizado_min_sobre_esas_celdas": 0.000371} | **False** |
| V5 | A1 > azar_uniforme + 0.010 en md y lg | {"A1_md": 0.17205317, "A1_lg": 0.17390555, "posicion_sin_umbral": {"A1_md": 0.17205317, "P1a": 0.3997, "P2_md": 0.2750921, "P2_lg": 0.272549, "campeon": 0.6159732487657944}} | **False** |
| V6 | A_perm <= 0.176667 y A_perm_val <= 0.176667 en md y lg, y el IC95 pareado de A1 - A_perm excluye cero | {"A_perm_md": 0.07994113, "A_perm_lg": 0.08067303, "A_perm_val_md": 0.06700395, "A_perm_val_lg": 0.06730392, "ic95_A1_menos_A_perm_md": [0.0900353, 0.0942188], "ic95_A1_menos_A_perm_lg": [0.091205, 0.0953618], "criterio_ | **True** |

**E4 (FPR conjunta de V1..V4)**: bajo permutacion de etiquetas sobreviven 0 etiquetas y 0 celdas de V1, frente a 5 etiquetas y 8 celdas reales. si el nulo ya supera los floors de V1..V4, los floors estan mal calibrados y el V1 real no puede leerse como senal. Se declara en el informe en vez de ocultarlo (E4)

**E11**: el error analitico se publica en las dos unidades. Con unidad token (pooled, conservador) hay 27 celdas significativas con Bonferroni; con unidad tweet (delta method) hay 55. El token-iid era el error, no el conservadurismo: la unidad de muestreo es el tweet. Aun asi, **la inferencia la gobierna el bootstrap por grupos de duplicados**, que es el instrumento correcto, y todo 'no significativo' del error token-iid se ha re-derivado y publicado.

## 7. Persistencia de evidencia (E8)

- `artifacts/v37/a1_prob_oof.npy`: probabilidades OOF de A1 indexadas por fila, anotador primario `md`, forma [169336, 6], SHA256 `9f7837032f0cbcaf...`. Se escribe **siempre**, tambien con veredicto de cierre, porque es evidencia reutilizable y no una apertura al campeon.
- `ids.npy` de referencia: SHA256 `d69ecb339f1dc57d...`; el orden de filas es el canonico.
- Prueba concreta que ejecutaria una spec posterior si el patron resultara potente (declarada en el pre-registro): Si el patron resultara potente (V1..V6 los cinco true), la spec posterior que se ejecutaria esta escrita por adelantado y es esta: 'P(POS_{i+1} | POS_i, edad) con LogReg L2 C=1.0 sobre indicadores de uni/bi/trigramas POS multiplicados por un indicador de edad, LeaveOneOut por cluster, integrado con el campeon por alfa 0.5 con los pesos congelados, gate delta >= +0.0015 y 4/5 folds'. Esa prueba NO se ejecuta en V37: V37 no produce ninguna corrida predictiva (enmienda E8)

## 8. Artefactos de esta corrida

- `experiments/age_linguistic_14_diagnostico_pos.json`
- `artifacts/v37/a1_prob_oof.npy` (SHA256 `9f7837032f0cbcaf...`)
- `reports/tables/v37_pos_por_edad.csv` (SHA256 `d140bafac36ff647...`)
- `reports/tables/v37_pos_estabilidad.csv` (SHA256 `7bf353662652666d...`)
- `reports/tables/v37_pos_ic_por_grupo.csv` (SHA256 `315e7b612d21391a...`)
- `reports/tables/v37_pos_contraste.csv` (SHA256 `48fa6cda975690a9...`)
- `reports/tables/v37_pos_transiciones.csv` (SHA256 `a45c412e7eb54cf2...`)
- `reports/v37_diagnostico_pos.md` (SHA256 `99ec488f4f6eba11...`)
- `reports/figures/v37_pos_01_perfil_por_clase.png` (SHA256 `6cc141d1fab8107a...`)
- `reports/figures/v37_pos_02_perfil_md_vs_lg.png` (SHA256 `eb01b02db930e4f7...`)
- `reports/figures/v37_pos_03_estabilidad_folds.png` (SHA256 `3cabe5e9341d74b0...`)
- `reports/figures/v37_pos_04_ic_por_grupo.png` (SHA256 `02efb662436cbdee...`)
- `reports/figures/v37_pos_05_transiciones_divergencia.png` (SHA256 `055c550a8fcba741...`)

## 9. Limites declarados

- La cota de la seccion 5 es una **construccion propia de este plan**, cerrada y sin hiperparametros. No se presenta como replicar literatura y no existe evidencia publicada de que un clasificador cerrado de Bayes sobre proporciones POS sea la cota de potencia de nada.
- Las metricas publicadas no son comparables con las nuestras: 3 grupos de edad y accuracy contra 6 clases y macro-F1, y unidad de autor con 100-200 tweets contra un tweet por fila.
- `p[c,t]` es una razon de sumas ponderada por token, no la media de proporciones por tweet; la version de media se publica en paralelo pero no es la que entra en el criterio.
- B2 es un sanity check de la proyeccion universal, no una comparacion entre anotadores, porque los parquets solo guardan UPOS (E13).
- **E9 (sonda previa)**: antes de congelar el pre-registro se ejecuto una sonda con los modelos exactos de la seccion 5.2, sin ajustar nada. La corrida debe reproducirla: A1 dif 3.0e-08 y A2 dif 0.0e+00 frente a la tolerancia declarada 1.0e-06. Es un test de reproducibilidad, no una medida de resultado.
- **`A_perm` es estocastico**: depende de la permutacion implementada, luego NO se compara con los valores de la sonda previa sino por su criterio unilateral y por la banda declarada. El valor medido puede variar entre implementaciones sin que ello indique fuga ni desalineacion.
- La sonda previa se declaro para el anotador primario (md). Para lg se publica el valor medido pero NO se exige tolerancia: comparar lg contra una sonda calculada con md seria exigir que dos anotadores den el mismo numero.
- El bootstrap de la fase D reestima las matrices de transiciones en cada replica; las 2000 replicas son pareadas (el mismo remuestreo de grupos para `div_interclase` y para `div_ruido`).
- No se ha reevaluado ni reentrenado el campeon; `v30_prob` y los seis pesos del plug-in estan intactos.

