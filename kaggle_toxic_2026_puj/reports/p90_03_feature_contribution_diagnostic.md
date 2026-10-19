# P90-03 - Contribuciones de features de V03, mecanismos de error y gap analysis

Generado por `experiments/p90_03_diagnostico_contribuciones.py` y `experiments/p90_03b_informe_handoff.py`.
Commit auditado: `e43c594917fc643d0e300b73646aefb7dbd6b61d`. Fecha: 2026-10-10. Sin entrenamiento, sin test, sin holdout,
sin submissions, sin commits.

## 1. Resumen ejecutivo

La fase responde la pregunta central con cuatro resultados: (1) **la atribucion exacta OOF esta
BLOCKED_BY_MISSING_ARTIFACTS**: los modelos y vectorizadores por pliegue que generaron las
predicciones OOF no estan serializados (`experiments/14_oof_corregido_v2.py` los ajusta en
memoria y guarda solo probabilidades); (2) el joblib final de V03 (todo el train) se uso como
**evidencia AUXILIAR claramente separada**, con validacion matematica exacta
(sigmoid(z) == predict_proba: 0.0e+00; reconstruccion de z:
residuo maximo 9.3e-15); (3) las firmas de error son
**acumulativas, no de pocas features dominantes** (0 celdas "pocas dominantes" entre los 466
errores lejanos; 76-86 por ciento de los FN son "acumulacion"), con **conflicto word-char en
43-50 por ciento de los FP de HS/AG** y **canal char dominante en 56-62 por ciento de los FN de
HS/AG**; (4) **ninguna hipotesis genuinamente nueva**: los 18 mecanismos evaluados mapean a
familias cerradas o duplicadas del historial (HIB-H11/H12/H14, ENS, REP, targeting,
contextual_selection, reweighting, structural, augmentation, cita/negacion, P68).
Decision final: **D + A** (atribucion exacta bloqueada; sin hipotesis nueva). La evidencia
auxiliar se interpreta solo como estructura del modelo final, nunca como causa de los errores OOF.

## 2. Alcance y compliance

Sin entrenamiento, sin reconstruir modelos por pliegue, sin recalibrar, sin thresholds, sin
submissions, sin test, sin holdout, sin corpus externos, sin pseudo-labeling, sin operaciones
Git destructivas. Solo se leyeron artefactos de desarrollo autorizados, el modelo final
congelado y los textos dev. Las contribuciones por celda son auxiliares; las distancias,
probabilidades y margenes OOF son canonicos.

## 3. Artefactos utilizados

| artefacto | sha256[:16] | uso |
|---|---|---|
| features/oof_v03_final.npz | 559265816ce4f631 | OOF canonico (proba, pred, pliegues, umbrales) |
| features/v2_oof_base_corrected.npz | 27dfa83736b97951 | linaje M2 (verificado exacto) |
| experiments/baseline_v03_frozen/model_v03.joblib | 263ed5bd0fc0de7c | modelo final auxiliar (vectorizador + 3 LR) |
| data/raw/toxic_train.csv | 3d40e7345b588860 | textos de desarrollo |
| reports/v03_error_cells.csv | 6a582e67d6450bcf | 757 celdas error-etiqueta |
| reports/v03_error_controls.csv | 75bc065a15e5a0b5 | 757 controles 1:1 |
| reports/v03_error_review_sample.csv | 33272bbc92a65e81 | muestra de revision |
| reports/eda/fase2_lexico_inducido_agresividad.csv | abcb61f0d6f93f33 | lexico in-sample (categorizacion descriptiva) |

## 4. Disponibilidad de los modelos y vectorizadores por fold

**No disponibles.** El OOF corregido v2 se genero ajustando por pliegue un `TfidfVectorizer`
word + `TfidfVectorizer` char y una `LogisticRegression` por etiqueta, en memoria, guardando
solo `ids, y_true, proba (1483 x 4 modelos x 3), modelos, etiquetas, grupos`. No existe ningun
artefacto joblib/pkl por pliegue en el repositorio (busqueda exhaustiva). Consecuencia:
`EXACT_OOF_ATTRIBUTION = BLOCKED_BY_MISSING_ARTIFACTS`. Artefactos que faltan: modelos ajustados
por pliegue (5 x 3 LR) y sus vectorizadores (5 x word + 5 x char), o equivalentes (coeficientes
y vocabularios por pliegue).

## 5. Limitaciones de atribucion OOF

El modelo final se entreno con todo el train (1731 filas, incluido el desarrollo). Su
probabilidad in-sample no reproduce la decision OOF: el modelo final coincide en direccion con
la prediccion OOF solo en 38/173 (22.0%) de HS FP, 27/82 (32.9%) de TR FP,
38/135 (28.1%) de AG FP, 3/143 (2.1%) de HS FN, 5/82 (6.1%) de
TR FN y 2/142 (1.4%) de AG FN. Por tanto **ninguna contribucion por celda se
interpreta como causa del error OOF**; las tablas por celda llevan
`attribution_status = AUXILIARY_FULL_MODEL_ATTRIBUTION`. La distorsion media |p_aux - p_oof| es
0.109 (p90 0.276).

## 6. Metodologia de descomposicion

Para cada fila i y etiqueta j: `z = intercepto + sum(x_k * coef_k)`, `p = sigmoid(z)`,
`margen = z - logit(threshold)`. La contribucion de la feature k es `x_k * coef_k`. Se separan
los bloques word (`bloque_0__`, 4024 features) y char_wb (`bloque_1__`,
23407 features). Categorias descriptivas de features (documentadas en el
script): URLs, menciones, hashtags, puntuacion, numeros, insultos y vulgaridad (lexico
in-sample existente), expresiones interpersonales, colectivos, elongaciones, char con espacio,
char alfabetico, otras. Firmas de cobertura: pocas features dominantes (top3 >= 50 por ciento de
|contribucion|), acumulacion (top3 < 25 por ciento), conflicto word-char (signos opuestos),
conflicto positivas-negativas (ratio >= 0.3), lexico dominante sin contexto (FP) / sin evidencia
hostil activada (FN), no interpretable. Los controles se comparan por solapamiento de top-5
features (Jaccard >= 0.4) y banderas estructurales (>= 5 de 7 iguales).

## 7. Validacion matematica

- `sigmoid(z) == predict_proba` en las 1483 x 3: maximo 0.0e+00 (exacto).
- `intercepto + suma(contribuciones) == z`: residuo maximo 9.3e-15.
- Linaje del OOF: M2 de `v2_oof_base_corrected.npz` reordenado por ids reproduce la proba canonica con diferencia 0.0.
- Estratos historicos verificados: HS 40/87/189, TR 13/40/111, AG 31/80/166 (total 84/207/466).
- `reconstruction_error` por celda: maximo 3.3e-15.

## 8. Analisis de errores HS

- HS FP (173): media word -0.210063, media char 0.013142;
  conflicto word-char 75/173; acumulacion 70;
  top-1 positivo del lexico 89/173 (perra 32, zorra 9, callate 8...);
  top-negativos recurrentes: en, la, **mi perra (9)**, la puta (7).
- HS FN (143): sin lexico en top10 70/143; char dominante 89/143;
  acumulacion 96; top-negativos: la, de, en, **https (9)**;
  con url en texto 62/143 (43 por ciento).
- Nota: las medias de canal del modelo final en FP (word negativo) reflejan que el modelo final a
  menudo no comete el error (seccion 5); la lectura valida es la estructura de features, no la direccion.

## 9. Analisis de errores TR

- TR FP (82): top-1 del lexico 67/82 (82 por ciento); conflicto word-char 26/82;
  top-positivos: perra (27), zorra (9), eres (8); top-negativos: la puta (8), **mi perra (8)**.
- TR FN (82): mismo signo word-char 82/82 (100 por ciento); sin mencion en texto 50/82 (61 por ciento);
  acumulacion 66; top-positivos: puta (12), perra (10), sudaca (7).
- Estructura: TR FN con HS correcta 34/82; TR FP con HS error simultaneo 67/82 (los errores de TR
  viven mayoritariamente junto a los de HS).

## 10. Analisis de errores AG

- AG FP (135): conflicto word-char 68/135 (50 por ciento); top-1 del lexico 87/135;
  top-positivos: perra (30), zorra (10), callate (6); top-negativos: mi perra (7), la puta (5).
- AG FN (142): char dominante 79/142 (56 por ciento); acumulacion 119/142;
  top-negativos: en, la, el, **rabe (6)**, polla (6), https (5); con url 63/142.

## 11. Word versus char

| etiqueta | tipo | n | word medio | char medio | |word| | |char| | mismo signo | dominantes |
|---|---|---|---|---|---|---|---|---|
| HS | FP | 173 | 19 | 42 | 112 | -0.210063 | 0.013142 | 0.342729 | 0.295491 | 0.5665 | otras (word):95; insultos y vulgaridad:71; expresiones interpersonales:3 |
| HS | FN | 143 | 21 | 45 | 77 | 0.297258 | 0.642523 | 0.349229 | 0.649747 | 0.8252 | otras (word):110; insultos y vulgaridad:33 |
| TR | FP | 82 | 7 | 21 | 54 | 0.406196 | 0.470407 | 0.493366 | 0.522020 | 0.6829 | insultos y vulgaridad:57; otras (word):21; expresiones interpersonales:3 |
| TR | FN | 82 | 6 | 19 | 57 | 0.864343 | 0.933020 | 0.864343 | 0.933020 | 1.0000 | otras (word):47; insultos y vulgaridad:33; expresiones interpersonales:1 |
| AG | FP | 135 | 18 | 40 | 77 | 0.104677 | 0.197783 | 0.333467 | 0.341501 | 0.4963 | insultos y vulgaridad:73; otras (word):49; expresiones interpersonales:12 |
| AG | FN | 142 | 13 | 40 | 89 | 0.528069 | 0.869930 | 0.541589 | 0.869930 | 0.9718 | otras (word):96; insultos y vulgaridad:36; elongaciones:6 |

Masa de coeficientes del modelo final (|coef|): word 0.31 / char
0.69 (HS); 0.30 / 0.70 (TR);
0.30 / 0.70 (AG). El canal char
concentra ~70 por ciento de la masa de pesos pero solo 4024 features word existen: las decisiones
mezclan ambos; el conflicto word-char es una firma sustantiva de los FP de HS/AG y el canal char
domina en los FN de HS/AG. Es una propiedad estructural del modelo union (auxiliar), no una causa
demostrada del error OOF.

## 12. Falsos positivos

| hipotesis | evidencia cuantificada (auxiliar) | estado |
|---|---|---|
| H1 vulgaridad fuerte | top1 lexico HS 89/173, TR 67/82, AG 87/135; coef perra 3.23/4.34/3.25 | CLOSED (reweighting/term weighting) |
| H2 afiliativo | mi perra/la puta con coef NEGATIVO (mi perra: 18 docs, 12 no toxicos; la puta: 98 docs, 88 no toxicos) | CLOSED (HIB-H11; contextual_selection) |
| H3 char_wb superficial | conflicto word-char: HS 75, TR 26, AG 68 | DUPLICATE (ENS/REP) |
| H4 menciones | menciones en top10: HS 2, TR 0, AG 0 | CLOSED (targeting/P68); sin respaldo |
| H5 literales/no dirigidas | sin mencion en texto: HS 90/173, AG 65/135 | CLOSED (contextual_selection) |
| H6 estructura | puntuacion en top10: HS 31, AG 14 | CLOSED (structural_features/HIB) |
| H7 citas | comillas: HS 5, TR 0, AG 0 | CLOSED (P19/P66) |

## 13. Falsos negativos

| hipotesis | evidencia cuantificada (auxiliar) | estado |
|---|---|---|
| H1 indirecta sin insulto | sin lexico top10: HS 70/143, TR 27/82, AG 57/142 | CLOSED (HIB-H14) |
| H2 colectivas | colectivo top10: HS 12, AG 5; rabe: 149 docs, 114 no toxicos | CLOSED (HIB-H14/joint_labels) |
| H3 generalizaciones | acumulacion HS 96/143, AG 119/142 | CLOSED (representacion) |
| H4 TR sin mencion | sin mencion 50/82; TR FN con HS correcta 34/82 | CLOSED (HIB-H12); anotacion BLOCKED |
| H5 retorica/cita/relato | url: HS 62, AG 63; https top-negativo | CLOSED (cita/negacion; registro) |
| H6 ortografia | elongacion: HS 7, AG 6 | CLOSED (augmentation/fuzzy) |
| H7 word-char contradictorio | conflicto: HS 25, TR 0, AG 4 | DUPLICATE (ENS) |
| H8 conflicto evidencia | neg domina: HS 140/143, TR 63/82, AG 126/142 | CLOSED (reweighting/hardness) |

## 14. Errores alejados del threshold

| etiqueta | tipo | n lejos | acumulacion | conflicto wc | mismo signo | top features positivas |
|---|---|---|---|---|---|---|
| HS | FP | 112 | 83 | 63 | 49 | perra:27; los:12; zorra:9; se:5; guarra:5; subsaharianos:4; al:3; cllate:3 |
| HS | FN | 77 | 64 | 20 | 57 | los:9; se:5; al:4; el culo:3; parte:2; perra:2; no para:2; mi polla:2 |
| TR | FP | 54 | 28 | 13 | 41 | perra:20; eres:8; zorra:7; puta:5; guarra:3; callate puta:2; te:2; novia es:2 |
| TR | FN | 57 | 47 | 0 | 57 | puta:7; perra:6; sudaca:5; mi polla:4; zorra:3; te:2; ella:2; callada:2 |
| AG | FP | 77 | 51 | 27 | 50 | perra:25; zorra:9; te:5; guarra:4; cllate:3; eres una:3; callate:3; callate puta:2 |
| AG | FN | 89 | 75 | 2 | 87 | al:11; perra:4; mi polla:3; nos:3; todas las:3; manteros:2; zorra:2; te:2 |

En los 466 errores lejanos no hay ninguna celda dominada por pocas features (0 "pocas
dominantes"): son acumulativos. Los FP lejanos de las tres etiquetas tienen a **perra** como
top-1 positivo en todas las folds (HS: f1 5, f2 zorra 3, f3 8, f4 7, f5 4; AG: 3/5/6/7/4;
TR: 3/4/4/6/3). Los FN lejanos son acumulativos (64-75 de 77-89). No se denomina a los errores
lejanos "sin informacion": la lectura es que la evidencia disponible es insuficiente o
contradictoria en el modelo auxiliar, sin mecanismo nuevo demostrado.

## 15. Comparacion con controles

757 relaciones error-control sobre 454 controles unicos (reuso maximo 8;
284 con reuso 1). Clasificacion (reglas descriptivas): parecido 18, lexicalmente_similar 7,
estructuralmente_parecido 378, sin_control_comparable 354. El criterio estructural (>=5 de 7
banderas) es debil: la media de banderas iguales es 4.69 y las banderas mayoritarias (mayusculas,
mencion) coinciden por frecuencia. El criterio lexical (Jaccard top-5 >= 0.4) casi nunca se
cumple (25/757): los errores y sus controles emparejados por distancia no comparten las mismas
features dominantes (esperable en un espacio disperso de 27431 dimensiones). Subset de la
muestra de revision: 185 relaciones (100 errores), con la
misma estructura. Los controles no son semanticamente equivalentes por construccion.

## 16. Diferencias por slice

| slice | label | errores dentro | word medio | char medio | features activas dentro | features activas fuera |
|---|---|---|---|---|---|---|
| URLs | HS | 112 | -0.077482 | 0.462721 | 248.91 | 184.97 |
| URLs | TR | 39 | 0.569302 | 0.810698 | 202.10 | 172.22 |
| URLs | AG | 102 | 0.221992 | 0.689164 | 258.39 | 190.11 |
| hashtags | HS | 37 | -0.128379 | 0.191623 | 244.08 | 202.80 |
| hashtags | TR | 12 | 0.574941 | 0.329742 | 216.42 | 176.40 |
| hashtags | AG | 30 | 0.186109 | 0.510046 | 260.77 | 209.72 |
| colectivos_superficie | HS | 100 | 0.025705 | 0.349586 | 272.68 | 177.52 |
| colectivos_superficie | TR | 25 | 0.789038 | 0.793685 | 245.32 | 167.46 |
| colectivos_superficie | AG | 95 | 0.354573 | 0.604643 | 288.93 | 176.80 |
| menciones | HS | 123 | -0.039709 | 0.229080 | 206.62 | 208.28 |
| menciones | TR | 83 | 0.649682 | 0.657474 | 182.69 | 175.89 |
| menciones | AG | 107 | 0.316261 | 0.460290 | 212.00 | 217.30 |
| mencion_inicial | HS | 100 | -0.022465 | 0.217210 | 192.98 | 214.42 |
| mencion_inicial | TR | 78 | 0.640290 | 0.657190 | 177.19 | 181.27 |
| mencion_inicial | AG | 88 | 0.315269 | 0.418377 | 191.44 | 226.34 |
| mayusculas_sostenidas | HS | 186 | -0.051353 | 0.328929 | 237.34 | 165.12 |
| mayusculas_sostenidas | TR | 98 | 0.621969 | 0.679523 | 201.18 | 146.88 |
| mayusculas_sostenidas | AG | 173 | 0.251146 | 0.548661 | 242.14 | 170.53 |
| elongaciones | HS | 22 | -0.073959 | 0.170469 | 221.82 | 206.57 |
| elongaciones | TR | 9 | 0.407990 | 0.374353 | 146.11 | 181.26 |
| elongaciones | AG | 18 | 0.148089 | 0.286405 | 203.94 | 216.04 |
| segunda_persona | HS | 134 | 0.031048 | 0.270908 | 216.07 | 201.42 |
| segunda_persona | TR | 96 | 0.636737 | 0.661781 | 188.01 | 167.07 |
| segunda_persona | AG | 120 | 0.351536 | 0.515075 | 228.56 | 205.08 |
| imperativo | HS | 67 | -0.019720 | 0.314082 | 242.28 | 198.31 |
| imperativo | TR | 38 | 0.508674 | 0.735468 | 209.08 | 170.36 |
| imperativo | AG | 57 | 0.317892 | 0.635662 | 267.26 | 201.78 |
| cita | HS | 10 | -0.201242 | 0.386593 | 321.20 | 203.92 |
| cita | TR | 1 | 0.707860 | 0.626937 | 377.00 | 178.12 |
| cita | AG | 6 | -0.044675 | 1.795151 | 263.50 | 214.18 |

Las regiones de bajo rendimiento (URLs, hashtags, colectivos_superficie) tienen errores con
**mas features activas** que el resto (URLs HS 248.9 vs 185.0; colectivos HS 272.7 vs 177.5) y
mezcla de canales distinta, pero las firmas dominantes siguen siendo acumulacion/conflicto: no
aparece una firma propia que no exista fuera. Interpretacion descriptiva; grupos pequenos
(hashtags TR 12 errores) con incertidumbre alta. El enrutado por region fue cerrado (P68).

## 17. Estructura de etiquetas

Identidad TR=>HS verificada (0 violaciones en 1483). TR FN: 48/82 con error HS simultaneo y
34/82 con HS correcta; TR FP: 67/82 con error HS simultaneo y 15/82 con HS correcta. Los errores
de TR acompanan mayoritariamente a los de HS; la factorizacion condicional ya fue explorada y
cerrada (D-044/D-045, familia joint_labels). No se impone ninguna relacion nueva.

## 18. Cruce con resultados historicos

| etiqueta | intercepto | share masa word | top positivos | top negativos |
|---|---|---|---|---|
| HS | -0.3573 | 0.3054 | perra|3.2317; zorra|2.9683; callate puta|1.9234; eres una|1.9083; cllate|1.7805; rra|1.5659; cllate puta|1.5651; sudaca|1.4816; rra |1.4598; callate|1.4030; putos|1.3700; | la puta|-3.7785; hijo|-3.1833; hijo de|-2.8825; mi perra|-2.0785; escoria|-1.9588; puta madre|-1.7855; gera|-1.5499; acoso|-1.5373; puta boca|-1.5218; rabe|-1.5178; la|-1 |
| TR | -0.9999 | 0.3039 | perra|4.3424; zorra|3.8131; cllate|2.4389; puta|2.3404; eres una|2.2231; rra |2.1092; rra|2.0882; callate puta|1.9820; callate|1.9232; eres|1.9195; guarra|1.8284; sudaca| | la puta|-3.3520; hijo|-2.8932; hijo de|-2.6438; mi perra|-1.9821; de|-1.8071; escoria|-1.6387; en|-1.6363; los|-1.4900; puta boca|-1.4777; puta madre|-1.4731; son|-1.4028 |
| AG | -0.7346 | 0.2985 | perra|3.2467; zorra|3.2423; """|2.8700; eres una|2.1363; """"|2.1170; cllate|2.0855; callate puta|2.0676;  ""|1.9207; "" |1.8393; rra|1.6973; rra |1.6447; cllate puta|1.5 | la puta|-3.8145; hijo|-3.0789; hijo de|-2.8042; escoria|-1.7166; mi perra|-1.7036; puta madre|-1.5971; en|-1.4834; puta boca|-1.3997; de puta|-1.2173; rabe|-1.1768; polla |

La tabla completa de gap analysis (18 mecanismos) esta en `p90_03_hypothesis_gap_analysis.csv`.
Resumen: 15 CLOSED (incluida FN-H4, cuyo apendice de anotacion queda BLOCKED_BY_RULES),
2 DUPLICATE (FP-H3, FN-H7), 1 UNRESOLVED (X2), 0 POTENTIALLY_NEW. Los precedentes mas directos: HIB-H11 (bloques
bromista/interpersonal, -0.0213), HIB-H12 (objetivo individual/colectivo, -0.0091), HIB-H14
(lexico inducido x colectiva, -0.0297), HIB-UNION (-0.0316), ENS-01..04 (pesos por
representacion), REP-05/06 (union adoptada), contextual_selection (P0=V03), reweighting V30/P69,
structural V15-01, augmentation V6-08, fuzzy V29, cita/negacion P19, P68 enrutado.

## 19. Hipotesis cerradas

FP-H1, FP-H2, FP-H4, FP-H5, FP-H6, FP-H7, FN-H1, FN-H2, FN-H3, FN-H4, FN-H5, FN-H6, FN-H8,
X1, X3 (15 filas CLOSED) y las dos DUPLICATE (FP-H3, FN-H7): todas con experimento historico
equivalente identificado y resultado conocido.

## 20. Hipotesis complementarias

No se identifica ninguna linea complementaria nueva con evidencia suficiente para experimentar.
Las observaciones de P90-00/P90-02 (casebook, slices, grupos) siguen siendo el diagnostico
valido; P90-03 agrega la estructura de contribuciones auxiliares y la confirma.

## 21. Posibles nuevas hipotesis

**Ninguna.** La novedad exigida (mecanismo o procedimiento no evaluado) no se cumple en ninguna
de las firmas observadas: reweighting por familia, gating contextual, ponderacion de canales,
bloques estructurales y enrutado por region tienen experimentos historicos ejecutados y
cerrados. La unica linea realmente no ejecutada es la atribucion exacta OOF, bloqueada por
artefactos faltantes (no por reglas), y la revision humana de anotacion, bloqueada por reglas
(P87). No se propone experimentar.

## 22. Limitaciones

- Atribucion exacta OOF bloqueada (modelos por pliegue no serializados); la evidencia auxiliar
  del modelo final no explica las decisiones OOF (acuerdo 1.4-32.9 por ciento).
- El modelo final vio el desarrollo en entrenamiento: sus probabilidades y contribuciones
  in-sample estan infladas; solo se usan para estructura de features.
- Lexico in-sample (150 lemas) usado solo para categorizar descriptivamente.
- Controles no independientes, reutilizados y emparejados por distancia; los umbrales de
  similitud (Jaccard 0.4; 5/7 banderas) son convenciones descriptivas declaradas.
- Los grupos pequenos de slices tienen incertidumbre alta; sin inferencia poblacional.
- Distancia al threshold no es confianza calibrada (heredado de P90.02).

## 23. Decision final

Clasificacion del diagnostico: **VERIFIED** (validacion matematica, linaje, estratos,
reconstruccion); **SUPPORTED** (estructura acumulativa del error; lexico de misoginia como top-1
positivo en FP; evidencia negativa anti-toxica aprendida; char dominante en FN; conflicto
word-char en FP); **TENTATIVE** (lecturas de registro/contexto en FN); **UNRESOLVED** (mecanismo
especifico de los 466 errores lejanos).

Hipotesis: 15 CLOSED, 2 DUPLICATE, 0 POTENTIALLY_NEW, 1 UNRESOLVED (X2).

**Recomendacion combinada: D. EXACT_OOF_ATTRIBUTION_BLOCKED + A. NO_NEW_HYPOTHESIS.**
La atribucion exacta no es posible sin reentrenar los modelos por pliegue (prohibido); ninguna
hipotesis nueva, concreta, falsable y no duplicada sobrevive al cruce con el historial. V03 se
mantiene FROZEN; no se recomienda reemplazarlo ni experimentar en esta fase.


## Anexo - Asociaciones descriptivas de frases clave (desarrollo)

| frase | docs dev | HS=1 | TR=1 | AG=1 | sin etiqueta (000) |
|---|---|---|---|---|---|
| perra | 197 | 155 | 152 | 149 | 40 |
| zorra | 104 | 91 | 91 | 89 | 13 |
| mi perra | 18 | 5 | 5 | 5 | 12 |
| la puta | 98 | 10 | 9 | 8 | 88 |
| hijo | 113 | 7 | 3 | 7 | 106 |
| escoria | 37 | 4 | 2 | 4 | 33 |
| acoso | 56 | 6 | 1 | 5 | 49 |
| rabe | 149 | 31 | 2 | 24 | 114 |
| sudaca | 24 | 20 | 7 | 17 | 3 |
| callate | 186 | 104 | 104 | 103 | 82 |
| eres una | 33 | 30 | 30 | 29 | 3 |

Lectura descriptiva (sin construir lexico nuevo): "perra" y "zorra" son mayoritariamente
toxicas en el corpus (155/197 y 91/104 con HS=1) y reciben los coeficientes positivos mas altos;
"mi perra" (12/18 sin etiqueta), "la puta" (88/98), "hijo" (106/113), "escoria" (33/37),
"acoso" (49/56) y "rabe" (114/149) son mayoritariamente no toxicas y reciben coeficientes
negativos. Esta estructura explica el mecanismo observado: los FP ocurren cuando el lexico
positivo domina sin contrapeso afiliativo, y los FN cuando tokens con asociacion anti-toxica
aprendida (rabe, acoso, hijo, la puta) restan puntuacion en textos que si son toxicos.


## Anexo - Firmas por etiqueta y tipo (recuentos)

| etiqueta | tipo | firma | n |
|---|---|---|---|
| HS | FP | conflicto word-char | 75 |
| HS | FP | acumulacion de muchas features | 70 |
| HS | FP | conflicto positivas-negativas | 27 |
| HS | FP | no interpretable con los artefactos disponibles | 1 |
| HS | FN | conflicto word-char | 25 |
| HS | FN | acumulacion de muchas features | 96 |
| HS | FN | conflicto positivas-negativas | 20 |
| HS | FN | sin evidencia hostil activada | 2 |
| TR | FP | conflicto word-char | 26 |
| TR | FP | acumulacion de muchas features | 30 |
| TR | FP | conflicto positivas-negativas | 25 |
| TR | FP | no interpretable con los artefactos disponibles | 1 |
| TR | FN | acumulacion de muchas features | 66 |
| TR | FN | conflicto positivas-negativas | 6 |
| TR | FN | pocas features dominantes | 2 |
| TR | FN | sin evidencia hostil activada | 4 |
| TR | FN | no interpretable con los artefactos disponibles | 4 |
| AG | FP | conflicto word-char | 68 |
| AG | FP | acumulacion de muchas features | 45 |
| AG | FP | conflicto positivas-negativas | 22 |
| AG | FN | conflicto word-char | 4 |
| AG | FN | acumulacion de muchas features | 119 |
| AG | FN | conflicto positivas-negativas | 14 |
| AG | FN | pocas features dominantes | 1 |
| AG | FN | sin evidencia hostil activada | 2 |
| AG | FN | no interpretable con los artefactos disponibles | 2 |

