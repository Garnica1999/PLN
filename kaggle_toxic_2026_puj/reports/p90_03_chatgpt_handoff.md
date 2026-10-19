# P90-03 - Handoff autocontenido: contribuciones de features de V03 y mecanismos de error

Commit auditado: `e43c594917fc643d0e300b73646aefb7dbd6b61d`. Fecha: 2026-10-10. Sin entrenamiento, sin test, sin
holdout, sin submissions, sin commits. Este documento contiene los hallazgos completos; las
tablas viven en `reports/p90_03_*.csv`.

## 1. Baseline y estado Git

V03 FROZEN: TF-IDF word(1,2)+char_wb(3,5), min_df=2, sublinear_tf, lowercase, strip_accents=None;
LogisticRegression C=2.0 balanced liblinear x3; umbrales 0.44/0.51/0.46; corrected_cv
0.7772114525165721; LB 0.78306. Git: rama main, HEAD e43c594917fc643d0e300b73646aefb7dbd6b61d, cambios locales de
P90.02 preservados, sin commits ni push en esta fase.

## 2. Artefactos utilizados y hashes

OOF canonico `features/oof_v03_final.npz` (559265816ce4f631); padre v2
`features/v2_oof_base_corrected.npz` (27dfa83736b97951, linaje M2 verificado exacto);
modelo final `experiments/baseline_v03_frozen/model_v03.joblib` (263ed5bd0fc0de7c);
textos `data/raw/toxic_train.csv` (3d40e7345b588860); casebook/celdas/controles/muestra
(6a582e67d6450bcf/75bc065a15e5a0b5/33272bbc92a65e81);
lexico in-sample (abcb61f0d6f93f33).

## 3. Disponibilidad de modelos OOF por fold

**AUSENTES.** `experiments/14_oof_corregido_v2.py` ajusta por pliegue vectorizadores y LR en
memoria y guarda solo probabilidades. EXACT_OOF_ATTRIBUTION = BLOCKED_BY_MISSING_ARTIFACTS.
Disponible solo el modelo final (todo el train) como AUXILIARY_FULL_MODEL_ATTRIBUTION.

## 4. Metodologia de contribuciones

z = intercepto + sum(x_k*coef_k); p = sigmoid(z); margen = z - logit(threshold); contribucion_k =
x_k*coef_k; bloques word (4024) y char_wb (23407); categorias
descriptivas de features y firmas de cobertura (pocas dominantes, acumulacion, conflicto
word-char, conflicto pos-neg, lexico sin contexto / sin evidencia hostil, no interpretable).

## 5. Validacion de la reconstruccion del score

sigmoid(z) == predict_proba: maximo 0.0e+00 (exacto);
intercepto + suma(contribuciones) == z: residuo maximo 9.3e-15;
reconstruction_error por celda <= 3.3e-15; estratos historicos verificados (HS 40/87/189,
TR 13/40/111, AG 31/80/166). El modelo final coincide en direccion con la prediccion OOF en
38/173 (22.0%), 27/82 (32.9%), 38/135 (28.1%) (FP) y
3/143 (2.1%), 5/82 (6.1%), 2/142 (1.4%) (FN): por eso la
atribucion por celda es auxiliar y no causal.

## 6. Hallazgos por etiqueta

| etiqueta | tipo | n | cerca | intermedio | lejos | word medio | char medio | mismo signo |
|---|---|---|---|---|---|---|---|---|
| HS | FP | 173 | 19 | 42 | 112 | -0.210063 | 0.013142 | 0.5665 |
| HS | FN | 143 | 21 | 45 | 77 | 0.297258 | 0.642523 | 0.8252 |
| TR | FP | 82 | 7 | 21 | 54 | 0.406196 | 0.470407 | 0.6829 |
| TR | FN | 82 | 6 | 19 | 57 | 0.864343 | 0.933020 | 1.0000 |
| AG | FP | 135 | 18 | 40 | 77 | 0.104677 | 0.197783 | 0.4963 |
| AG | FN | 142 | 13 | 40 | 89 | 0.528069 | 0.869930 | 0.9718 |

- HS FP: top-1 positivo del lexico 89/173; conflicto word-char
  75/173; top-negativos en, la, mi perra (9), la puta (7).
- HS FN: sin lexico top10 70/143; char dominante
  89/143; https entre los top-negativos (9).
- TR FP: top-1 lexico 67/82 (82 por ciento); perra 27, zorra 9.
- TR FN: sin mencion 50/82; mismo signo 82/82; puta 12, perra 10, sudaca 7.
- AG FP: conflicto word-char 68/135 (50 por ciento); perra 30.
- AG FN: char dominante 79/142; rabe (6) y https (5) entre top-negativos.

## 7. Firmas de falsos positivos

Acumulacion 70/30/45
(HS/TR/AG); conflicto word-char 75/26/68;
conflicto pos-neg 27/25/22;
pocas dominantes ~0. Mecanismo descriptivo: lexico de misoginia (perra/zorra/callate/eres una)
con coeficientes positivos altos sin contrapeso activo; los marcadores afiliativos
("mi perra", "la puta") ya tienen coeficiente NEGATIVO aprendido (asociacion descriptiva:
mi perra 12/18 y la puta 88/98 sin etiqueta en dev) pero insuficiente en estos FP.

## 8. Firmas de falsos negativos

Acumulacion 96/66/119;
char dominante HS/AG 89/79; neg domina
(ratio>=0.3) 140/63/126.
Evidencia negativa recurrente: palabras funcionales (en, la, de, el), https y tokens con
asociacion anti-toxica aprendida (rabe, acoso, hijo, la puta). El canal char concentra la senal
positiva disponible (56-62 por ciento) pero insuficiente.

## 9. Word frente a char

Masa de |coef|: word ~0.30 / char ~0.70 en las tres etiquetas. Conflicto word-char: FP
75/26/68;
FN 25/0/4.
Mismo signo: FN TR 82/82; FN AG 138/142. El conflicto de canales es firma de una parte
sustancial de los FP; ponderar canales es familia ENS/REP ya ejecutada (DUPLICATE).

## 10. Errores alejados del threshold

466/757 (61.6 por ciento). Lejanos por etiqueta/tipo: HS FP 112, HS FN 77,
TR FP 54, TR FN 57, AG FP 77, AG FN 89.
Cero celdas "pocas dominantes"; "perra" es el top-1 positivo de FP lejanos en las cinco folds;
FN lejanos acumulativos (64-75 de 77-89). Sin mecanismo nuevo; la distancia no es calibracion.

## 11. Comparacion con controles

757 relaciones / 454 controles unicos (reuso max 8): parecido 18,
lexicalmente_similar 7, estructuralmente_parecido 378, sin_control_comparable 354. El criterio
estructural es debil (media 4.69 banderas iguales); el lexical casi nunca se cumple (25/757).
No son semanticamente equivalentes; lectura cualitativa.

## 12. Patrones por slice

URLs/colectivos: los errores dentro tienen mas features activas (248.9 vs 185.0; 272.7 vs 177.5
para HS) y mezcla de canales distinta, pero sin firma propia nueva; hashtags con grupos pequenos
(12-37 errores). Enrutado por region cerrado (P68).

## 13. Hipotesis contrastadas

18 mecanismos: 15 CLOSED (FN-H4 incluida, con su apendice de anotacion TR en
BLOCKED_BY_RULES), 2 DUPLICATE (FP-H3 char/canales, FN-H7 conflicto de canales),
1 UNRESOLVED (X2 mecanismo de lejanos). 0 POTENTIALLY_NEW.
Tabla completa: `p90_03_hypothesis_gap_analysis.csv`.

## 14. Experimentos historicos relacionados

HIB-H11 -0.0213 (bromista/interpersonal), HIB-H12 -0.0091 (individual/colectivo), HIB-H14
-0.0297 (lexico x colectiva), HIB-UNION -0.0316; ENS-01..04 (pesos por representacion,
descartados); REP-05/06 (union adoptada); contextual_selection (P0=V03); reweighting V30/P69;
structural V15-01; augmentation V6-08; fuzzy V29; cita/negacion P19; P66 negacion no adoptada;
P68 enrutado; joint_labels cerrado (D-045).

## 15. Hipotesis genuinamente nuevas

**Ninguna.** No se propone candidato (archivo de candidatos con fila NO_CANDIDATE explicita).
La unica linea no ejecutada es la atribucion exacta OOF (bloqueada por artefactos) y la revision
de anotacion (bloqueada por reglas, P87).

## 16. Limitaciones

Atribucion exacta bloqueada; evidencia auxiliar in-sample (no causal); lexico in-sample solo
descriptivo; controles no independientes; grupos pequenos; distancia no calibrada.

## 17. Decision final

Clasificacion: VERIFIED (validacion), SUPPORTED (firmas acumulativas, lexico dominante en FP,
evidencia negativa anti-toxica, char dominante en FN, conflicto word-char en FP), TENTATIVE
(registro/contexto), UNRESOLVED (mecanismo de lejanos).
**D. EXACT_OOF_ATTRIBUTION_BLOCKED + A. NO_NEW_HYPOTHESIS.** V03 FROZEN.

## Anexo - Asociaciones descriptivas de frases clave (dev)

| frase | docs | HS=1 | TR=1 | AG=1 | 000 |
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

