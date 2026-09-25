# Screening de candidatos finales (FASE FINAL)

Fecha: 2026-09-25. Registro: `experiments.csv` con 125 filas (118 previas + 7 filas `FIN-*` de esta fase).

Objetivo: decidir si alguna linea de la investigacion v2 justifica cambiar
el modelo final V03 (LB 0.78306) o gastar sondas de leaderboard. Las
categorias son: A mejora standalone robusta; B mejora pequena consistente;
C complementariedad fuerte; D local no estable; E negativo; F no
transferido al leaderboard; G trazabilidad (referencia, control u
artefacto de auditoria).

## Conteo por categoria

| categoria | n |
|---|---|
| A - Mejora standalone robusta | 5 |
| B - Mejora pequena consistente | 9 |
| C - Complementariedad fuerte | 3 |
| D - Local no estable | 18 |
| E - Negativo | 67 |
| F - No transferido al leaderboard | 8 |
| G - Trazabilidad / sin modelo nuevo | 15 |
| **Total** | **125** |

## A - Mejora standalone robusta (5)

Adoptado con la regla vigente (nivel +0.005 y >= 3/5 pliegues o complementariedad con IC > 0).

| experiment_id | f1 | decision | hipotesis (resumen) | nota |
|---|---|---|---|---|
| E2-02-META | 0.7853 | ADOPTADA (Gate P1) | LR regularizada sobre probabilidades, margenes, entropias y rasgos estructurales mejora el corrected_cv | 0.7853 (+0.0081 vs V03), 4/5 pliegues, IC [+0.0021,+0.0248]; D-017 |
| E2-18-META-BASES-RIDGE | 0.7874 | ADOPTADO (Gate P6) | La fusion tardia regularizada de scores OOF de bases + canales de targeting + incertidumbre + estructura super | 0.7874, 4/5 pliegues; ensemble_gain +0.0089 IC [+0.0019,+0.0160]; D-019 |
| FIN-01-C-REF-44 | 0.7874 | PROMISING | Control: el meta adoptado en P6 (44 columnas) se reproduce con este pipeline | Reproduce 0.7874 exacto con el OOF congelado (control de comparabilidad) |
| S03 | 0.7989 | MANTENER los umbrales por etiqueta: v03 (0.78306) supera a v02 (0.7799 | Los umbrales optimos por etiqueta superan la regla pre-registrada (+0.005 absoluto y >= 3/5 pliegues) | Unica mejora confirmada en el LB de forma pareada (+0.0031: v02 0.77993 -> v03 0.78306) |
| TH-01 | 0.7989 | adoptada: los umbrales optimos se aplican en la submission v03 | Con class_weight=balanced el umbral 0.5 no es optimo y un umbral por etiqueta sube f1_media_columnas al menos  | Umbrales por etiqueta sobre el OOF (0.7989 historico): base de la decision D-012 |

## B - Mejora pequena consistente (9)

Delta positivo con 3/5 pliegues; por debajo del umbral de adopcion de nivel.

| experiment_id | f1 | decision | hipotesis (resumen) | nota |
|---|---|---|---|---|
| FIN-02-C-EMO | 0.7809 | no adoptado | La fusion tardia de las probabilidades de V03 con emocion mejora la referencia V03 (0.7772) con >= 3/5 pliegue | 0.7809 (+0.0037, 3/5 pliegues); holdout 0.8364 (+0.0081 pareado, IC incluye 0) |
| FIN-04-C-EMO-SYN | 0.7797 | no adoptado | La fusion tardia de las probabilidades de V03 con emocion, sintaxis mejora la referencia V03 (0.7772) con >= 3 | 0.7797 (+0.0025, 3/5); holdout 0.8336 (+0.0052 pareado, IC incluye 0) |
| FIN-05-C-SYN | 0.7793 | no adoptado | La fusion tardia de las probabilidades de V03 con sintaxis mejora la referencia V03 (0.7772) con >= 3/5 pliegu | 0.7793 (+0.0021, 3/5 pliegues) |
| MOD-01_BEST | 0.7849 | adoptada | Un margen maximo (SVM) o un modelo probabilistico distinto mejora la LR del baseline | Mejor variante de MOD-01 (0.7849); paso previo a la union |
| MOD-03a_BEST | 0.7723 | adoptada | Un margen maximo (SVM) o un modelo probabilistico distinto mejora la LR del baseline | 0.7723; paso intermedio de la familia union |
| MOD-03b_BEST | 0.7927 | adoptada | Un margen maximo (SVM) o un modelo probabilistico distinto mejora la LR del baseline | Mejor de la familia union historica (0.7927); D-014: la banda 0.7703-0.7831 es 1 error estandar |
| MOD-03b_W12 | 0.7699 | adoptada | Un margen maximo (SVM) o un modelo probabilistico distinto mejora la LR del baseline | Formulacion W12 (0.5) del adoptado; superada por TH-01 |
| REP-05 | 0.7865 | adoptada | Los n-gramas de caracter capturan evasion ortografica mejor que los de palabra (H1) y su union aporta (H4) | Union word+char adoptada (0.7865) en el protocolo historico |
| REP-06 | 0.7855 | adoptada | Los n-gramas de caracter capturan evasion ortografica mejor que los de palabra (H1) y su union aporta (H4) | Variante de union adoptada (0.7855); diferencia con REP-05 dentro del ruido |

## C - Complementariedad fuerte (3)

Sin mejora standalone suficiente pero con ganancia de fusion e IC bootstrap > 0.

| experiment_id | f1 | decision | hipotesis (resumen) | nota |
|---|---|---|---|---|
| E2-18-META-BASES | 0.7850 | No adoptado / diagnostico | La fusion tardia regularizada de scores OOF de bases + canales de targeting + incertidumbre + estructura super | 0.7850; ensemble_gain +0.0154 IC > 0 (candidato v2-A por complementariedad, D-017) |
| ENS-02 | 0.7919 | descartada: mejor ensemble anidado (0.7949) pero no supera a MOD-03b p | La diversidad entre word, char_wb, union y LinearSVC aporta senal complementaria que el single union+LR no cap | Promedio M2+M3: 0.7949 anidado, ensemble_gain +0.0023; no adoptado por nivel |
| FIN-06-C-REF-44-CH | 0.7782 | COMPLEMENTARY | Los canales v2 aportan al meta adoptado de 44 columnas (P6 solo habia probado los canales de targeting) | 0.7782 (-0.0092 vs el meta de 44 columnas); fusion con V03 +0.0096 IC [+0.0022,+0.0172] |

## D - Local no estable (18)

Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable.

| experiment_id | f1 | decision | hipotesis (resumen) | nota |
|---|---|---|---|---|
| DL-01 | 0.7949 | supera a MOD-03b_BEST en OOF: candidata a modelo final, pero requiere  | La compresion a 300 componentes descarta los n-gramas poco frecuentes que sostienen al modelo lineal; con 1.38 | PCA de la union a 300 componentes: 0.7949 en OOF; nunca se probo en el LB |
| E2-01-AG-HS-TR | 0.7726 | No adoptada (no supera la regla D-016) | TR y AG mejoran si conocen P(HS) y P(TR) como feature continua (E2-01) | Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable. |
| E2-01-AG-HS-TR-STD | 0.7672 | No adoptada (no supera la regla D-016) | TR y AG mejoran si conocen P(HS) y P(TR) como feature continua (E2-01) | Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable. |
| E2-01-HS-AG-TR | 0.7744 | No adoptada (no supera la regla D-016) | TR y AG mejoran si conocen P(HS) y P(TR) como feature continua (E2-01) | Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable. |
| E2-01-HS-AG-TR-STD | 0.7716 | No adoptada (no supera la regla D-016) | TR y AG mejoran si conocen P(HS) y P(TR) como feature continua (E2-01) | Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable. |
| E2-01-HS-TR-AG | 0.7737 | No adoptada (no supera la regla D-016) | TR y AG mejoran si conocen P(HS) y P(TR) como feature continua (E2-01) | Cadenas de etiquetas: +0.0022, 1/5 pliegues; cerrada (E2-01) |
| E2-01-HS-TR-AG-STD | 0.7711 | No adoptada (no supera la regla D-016) | TR y AG mejoran si conocen P(HS) y P(TR) como feature continua (E2-01) | Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable. |
| E2-13-META-ENT-LR | 0.7845 | CERRADA por sparsity | P(label/entidad) con suavizado aporta para TR/AG (E2-13) | Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable. |
| E2-13-META-ENT-RIDGE | 0.7885 | CERRADA por sparsity | P(label/entidad) con suavizado aporta para TR/AG (E2-13) | 0.7885 por encima del meta pero el encoder de entidades cubre 0.5 % del corpus (E2-13) |
| E2-14-MEJOR-META | 0.7857 | CERRADA sin mejora | La calibracion mejora la F1 y la estabilidad de umbrales (E2-14) | Calibracion: Brier 0.1583 -> 0.1199 y log-loss 0.5002 -> 0.4035, F1 invariante (E2-14) |
| E2-18-META+CH | 0.7807 | No adoptado / diagnostico | La fusion tardia regularizada de scores OOF de bases + canales de targeting + incertidumbre + estructura super | 0.7807; anadir los canales de targeting al meta no aporta (delta -0.0046 vs v2-A, D-019) |
| E2-18-META+CH+ESTR | 0.7844 | No adoptado / diagnostico | La fusion tardia regularizada de scores OOF de bases + canales de targeting + incertidumbre + estructura super | 0.7844; 2/5 pliegues vs v2-A |
| E2-18-META-BASES-ANID | 0.7853 | No adoptado / diagnostico | La fusion tardia regularizada de scores OOF de bases + canales de targeting + incertidumbre + estructura super | 0.7853: reproduce el meta anidado de P1 con 0/5 pliegues mejorados (empate) |
| E2-21-TR | 0.7934 | diagnostico E2-21 | Cada etiqueta puede necesitar una representacion distinta (E2-21) | 0.7934 con espacio propio de TR, pero el conjunto por etiqueta cae a 0.7519 (E2-21) |
| FIN-03-C-EMO-margenes | 0.7798 | no adoptado | La fusion tardia de las probabilidades de V03 con emocion mejora la referencia V03 (0.7772) con >= 3/5 pliegue | C-EMO + margen/entropia de V03: 0.7798 (+0.0026 pero 2/5 pliegues) |
| PRE-03 | 0.7623 | descartada | Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7) | Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable. |
| PRE-05 | 0.7607 | descartada | Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7) | Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable. |
| PRE-06 | 0.7586 | descartada | Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7) | Delta positivo o ambiguo sin consistencia entre pliegues ni IC bootstrap favorable. |

## E - Negativo (67)

Delta negativo claro frente a su referencia; linea cerrada.

| experiment_id | f1 | decision | hipotesis (resumen) | nota |
|---|---|---|---|---|
| B5-DENSE | 0.6607 | descartada | Las conjunciones explicitas de registro y objetivo aportan senal que los n-gramas no representan (H11, H12, H1 | Delta negativo claro frente a su referencia; linea cerrada. |
| DL-02 | 0.7410 | por debajo de B1 (0.7587) y de MOD-03b_BEST (0.7927): descartada; con  | Una CNN de caracteres necesita muchos mas datos que 1.385 ejemplos por pliegue para aprender embeddings utiles | Delta negativo claro frente a su referencia; linea cerrada. |
| DL-03 | 0.7364 | por debajo de B1 (0.7587) y de MOD-03b_BEST (0.7927): descartada; con  | El orden de las palabras aporta poco en tweets de ~16 palabras y el vocabulario del pliegue (~1.700 tipos) es  | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-03-POS | 0.5637 | NO adoptada: -0.2135 vs V03 (0/5 pliegues); linea de sintaxis/cita/neg | POS n-gramas como TF-IDF aportan estructura gramatical que word/char no capturan (E2-03) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-03-POS-TXT | 0.7656 | NO adoptada: -0.0116 vs V03 (0/5 pliegues); linea de sintaxis/cita/neg | POS n-gramas como TF-IDF aportan estructura gramatical que word/char no capturan (E2-03) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-04-DEPS | 0.5514 | NO adoptada: -0.2258 vs V03 (0/5 pliegues); linea de sintaxis/cita/neg | Los patrones de dependencia (PRON_VERB, NEG_VERB, AMOD_HEAD) son senal complementaria (E2-04, H12) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-04-DEPS-TXT | 0.7653 | NO adoptada: -0.0119 vs V03 (1/5 pliegues); linea de sintaxis/cita/neg | Los patrones de dependencia (PRON_VERB, NEG_VERB, AMOD_HEAD) son senal complementaria (E2-04, H12) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-04-MORPH | 0.5477 | NO adoptada: -0.2295 vs V03 (0/5 pliegues); linea de sintaxis/cita/neg | La morfologia de persona/tiempo/modo distingue agresion (E2-04, H12) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-04-MORPH-TXT | 0.7567 | NO adoptada: -0.0205 vs V03 (0/5 pliegues); linea de sintaxis/cita/neg | La morfologia de persona/tiempo/modo distingue agresion (E2-04, H12) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-04-SYN3 | 0.5865 | NO adoptada: -0.1907 vs V03 (0/5 pliegues); linea de sintaxis/cita/neg | La sintaxis completa aporta estructura que word/char no capturan (E2-04, H12) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-04-SYN3-TXT | 0.7376 | NO adoptada: -0.0396 vs V03 (0/5 pliegues); linea de sintaxis/cita/neg | La sintaxis completa aporta estructura que word/char no capturan (E2-04, H12) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-05-CH-TGT | 0.6076 | NO adoptada: -0.1696 vs V03 (0/5 pliegues); canal guardado solo para e | La relacion target-insulto (tipo de target, distancia, orden) y el colectivo explican confusiones de targeting | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-05-CH-TGT-DIST | 0.6484 | NO adoptada: -0.1288 vs V03 (0/5 pliegues); canal guardado solo para e | La relacion target-insulto (tipo de target, distancia, orden) y el colectivo explican confusiones de targeting | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-06-TXT-CH | 0.7356 | NO adoptada: -0.0416 vs V03 (1/5 pliegues); canal guardado solo para e | La relacion target-insulto (tipo de target, distancia, orden) y el colectivo explican confusiones de targeting | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-06-TXT-TGT-SPARSE | 0.7487 | NO adoptada: -0.0285 vs V03 (1/5 pliegues); canal guardado solo para e | La relacion target-insulto (tipo de target, distancia, orden) y el colectivo explican confusiones de targeting | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-07-CITA | 0.7437 | NO adoptada: -0.0335 vs V03 (1/5 pliegues); linea de sintaxis/cita/neg | Separar autor de citado mejora HS/AG en regimen noticia (E2-07, H13) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-07-CITA-NEG-TXT | 0.7552 | NO adoptada: -0.0220 vs V03 (1/5 pliegues); linea de sintaxis/cita/neg | Cita y negacion juntas explican el regimen de reporte y la inversion de polaridad (E2-07/E2-08) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-07-CITA-TXT | 0.7554 | NO adoptada: -0.0218 vs V03 (1/5 pliegues); linea de sintaxis/cita/neg | Separar autor de citado mejora HS/AG en regimen noticia (E2-07, H13) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-08-NEG | 0.5267 | NO adoptada: -0.2505 vs V03 (0/5 pliegues); linea de sintaxis/cita/neg | La negacion cambia la etiqueta ('no es X' vs 'es X') (E2-08) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-08-NEG-TXT | 0.7657 | NO adoptada: -0.0115 vs V03 (0/5 pliegues); linea de sintaxis/cita/neg | La negacion cambia la etiqueta ('no es X' vs 'es X') (E2-08) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-09-EMO | 0.6230 | NO adoptada: -0.1542 vs V03 (0/5 pliegues) | La emocion del autor complementa word/char de V03 (H5) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-09-EMO-EST | 0.5897 | NO adoptada: -0.1875 vs V03 (0/5 pliegues) | El insulto en registro de broma o afiliacion (risa, afecto, vocativo) es el falso positivo dominante (H5) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-09-EMO-LEX | 0.6205 | NO adoptada: -0.1567 vs V03 (0/5 pliegues) | La polaridad local alrededor de la mencion o del colectivo separa agresion de afecto (H5/E04) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-09-TXT-EMO | 0.6330 | NO adoptada: -0.1442 vs V03 (0/5 pliegues) | La emocion del autor complementa word/char de V03 (H5) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-10-ORT-A | 0.7486 | NO adoptada: -0.0286 vs V03 (0/5 pliegues) | El raw conserva el estilo y las normalizaciones pueden destruir senal (E2-10) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-10-ORT-ABCD | 0.7534 | NO adoptada: -0.0238 vs V03 (0/5 pliegues) | Un canal normalizado recupera lexemas ofuscados y el raw conserva estilo (E2-10) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-10-ORT-AE | 0.7496 | NO adoptada: -0.0276 vs V03 (0/5 pliegues) | El enmascarado separa la presencia de insulto de su lexema concreto (E2-10) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-13-ENT-SOLO-LR | 0.5393 | CERRADA por sparsity | P(label/entidad) con suavizado aporta para TR/AG (E2-13) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-15-NB-CHAR | 0.7380 | NO adoptada: -0.0392 vs V03 (0/5 pliegues) | El peso NB sobre caracteres captura ofuscacion que word(1,2) pierde (E2-15) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-15-NB-WORD | 0.7538 | NO adoptada: -0.0234 vs V03 (0/5 pliegues) | El peso NB da un modelo diverso y complementario (E2-15) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-15-NBSVM | 0.7612 | NO adoptada: -0.0160 vs V03 (2/5 pliegues) | El peso log-count ratio sobre la union de V03 da el modelo complementario mas fuerte (E2-15) | 0.7612 standalone; su correccion mutua frente a V03 es alta pero no se adopto como canal |
| E2-15-NBSVM-SVC | 0.7429 | NO adoptada: -0.0343 vs V03 (0/5 pliegues) | El margen de la SVC sobre las features NB mejora la diversidad del canal (E2-15) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-16-LOCAL-SIN-SHRINKAGE | 0.7692 | diagnostico E2-16 | Las rebanadas con distribucion distinta se benefician de un umbral propio (H15/E2-16) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-16-MEJOR | 0.7750 | CERRADA sin mejora | El shrinkage controla el sobreajuste de los umbrales locales (E2-16) | Umbrales por rebanada con shrinkage: 0.7750, por debajo del umbral global anidado (E2-16) |
| E2-16-SHRINK-M100 | 0.7750 | diagnostico E2-16 | Las rebanadas con distribucion distinta se benefician de un umbral propio (H15/E2-16) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-16-SHRINK-M25 | 0.7706 | diagnostico E2-16 | Las rebanadas con distribucion distinta se benefician de un umbral propio (H15/E2-16) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-16-SHRINK-M50 | 0.7723 | diagnostico E2-16 | Las rebanadas con distribucion distinta se benefician de un umbral propio (H15/E2-16) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-18-META+CH+ESTR-ANID | 0.7802 | No adoptado / diagnostico | La fusion tardia regularizada de scores OOF de bases + canales de targeting + incertidumbre + estructura super | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-18-META+CH+ESTR-RIDGE | 0.7822 | No adoptado / diagnostico | La fusion tardia regularizada de scores OOF de bases + canales de targeting + incertidumbre + estructura super | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-18-META+CH+ESTR-SVC | 0.7809 | No adoptado / diagnostico | La fusion tardia regularizada de scores OOF de bases + canales de targeting + incertidumbre + estructura super | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-20-ESPECIALISTAS | 0.7749 | CERRADA sin mejora | Las rebanadas significativamente distintas se benefician de un modelo propio (E2-20) | 11 de 21 celdas significativas pero los especialistas rinden -0.045 dentro de su rebanada |
| E2-21-AG | 0.7251 | diagnostico E2-21 | Cada etiqueta puede necesitar una representacion distinta (E2-21) | Delta negativo claro frente a su referencia; linea cerrada. |
| E2-21-ESPACIOS | 0.7519 | CERRADA sin mejora | Cada etiqueta puede necesitar una representacion distinta (E2-21) | Espacios por etiqueta: -0.0253 (0/5 pliegues) |
| E2-21-HS | 0.7372 | diagnostico E2-21 | Cada etiqueta puede necesitar una representacion distinta (E2-21) | Delta negativo claro frente a su referencia; linea cerrada. |
| ENS-01 | 0.7906 | descartada: F1 anidada 0.7906 por debajo del mejor ensemble (0.7919) | La diversidad entre word, char_wb, union y LinearSVC aporta senal complementaria que el single union+LR no cap | Delta negativo claro frente a su referencia; linea cerrada. |
| ENS-03 | 0.7836 | descartada: F1 anidada 0.7836 por debajo del mejor ensemble (0.7919) | La diversidad entre word, char_wb, union y LinearSVC aporta senal complementaria que el single union+LR no cap | Delta negativo claro frente a su referencia; linea cerrada. |
| ENS-04 | 0.7916 | descartada: F1 anidada 0.7916 por debajo del mejor ensemble (0.7919) | La diversidad entre word, char_wb, union y LinearSVC aporta senal complementaria que el single union+LR no cap | Delta negativo claro frente a su referencia; linea cerrada. |
| FIN-07-C-EMO-SYN-TGT-CYT | 0.7777 | no adoptado | La fusion tardia de las probabilidades de V03 con emocion, sintaxis, targeting, cita_negacion mejora la refere | Delta negativo claro frente a su referencia; linea cerrada. |
| HIB-H11 | 0.7713 | descartada | El registro bromista (risa, afecto, diminutivos, vocativo con posesivo, cita) modula la lectura del insulto (H | Delta negativo claro frente a su referencia; linea cerrada. |
| HIB-H12 | 0.7836 | descartada | Separar objetivo individual de objetivo colectivo (nombre propio, segunda persona, referencia colectiva, atrib | Delta negativo claro frente a su referencia; linea cerrada. |
| HIB-H14 | 0.7630 | descartada | El lexico agresivo inducido dentro del pliegue y su cruce con la referencia colectiva recuperan la agresion im | Delta negativo claro frente a su referencia; linea cerrada. |
| HIB-UNION | 0.7610 | descartada | La union de n-gramas con conjunciones densas reduce los falsos positivos de broma y los falsos negativos de od | Delta negativo claro frente a su referencia; linea cerrada. |
| MOD-01_W12 | 0.7699 | adoptada | Un margen maximo (SVM) o un modelo probabilistico distinto mejora la LR del baseline | Delta negativo claro frente a su referencia; linea cerrada. |
| MOD-02_BEST | 0.6888 | descartada | Un margen maximo (SVM) o un modelo probabilistico distinto mejora la LR del baseline | Delta negativo claro frente a su referencia; linea cerrada. |
| MOD-02_W12 | 0.7362 | descartada | Un margen maximo (SVM) o un modelo probabilistico distinto mejora la LR del baseline | Delta negativo claro frente a su referencia; linea cerrada. |
| MOD-03a_W12 | 0.7442 | descartada | Un margen maximo (SVM) o un modelo probabilistico distinto mejora la LR del baseline | Delta negativo claro frente a su referencia; linea cerrada. |
| MOD-04 | 0.7870 | descartada (empeora a su origen MOD-03b_BEST) | Un margen maximo (SVM) o un modelo probabilistico distinto mejora la LR del baseline | Delta negativo claro frente a su referencia; linea cerrada. |
| PRE-01 | 0.7359 | descartada | Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7) | Delta negativo claro frente a su referencia; linea cerrada. |
| PRE-02 | 0.7562 | descartada | Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7) | Delta negativo claro frente a su referencia; linea cerrada. |
| PRE-04 | 0.7577 | descartada | Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7) | Delta negativo claro frente a su referencia; linea cerrada. |
| PRE-07 | 0.7587 | descartada | Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7) | Delta negativo claro frente a su referencia; linea cerrada. |
| PRE-08 | 0.7584 | descartada | Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7) | Delta negativo claro frente a su referencia; linea cerrada. |
| PRE-09 | 0.7425 | descartada | Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7) | Delta negativo claro frente a su referencia; linea cerrada. |
| REP-01 | 0.7567 | descartada | Los n-gramas de caracter capturan evasion ortografica mejor que los de palabra (H1) y su union aporta (H4) | Delta negativo claro frente a su referencia; linea cerrada. |
| REP-02 | 0.7511 | descartada | Los n-gramas de caracter capturan evasion ortografica mejor que los de palabra (H1) y su union aporta (H4) | Delta negativo claro frente a su referencia; linea cerrada. |
| REP-03 | 0.7417 | descartada | Los n-gramas de caracter capturan evasion ortografica mejor que los de palabra (H1) y su union aporta (H4) | Delta negativo claro frente a su referencia; linea cerrada. |
| REP-04 | 0.7562 | descartada | Los n-gramas de caracter capturan evasion ortografica mejor que los de palabra (H1) y su union aporta (H4) | Delta negativo claro frente a su referencia; linea cerrada. |

## F - No transferido al leaderboard (8)

Mejora local o sonda pre-registrada que no transfirio al leaderboard.

| experiment_id | f1 | decision | hipotesis (resumen) | nota |
|---|---|---|---|---|
| OPT-01 | 0.8012 | adoptada: el mejor trial cumple la regla pre-registrada; se reajusta c | Una busqueda TPE acotada alrededor del ganador (n-gramas, min_df/max_df, sublinear_tf, C y class_weight) puede | Mejor trial de Optuna (0.8012 historico) adoptado localmente; su sonda S05 no transfirio |
| R1-C1 | 0.7865 | sonda pre-registrada D-013 (maximo 2 envios): se adopta como modelo fi | Con C=1.0 el modelo memoriza menos el train (gap train-OOF menor) y la F1 OOF se mantiene dentro de 0.005 de M | Sonda pre-registrada D-013 (C=1.0): OOF 0.7929, LB 0.77227 |
| R2-MINDF5 | 0.7953 | sonda pre-registrada D-013 (maximo 2 envios): se adopta como modelo fi | min_df=5 descarta n-gramas que solo aparecen en train y reduce el gap train-OOF mas que C=1.0, con una perdida | Sonda pre-registrada D-013 (min_df=5): OOF 0.7993, LB 0.77034 |
| S04 | 0.7985 | DESCARTADA: el ensemble no se traslada al LB (0.77510 < 0.78306 de v03 | El promedio de M2 (LR) y M3 (LinearSVC) sobre la union word+char aporta diversidad que reduce el sesgo de cada | Ensemble ENS-04 con umbrales medianos: LB 0.77510; el umbral fue el sospechoso confirmado |
| S05 | 0.8039 | DESCARTADA: OPT-01 no transfiere (0.77602 < 0.78306 de v03). El gap -0 | El mejor trial de Optuna, reajustado con todo el train y con umbrales por etiqueta, mejora el LB frente a v03  | OPT-01 (Optuna): LB 0.77602 |
| S06 | 0.7929 | DESCARTADA: LB 0.77227 < 0.78306 de v03. El gap train-OOF no predice e | La sonda R1-C1 transfiere mejor al test que v03 (0.78306) porque su menor capacidad reduce el gap train-OOF | Sonda de robustez C=1.0: LB 0.77227 |
| S07 | 0.7993 | DESCARTADA: LB 0.77034 < 0.78306 de v03. El gap train-OOF no predice e | La sonda R2-MINDF5 transfiere mejor al test que v03 (0.78306) porque su menor capacidad reduce el gap train-OO | Sonda de robustez min_df=5: LB 0.77034 |
| S08 | 0.8465 | NO ADOPTADA en LB (0.78163 < 0.78306): la senal del meta no transfiere | El meta v2 mantiene en el holdout la mejora medida en el desarrollo y supera el LB de v03 (0.78306) | Meta v08: holdout 0.8465 (delta +0.0180) pero LB 0.78163 < 0.78306; no transfiere (D-020) |

## G - Trazabilidad / sin modelo nuevo (15)

Referencia, control, artefacto de auditoria o protocolo; sin modelo nuevo.

| experiment_id | f1 | decision | hipotesis (resumen) | nota |
|---|---|---|---|---|
| AUD-01 | 0.7772 | artefacto de auditoria; sin modelo nuevo | Los errores de las tres etiquetas son un fenomeno prioritario y de baja frecuencia (FASE 43/44) | Artefacto de auditoria: 101/1731 tweets fallan las 3 etiquetas (5.83 %) |
| AUD-02 | 0.7772 | artefacto de auditoria; sin modelo nuevo | Cada combinacion verdadera tiene un perfil estructural propio que explica el error por etiqueta (FASE 47) | Artefacto de auditoria por combinacion verdadera (FASE 47) |
| AUD-03 | 0.7874 | artefacto de auditoria; sin modelo nuevo | Hay canales con F1 menor pero complementarios: acuerdo, correlacion de errores y correccion mutua lo cuantific | Tabla de complementariedad entre 29 canales OOF (FASE 50/51) |
| B1 | 0.7587 | Pendiente de comparacion con B2-B5 y con la sonda E00 | TF-IDF de palabras + regresion logistica balanceada supera la plantilla todo-1 | Baseline TF-IDF word(1,2)+LR: 0.7578 local / 0.75677 LB |
| DL-01-SEEDS | 0.7947 | estabilidad entre semillas 0.0011 (desviacion); supera a MOD-03b_BEST  | Con 1.731 ejemplos la varianza entre semillas es apreciable (del orden de 0.01 en F1) y el nivel absoluto no a | Clasificacion automatica por el texto de la decision (fila secundaria) |
| E00 | 0.5149 | ADOPTADO: la metrica primaria local es f1_media_columnas; macro-6, mic | La metrica es la media column-wise de F1 binaria (clase positiva) | Sonda de metrica: LB 0.51494 con la plantilla todo-1 (identifica F1 medio por columna) |
| E2-14-MEJOR-M2 | 0.7784 | CERRADA sin mejora | La calibracion mejora la F1 y la estabilidad de umbrales (E2-14) | Clasificacion automatica por el texto de la decision (fila secundaria) |
| HOLD-01 | - | Congelado; maximo 5 evaluaciones al holdout, solo finalistas | Un holdout agrupado y estratificado congelado controla la multiplicidad | Holdout congelado de 248 tweets; maximo 5 evaluaciones (4 usadas) |
| PA-01 | 0.7838 | Cierre de la brecha de D-004 detectada en la auditoria final | El control por pseudo-autor reduce el optimismo del split por near-duplicados | Clasificacion automatica por el texto de la decision (fila secundaria) |
| PRE-00 | 0.7587 | referencia | Alguna normalizacion explicita del ruido de Twitter supera el texto crudo (H6, H7) | Referencia de la familia de preprocesamiento (texto crudo, 0.7587) |
| S02 | 0.7927 | Control superado por v03; el umbral 0.5 queda como referencia. Gap loc | El refit con todo el train reproduce la F1 local agrupada y mejora el LB frente a la submission v01 | Refit con todo el train: reproduce el OOF agrupado (control) |
| V2-BASE-M1_word_lr | 0.7486 | Baseline del protocolo v2 | El protocolo corregido (near-dup + pseudo-autor) da la referencia honesta | Referencia, control, artefacto de auditoria o protocolo; sin modelo nuevo. |
| V2-BASE-M2_union_lr_c2 | 0.7718 | Baseline del protocolo v2 | El protocolo corregido (near-dup + pseudo-autor) da la referencia honesta | Referencia, control, artefacto de auditoria o protocolo; sin modelo nuevo. |
| V2-BASE-M3_union_svc | 0.7677 | Baseline del protocolo v2 | El protocolo corregido (near-dup + pseudo-autor) da la referencia honesta | Referencia, control, artefacto de auditoria o protocolo; sin modelo nuevo. |
| V2-BASE-M4_char_lr | 0.7377 | Baseline del protocolo v2 | El protocolo corregido (near-dup + pseudo-autor) da la referencia honesta | Referencia, control, artefacto de auditoria o protocolo; sin modelo nuevo. |

## Tabla de canales con evidencia real

Canales puros (sin la representacion de V03) del store OOF corregido;
F1 standalone a umbral 0.5 sobre el desarrollo (n=1483) y metricas de
complementariedad frente a V03 (M2) de `reports/model_complementarity_v2.csv`.

| canal | familia | F1 standalone (0.5) | F1 HS | F1 TR | F1 AG | desacuerdo vs V03 | correlacion de errores | correccion mutua (V03 correcto si canal falla) | decision |
|---|---|---|---|---|---|---|---|---|---|
| EMO-ALL@C=1.0 | emocion | 0.6230 | 0.6469 | 0.6265 | 0.5957 | 0.2612 | 0.3010 | 0.6547 | canal puro usado por los candidatos FIN |
| SYN-3 | sintaxis | 0.5865 | 0.5734 | 0.6131 | 0.5729 | 0.2767 | 0.2888 | 0.6695 | canal puro usado por los candidatos FIN |
| CH-TGT+DIST | targeting | 0.6484 | 0.6505 | 0.6727 | 0.6221 | 0.2250 | 0.3732 | 0.6007 | canal puro usado por los candidatos FIN |
| CITA | cita_negacion | 0.7437 | 0.7218 | 0.7840 | 0.7253 | 0.1059 | 0.6455 | 0.3329 | canal puro usado por los candidatos FIN |

## Lectura del screening

1. La unica mejora adoptada con evidencia del protocolo corregido sigue
   siendo el meta de dependencia (E2-02-META, D-017) y su version Ridge
   de 44 columnas (E2-18-META-BASES-RIDGE, D-019). El control `FIN-01-`
   reproduce 0.7874 exacto, lo que valida la comparabilidad de esta fase.
2. Todos los canales linguisticos nuevos (emocion, sintaxis, cita/negacion,
   ortografia, NB-SVM) son negativos como modelo standalone (categoria E):
   su F1 aislado va de 0.527 (NEG) a 0.766 (TXT+NEG) frente a 0.7772 de V03.
3. Los canales aportan diversidad (desacuerdo 10-36 %, correlacion de errores
   0.15-0.65) pero su fusion directa con V03 por media de probabilidades es
   negativa; solo la fusion aprendida (meta) recupera senal, y anadir canales
   al meta de 44 columnas DEGRADA el nivel (-0.0092, `FIN-06-`).
4. Las sondas de leaderboard (categoria F) muestran un plateau: 0.77034-0.78306
   en 8 envios, un rango de ~1 error estandar (n=1154). El holdout congelado
   favorecio al meta v08 (+0.0180) y aun asi el LB no lo confirmo (D-020).

## Honestidad sobre los limites

- Las filas `FIN-0*` comparten el mismo protocolo corregido; su diferencia
  con la referencia V03 (maximo +0.0037 de `FIN-02-C-EMO`) esta por debajo
  del umbral de adopcion (+0.005 y >= 3/5 pliegues).
- Los contrastes de esta fase son 6 candidatos x 6 configuraciones; el
  maximo observado esta sesgado al alza por seleccion (winner's curse).
- Ninguna cifra de este informe es una estimacion del LB: el unico contraste
  pareado fiable con el LB es S03 (+0.0031) y el fracaso de las sondas v04-v08.
