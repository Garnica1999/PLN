# P45 - Auditoria de la metrica

Generado por `experiments/v45_00_metrica_audit.py` (V45-00 (P0 auditoria de la metrica de la competicion)).

Fecha 2026-09-27T23:49:14+00:00. **0 modelos, 0 submissions.**

## 1. Reproduccion de V03

| magnitud | valor |
|---|---:|
| corrected_cv calculado | 0.7772114525165721 |
| corrected_cv objetivo | 0.7772114525165721 |
| diferencia | +0.000e+00 |
| F1 HS | 0.7616892911 |
| F1 TR | 0.8157303371 |
| F1 AG | 0.7542147294 |

Conteos por etiqueta:

| etiqueta | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| HS | 505 | 173 | 143 | 662 |
| TR | 363 | 82 | 82 | 956 |
| AG | 425 | 135 | 142 | 781 |

## 2. Agregacion: media de pliegues frente a OOF global

| pliegue | n | macro F1 |
|---|---:|---:|
| 1 | 297 | 0.7960101143 |
| 2 | 297 | 0.7940339498 |
| 3 | 296 | 0.7788600289 |
| 4 | 296 | 0.7385980131 |
| 5 | 297 | 0.7790564997 |
| **media** | | **0.7773117212** |
| **global OOF** | | **0.7772114525** |

Diferencia: +0.0001002686. No son intercambiables. El `corrected_cv` reportado en todo el proyecto es el **global**, no la media de pliegues.

## 3. Identidad con `f1_score(average='macro')`

| expresion | valor |
|---|---:|
| `f1_score(y, p, average='macro')` | 0.7772114525165721 |
| `f1_media_columnas` (metrica del proyecto) | 0.7772114525165721 |

Contraste con 600 casos aleatorios: **0 discrepancias**, maximo |diferencia| 0.000e+00.

f1_score(average='macro') sobre entrada multilabel promedia la F1 de la clase positiva de cada etiqueta, no las seis clases.

## 4. Las cuatro hipotesis del parametro `average`

| hipotesis | valor sobre el OOF de V03 |
|---|---:|
| `macro_3` **(la del proyecto)** | 0.7772114525 |
| `micro` | 0.7735566856 |
| `samples` | 0.3343897505 |
| `weighted` | 0.7736231399 |
| `macro_6_clases` (diagnostica) | 0.8182213363 |
| `muestra_por_fila_exact_match` (diagnostica) | 0.7026298045 |

Dispersion entre las cuatro candidatas: **0.4428217020**.

Scorer de Kaggle: metric/f1-score (kernelId 39244216). El parametro average esta sobrescrito por el host y no es legible.

Metrica del proyecto: macro_3 (f1_media_columnas)

## 5. Test de impacto: el ranking depende de `average`?

Se barrieron 1000 configuraciones de umbral sobre las probabilidades OOF de V03 (sin usar test ni holdout).

| comparacion | spearman | misma mejor config | mismo top-10 |
|---|---:|---|---|
| macro_3 vs `micro` | 0.983591 | False | False |
| macro_3 vs `samples` | 0.489635 | False | False |
| macro_3 vs `weighted` | 0.992105 | True | False |

Optimo de la rejilla bajo `macro_3`: HS=0.4 TR=0.5 AG=0.44. Umbrales congelados de V03: [0.44, 0.51, 0.46].

## 5b. Discriminante de escala: el host usa `samples`?

La hipotesis `samples` da 0.334 sobre el OOF de V03, contra 0.777 de `macro_3`. Si el host usara `samples`, el leaderboard de la submission construida con V03 deberia rondar ese valor. El log local registra 0.78306.

| magnitud | valor |
|---|---:|
| submissions con local y kaggle | 19 |
| pearson(local_f1, kaggle_f1) | 0.487565 |
| LB de V03 | 0.78306 |
| macro_3 sobre el OOF de V03 | 0.77721 |
| samples sobre el OOF de V03 | 0.33439 |
| brecha contra macro_3 | +0.00585 |
| brecha contra samples | +0.44867 |

**`samples` queda EXCLUIDO = True.** Una brecha de mas de 0.10 no se explica por un desplazamiento de prevalencia train->test del orden de los puntos. El argumento se sostiene sobre el valor absoluto que el host reporto, no sobre la correlacion.

El pearson(local_f1, kaggle_f1) es 0.4876, no alto. Eso no habilita `samples`, pero conviene dejarlo dicho: la transferencia local->leaderboard es ruidosa, y es coherente con el hecho historico de que 0 de 22 submissions superaron a V03 pese a metricas locales altas. El discriminante de escala no depende de esa correlacion.

## 6. Veredicto

**METRIC_MATCH = True**

Alcance exacto de esa afirmacion:

1. La metrica local `corrected_cv` **si** es exactamente `sklearn.metrics.f1_score(..., average='macro')` sobre entrada multilabel: diferencia 0.000e+00 y 0 discrepancias en 600 casos aleatorios.
2. `corrected_cv = 0.7772114525165721` **se reproduce** exactamente, con diferencia +0.000e+00.
3. `samples` **queda excluido** por el discriminante de escala del leaderboard.
4. Quedan `macro`, `micro` y `weighted`, separados por menos de 0.0036 entre si, y con ranking de umbrales practicamente identico (spearman >= 0.98).

Lo que la prueba SI SOSTIENE:

- La metrica que el proyecto ha optimizado es coherente con la del host bajo las tres hipotesis que sobreviven.
- Ninguna conclusion previa depende de resolver cual de las tres es.

Lo que la prueba NO SOSTIENE:

- Que `average='macro'` este confirmado de forma directa. El codigo del scorer del host no esta en el repositorio y el `average` que aplica no es legible.
- La distincion entre `macro`, `micro` y `weighted` queda sin resolver. Es inmaterial para el ranking de candidatos, pero es una resolucion pendiente.

