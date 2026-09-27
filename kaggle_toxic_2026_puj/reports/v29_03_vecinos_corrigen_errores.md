# V29-03: los vecinos de similitud, pueden corregir errores de V03?

Generado por `experiments/v29_03_vecinos_corrigen_errores.py` (V29-03 (los vecinos de similitud corrigen errores de V03 o solo los estorban?)).

**Estado `OOF_EVALUATED`. Alcance: selector de vecino con una sola regla y sin meta-modelo. La regla se elige dentro del training fold por CV interna, de modo que el OOF reportado no se usa para elegirla**

## 1. Reglas anti-leakage

| regla | se cumple |
|---|:--:|
| validacion consulta solo a training | si |
| etiquetas de validacion no usadas | si |
| senales calculadas solo con etiquetas de training | si |
| vectorizador ajustado dentro de cada training fold | si |
| grupos construidos con el training fold | si |
| regla elegida en cv interna del training fold | si |

**Fuga declarada:** dentro de la CV interna del training fold, las filas internas de validacion ya formaban parte del texto usado para ajustar el vectorizador. No toca etiquetas, pero es una fuga transductiva y se declara en vez de ocultarse

Rejilla: 240 configuraciones por pliegue, elegidas dentro del training fold por 4 pliegues internos.

## 2. Controles

| control | valor |
|---|---|
| similitud consigo mismo, minimo | 1.0 |
| simetria, asimetria maxima | 0.0 |
| rango de similitud | [0.0, 1.0] |
| vecinos efectivos en la diagonal, minimo | 1.0 |
| la senal de uno mismo reproduce su etiqueta | 1.0 |

**la senal de vecino es lo bastante fuerte como para reproducir la etiqueta de la propia fila, exactamente en 1,0000. Por eso el aislamiento depende por completo de que la validacion solo consulte al training fold, y no de una limitacion del codigo. Un predictor de vecino sin folds seria fuga total**

## 3. La regla elegida en cada pliegue

| pliegue | metodo | k | umbral | tau | macro interna | celdas cambiadas |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `B_coseno_char4_agresivo` | 10 | 0.60 | 1.00 | 0.7729 | 9 |
| 1 | `D_jaccard_char35_agresivo` | 10 | 0.60 | 1.00 | 0.7729 | 26 |
| 2 | `E_jaccard_tokens_agresivo` | 10 | 0.60 | 1.00 | 0.7771 | 17 |
| 3 | `E_jaccard_tokens_agresivo` | 3 | 0.60 | 0.75 | 0.7873 | 26 |
| 4 | `E_jaccard_tokens_agresivo` | 10 | 0.60 | 0.90 | 0.7771 | 24 |

## 4. La tabla de mecanismos

| mecanismo | cobertura | rescates | danos | net | delta OOF |
|---|---:|---:|---:|---:|---:|
| V03 | - | - | - | - | 0.000000 |
| vecinos_por_similitud_anidado | 0.001124 | 0.000000 | 5.000000 | -5.000000 | -0.001522 |

**corrected_cv V03 0.777211 | con vecino 0.775689 | delta -0.001522**

| label | F1 V03 | F1 con vecino | delta | rescates | danos | net | celdas |
|---|---:|---:|---:|---:|---:|---:|---:|
| HS | 0.7617 | 0.7611 | -0.0006 | 0 | 1 | -1 | 1 |
| TR | 0.8157 | 0.8157 | +0.0000 | 0 | 0 | +0 | 0 |
| AG | 0.7542 | 0.7502 | -0.0040 | 0 | 4 | -4 | 4 |

## 5. Rescates y danos por franja de similitud del vecino

| franja | celdas | rescates | danos | net |
|---|---:|---:|---:|---:|
| 0.90_1.01 | 45 | 0 | 0 | +0 |
| 0.50_0.70 | 42 | 0 | 0 | +0 |
| 0.70_0.90 | 15 | 0 | 2 | -2 |
| 0.30_0.50 | 0 | 0 | 0 | +0 |

## 6. Barrido exhaustivo: TODA configuracion, sin selector

El selector de la seccion 4 responde "que es lo mejor que puedo hacer con una regla
elegida sin ver la validacion". Esa no es la pregunta de la Fase 5, que es "hay algun
vecino que corrija un error de V03". Aqui se aplica la mayoria simple en las
48 configuraciones, sin elegir ninguna,
y se cuentan los rescates y los danos.

Las diez primeras por net, y las cinco ultimas:

| metodo | k | umbral | cobertura | rescates | danos | net | delta OOF | acierto en lo que cambia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

| `D_jaccard_char35_agr` | 1 | 0.60 | 0.0018 | 0 | 8 | -8 | -0.001850 | 0.0000 |
| `D_jaccard_char35_agr` | 1 | 0.50 | 0.0029 | 1 | 12 | -11 | -0.002389 | 0.0769 |
| `E_jaccard_tokens_agr` | 5 | 0.60 | 0.0038 | 3 | 14 | -11 | -0.002160 | 0.1765 |
| `E_jaccard_tokens_agr` | 10 | 0.60 | 0.0045 | 4 | 16 | -12 | -0.002350 | 0.2000 |
| `D_jaccard_char35_agr` | 3 | 0.60 | 0.0034 | 1 | 14 | -13 | -0.002970 | 0.0667 |
| `D_jaccard_char35_agr` | 5 | 0.60 | 0.0031 | 0 | 14 | -14 | -0.003281 | 0.0000 |
| `D_jaccard_char35_agr` | 10 | 0.60 | 0.0031 | 0 | 14 | -14 | -0.003281 | 0.0000 |
| `E_jaccard_tokens_agr` | 1 | 0.60 | 0.0036 | 1 | 15 | -14 | -0.003428 | 0.0625 |
| `E_jaccard_tokens_agr` | 3 | 0.60 | 0.0036 | 1 | 15 | -14 | -0.003428 | 0.0625 |
| `B_coseno_char4_agres` | 1 | 0.60 | 0.0049 | 3 | 19 | -16 | -0.003520 | 0.1364 |
| ... | | | | | | | | |
| `E_jaccard_tokens_agr` | 10 | 0.30 | 0.0524 | 48 | 185 | -137 | -0.027623 | 0.2060 |
| `B_coseno_char4_agres` | 1 | 0.30 | 0.0665 | 55 | 241 | -186 | -0.036035 | 0.1858 |
| `B_coseno_char4_agres` | 5 | 0.30 | 0.0697 | 51 | 259 | -208 | -0.041631 | 0.1645 |
| `B_coseno_char4_agres` | 3 | 0.30 | 0.0717 | 47 | 272 | -225 | -0.045438 | 0.1473 |
| `B_coseno_char4_agres` | 10 | 0.30 | 0.0764 | 53 | 287 | -234 | -0.047217 | 0.1559 |

**Configuraciones que producen al menos un rescate: 45 de 48.**

## 7. El detalle de la Fase 5 sobre las filas donde V03 falla

estas filas son celdas, no filas: una fila aporta hasta tres celdas, una por label.

| magnitud | valor |
|---|---:|
| filas con algun error de V03 | 441 |
| de ellas, con vecino por encima del umbral | 10 (0.0227) |
| celdas de error donde el vecino trae la etiqueta correcta | 3 |
| celdas de error donde el vecino contradice a V03 | 3 |
| celdas correctas donde el vecino contradice a V03 | 17 |

## 8. Lo que este script no hace

- No genera submission.
- No entrena ningun modelo: la regla es un umbral de confianza.
- No usa el test.
- No consulta etiquetas de validacion para nada.

Coste: 41.0 s, memoria 266 MB, 1 predictor auxiliar, 0 submissions.

