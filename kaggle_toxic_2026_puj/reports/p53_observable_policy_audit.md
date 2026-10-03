# P53 - Auditoria de politica observable de grano fino

Generado por `experiments/v53_00_politica_fina.py` (V53-00 (auditoria de politica observable de grano fino)).

Fecha 2026-09-28T02:22:22+00:00. **0 modelos, 0 submissions, 0 busqueda de umbrales, 0 consultas a Kaggle.**

## P52 CORRECTIONS

### Que afirmaciones se MANTIENEN

- El `STRATUM_ORACLE` de P52 esta bien calculado. Verificado leyendo el codigo: los estratos son v03_pred x bin de margen x bin de numero de discrepancias.
- El numero `78` de celdas con rescate unico se mantiene: P53 lo recalcula y obtiene **78**, coincidencia True.
- El `TRUE_ORACLE` de 0.9870178614 se reconstruye con el colapso de dos acciones y coincide con P50.

### Que afirmaciones se CORRIGEN

- **P52 no estaba contaminado**, pero la pregunta estaba mal planteada. Elegir la accion con `y_true` es justamente lo que hace que sea un **techo**: un techo holgado por construccion. Si un techo holgado cae por debajo de V03, todo lo mas cenido tambien cae, luego la conclusion se sostiene.
- **La frase `0 pares contradictorios` era ambigua pero su numero era correcto.** P53 lo descompone en A, B, C y D y obtiene B = 0 sobre los 190 pares. Y no es casualidad: es una identidad estructural.

### Que metricas estaban MAL DEFINIDAS

- **El HHI de P52 estaba mal definido.** Se computo como `sum((exclusivo_i / 715)^2)`, con proporciones que **no suman 1** porque el denominador 715 es el total de rescates y no la suma de los exclusivos. P53 reporta dos metricas separadas: el **HHI estandar**, que reparte la masa solo entre los exclusivos, y la **cuota de exclusivos sobre los 715**.

## 1. El colapso estructural, que es el hallazgo central

| magnitud | valor |
|---|---:|
| celdas con discrepancia | 3896 |
| celdas donde el valor propuesto **no** es `1 - v03_pred` | **0** |
| celdas donde dos canales discrepantes se contradicen | **0** |
| **espacio de acciones real** | **2** |

> cambiar al canal 3 o al canal 17 produce la misma prediccion, luego la mascara no puede portar informacion sobre el valor propuesto; solo las probabilidades pueden decir cuan fuerte empuja cada canal

Esto deja el problema entero reducido a **una decision binaria por celda: voltear o no**. Y explica de paso los `0 pares` de P52 sin necesidad de que sea una coincidencia.

## 2. Distribucion de las mascaras

| magnitud | valor |
|---|---:|
| mascaras distintas sobre las 4449 celdas | 1127 |
| mascaras distintas entre las que discrepan | 1126 |
| patrones en celdas de error de V03 | 1126 |
| frecuencia mediana | 1 |
| frecuencia maxima | 658 |
| patrones singleton | 843 |

## 3. Rescate y dano por patron, y simple frente a conjunta

| tipo | n | tasa de rescate | tasa de dano | net |
|---|---:|---:|---:|---:|
| un_solo_canal | 959 | 8.13 pct | 91.87 pct | -803 |
| dos_o_mas | 2937 | 21.69 pct | 78.31 pct | -1663 |

## 4. HHI estandar y metricas de exclusividad

| magnitud | valor |
|---|---:|
| celdas con rescate unico | 78 de 715 |
| coincide con P52 | True |
| suma de rescates exclusivos | 78 |
| **HHI estandar de los exclusivos** | 0.360289 |
| cuota de exclusivos sobre los 715 | 10.91 pct |
| canales con `net_exclusivo` positivo | 2 de 20 |

## 5. Los 190 pares, descompuestos

| magnitud | valor |
|---|---:|
| pares analizados | 190 |
| **B: ambos discrepan y se contradicen** | **0** |
| pares con B > 0 | 0 |
| jaccard de rescates, minimo | 0.0462 |
| jaccard de rescates, maximo | 1.0000 |
| jaccard de rescates, media | 0.2504 |

| canal i | canal j | A | B | C | D | i solo | j solo | jaccard |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `synt:SYN-POS` | `cita:NEG` | 664 | 0 | 2375 | 1410 | 199 | 227 | 0.2052 |
| `synt:SYN-DEPS` | `cita:NEG` | 668 | 0 | 2423 | 1358 | 197 | 232 | 0.1966 |
| `cita:NEG` | `emoc:EMO-EST@C=2.0` | 578 | 0 | 2399 | 1472 | 216 | 179 | 0.2345 |
| `synt:SYN-3` | `cita:NEG` | 640 | 0 | 2331 | 1478 | 177 | 219 | 0.2296 |
| `synt:SYN-MORPH` | `cita:NEG` | 803 | 0 | 2205 | 1441 | 172 | 212 | 0.2456 |
| `cita:NEG` | `emoc:EMO-ALL@C=1.0` | 597 | 0 | 2348 | 1504 | 200 | 168 | 0.2713 |
| `cita:NEG` | `emoc:EMO-LEX@C=1.0` | 658 | 0 | 2269 | 1522 | 197 | 166 | 0.2783 |
| `cita:NEG` | `emoc:NB-UNION-SVC@C=1.0` | 190 | 0 | 2491 | 1768 | 263 | 143 | 0.1542 |

## 6. Techos cross-fitted de P0 a P4

Todos son cross-fitted: la accion de cada estado se aprende con las labels del training fold y se aplica al validation fold, con repliegue `P4 -> P3 -> P2 -> P1 -> V03`.

| nivel | corrected_cv | delta vs V03 | cambiadas | net | estados | cobertura |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.7772114525 | +0.0000000000 | 0 | +0 | 1 | 100.0 pct |
| P1 | 0.7623696498 | -0.0148418027 | 146 | -16 | 18 | 100.0 pct |
| P2 | 0.7566874656 | -0.0205239869 | 113 | -61 | 1469 | 70.8 pct |
| P3 | 0.7588669459 | -0.0183445066 | 97 | -57 | 1877 | 64.7 pct |
| P4 | 0.7712584603 | -0.0059529922 | 23 | -19 | 2940 | 36.4 pct |

| soporte minimo en training | corrected_cv | delta | cambiadas | net |
|---|---:|---:|---:|---:|
| P2 soporte >= 5 | 0.7761995598 | -0.0010118927 | 11 | -3 |
| P2 soporte >= 10 | 0.7760870884 | -0.0011243641 | 3 | -3 |
| P2 soporte >= 20 | 0.7772114525 | +0.0000000000 | 0 | +0 |
| P3 soporte >= 5 | 0.7772114525 | +0.0000000000 | 0 | +0 |
| P3 soporte >= 10 | 0.7772114525 | +0.0000000000 | 0 | +0 |
| P3 soporte >= 20 | 0.7772114525 | +0.0000000000 | 0 | +0 |
| P4 soporte >= 5 | 0.7772114525 | +0.0000000000 | 0 | +0 |
| P4 soporte >= 10 | 0.7772114525 | +0.0000000000 | 0 | +0 |
| P4 soporte >= 20 | 0.7772114525 | +0.0000000000 | 0 | +0 |

## 7. La cadena de techos

| politica | corrected_cv | delta vs V03 |
|---|---:|---:|
| V03 | 0.7772114525 | +0,0000000000 |
| P52 STRATUM_ORACLE (18 estratos) | 0.7753804469 | -0.0018310056 |
| **P53 mejor techo observable** | **0.7772114525** | **+0.0000000000** |
| TRUE_ORACLE | 0.9870178614 | +0.2098064089 |

## 8. Decision

- el techo observable mas fino supera V03 + 0,003: **False**
- `CASE = C`

**Alcance de la conclusion.** Lo que la evidencia sostiene es: *ninguna politica basada en la mascara de desacuerdo, el margen, el numero de discrepancias, la identidad de la etiqueta y las probabilidades de los canales supera V03 en evaluacion cross-fitted*. Eso es mas de lo que P52 podia afirmar, porque el espacio de acciones ha quedado agotado a dos acciones y la identidad de la mascara esta incluida. Lo que **no** se afirma es que ninguna politica imaginable supere V03.

