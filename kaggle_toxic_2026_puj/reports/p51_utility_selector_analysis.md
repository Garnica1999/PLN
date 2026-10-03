# P51 - Analisis del selector utilitario

Generado por `experiments/v51_01_selector_utilidad.py` (V51-01 (selector utilitario S1-U)).

Fecha 2026-09-28T01:41:53+00:00. **0 submissions, 0 consultas a Kaggle.**

Target: `delta_F1 > 0, con los conteos del pliegue de entrenamiento`.

## 1. El objetivo cambia respecto a P50

| fase | target | lo que premia |
|---|---|---|
| P50 S1 | `channel_correct` | la **predecibilidad** de un canal |
| P51 S1-U | `delta_F1 > 0` | la **utilidad** del cambio |

## 2. El equilibrio no es 0,5

| magnitud | valor |
|---|---:|
| umbral de equilibrio para **subir** de 0 a 1, sobre P(y=1) | **0.3892** |
| umbral de equilibrio para **bajar** de 1 a 0, sobre P(y=1) | **0.3892** |
| los dos coinciden | True |
| tasa base al subir | 17.86 pct |
| tasa base al bajar | 37.61 pct |
| **elevacion necesaria al subir** | **2.18x** |
| **elevacion necesaria al bajar** | **1.03x** |

La consecuencia es que la regla utilitaria optima es un **unico umbral sobre P(y=1)** y no un 0,5 simetrico. Ese umbral sale de la forma cerrada, no de un grid.

## 3. Controles

| selector | corrected_cv | delta vs V03 | rescate | damage | net |
|---|---:|---:|---:|---:|---:|
| B0 V03 | 0.7772114525 | +0,0000000000 | 0 | 0 | 0 |
| B1 siempre KEEP_V03 | 0.7772114525 | +0,0000000000 | 0 | 0 | 0 |
| B2 P50 S1 | 0.7657442265 | -0.0114672260 | 48 | 67 | -19 |

B1 es identico a B0 por construccion: la accion por defecto ya es no cambiar.

## 4. S1-U

| regla | corrected_cv | delta vs V03 | delta vs P50 | cambiadas | rescate | damage | net |
|---|---:|---:|---:|---:|---:|---:|---:|
| umbral 0,5 (el de la directiva) | 0.0294622851 | -0.7477491674 | -0.7362819414 | 1744 | 414 | 1330 | -916 |
| umbral analitico | 0.7721856247 | -0.0050258278 | +0.0064413982 | 159 | 76 | 83 | -7 |

AUC out-of-fold de los 20 clasificadores: min 0.5507, max 0.9249, media 0.8091.

## 5. Pliegues

| pliegue | F1 de V03 | F1 de S1-U | delta |
|---:|---:|---:|---:|
| 0 | 0.7960101143 | 0.0372418761 | -0.7587682382 |
| 1 | 0.7940339498 | 0.0237511671 | -0.7702827827 |
| 2 | 0.7788600289 | 0.0307454603 | -0.7481145686 |
| 3 | 0.7385980131 | 0.0361879666 | -0.7024100465 |
| 4 | 0.7790564997 | 0.0193479454 | -0.7597085543 |

Pliegues positivos: **0 de 5**.

## 6. Interseccion de errores

| categoria | celdas |
|---|---:|
| ambos_correctos | 2362 |
| ambos_incorrectos | 343 |
| v03_mal_selector_bien | 414 |
| v03_bien_selector_mal | 1330 |

## 7. Por margen y por direccion

| corte | tipo | n | rescate | damage | net |
|---|---|---:|---:|---:|---:|
| [0,0.10) | margen | 445 | 207 | 238 | -31 |
| [0.10,0.20) | margen | 281 | 88 | 193 | -105 |
| [0.20,1] | margen | 1018 | 119 | 899 | -780 |
| sube_0_a_1 | direccion | 61 | 24 | 37 | -13 |
| baja_1_a_0 | direccion | 1683 | 390 | 1293 | -903 |

## 8. Comparacion con P50 y con el techo

| politica | corrected_cv | delta vs V03 |
|---|---:|---:|
| V03 | 0.7772114525 | +0,0000000000 |
| P50 S1 (`channel_correct`) | 0.7657442265 | -0.0114672260 |
| **P51 S1-U (`delta_F1 > 0`)** | **0.0294622851** | **-0.7477491674** |
| oracle (techo, usa `y_true`) | 0.9870178614 | +0.2098064089 |

**El oracle no es un modelo alcanzable.** Usa `y_true` para decidir cada celda, luego es un techo y no un resultado.

## 9. Gate y decision

- delta obtenido -0.747749 frente a 0.003 exigido
- pliegues positivos 0 de 4 exigidos
- gate pasado: **False**

**CASE = A.**

