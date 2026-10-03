# P51 - S2-U, accion directa con utilidad esperada

Generado por `experiments/v51_02_accion_utilidad_esperada.py` (V51-02 (S2-U, accion directa con utilidad esperada)).

Fecha 2026-09-28T01:45:48+00:00. **0 submissions, 0 consultas a Kaggle.**

> S1-U demostro senal observable (AUC medio 0,8091) con delta negativo, que es la condicion que la directiva fija para autorizar S2-U

## 1. La regla

    E[u(i,j)] = P_j(x) * g_dir - (1 - P_j(x)) * h_dir, elegir argmax y cambiar solo si es mayor que cero. Sin umbral, luego sin grid

Los umbrales de equilibrio salen de la propia regla: `h/(g+h)` subiendo y `g/(g+h)` bajando. No hay nada que ajustar.

## 2. La senal observable que justifica S2-U

| AUC out-of-fold | valor |
|---|---:|
| minimo | 0.5507 |
| maximo | 0.9249 |
| **media** | **0.8091** |

Para comparar, P50upidio entre 0,55 y 0,77 con el target de correccion. El objetivo de utilidad es mucho mas aprendible, y eso es un hallazgo de P51 con independencia de si la decisionFinal sale buena.

## 3. Resultado

| politica | corrected_cv | delta vs V03 | cambiadas | rescate | damage | net |
|---|---:|---:|---:|---:|---:|---:|
| V03 | 0.7772114525 | +0,0000000000 | 0 | 0 | 0 | 0 |
| P50 S1 (`channel_correct`) | 0.7657442265 | -0.0114672260 | - | 48 | 67 | -19 |
| P51 S1-U (binario) | 0.0294622851 | -0.7477491674 | - | 414 | 1330 | -916 |
| **P51 S2-U (utilidad esperada)** | **0.7670528094** | **-0.0101586431** | **304** | **121** | **183** | **-62** |
| oracle (techo, usa `y_true`) | 0.9870178614 | +0.2098064089 | - | 715 | 0 | +715 |

Tasa de abstencion: 93.17 pct.

## 4. Pliegues

| pliegue | F1 de V03 | F1 de S2-U | delta |
|---:|---:|---:|---:|
| 0 | 0.7960101143 | 0.7651145315 | -0.0308955828 |
| 1 | 0.7940339498 | 0.7709025496 | -0.0231314002 |
| 2 | 0.7788600289 | 0.7780568277 | -0.0008032012 |
| 3 | 0.7385980131 | 0.7567896316 | +0.0181916185 |
| 4 | 0.7790564997 | 0.7632606460 | -0.0157958537 |

Pliegues positivos: **1 de 5**.

## 5. Interseccion de errores

| categoria | celdas |
|---|---:|
| ambos_correctos | 3509 |
| ambos_incorrectos | 636 |
| v03_mal_selector_bien | 121 |
| v03_bien_selector_mal | 183 |

## 6. Por margen y por direccion

| corte | tipo | n | rescate | damage | net |
|---|---|---:|---:|---:|---:|
| [0,0.10) | margen | 222 | 99 | 123 | -24 |
| [0.10,0.20) | margen | 70 | 19 | 51 | -32 |
| [0.20,1] | margen | 12 | 3 | 9 | -6 |
| sube_0_a_1 | direccion | 209 | 71 | 138 | -67 |
| baja_1_to_0 | direccion | 95 | 50 | 45 | +5 |

## 7. Permutacion del net de celdas

> el gate no pasa, luego la directiva no pide bootstrap. Un bootstrap sobre las celdas cambiadas seria degenerado porque al reasignar la prediccion del selector a la celda muestreada el valor no cambia y el intervalo sale de ancho cero. La permutacion si responde a la pregunta util: si el net de celdas se distingue de cero

| magnitud | valor |
|---|---:|
| net observado | -62 celdas |
| mediana de la nula | -115.0 |
| IC95 de la nula | [-136.0, -93.0] |
| p de una cola, mejor que el azar | 0.0000 || celdas mejor que el azar | +53 |
| el net observado cae dentro de la nula | False |
| replicas | 2000 |

## 7 bis. Variante restringida a la direccion 1->0

> consecuencia directa del reparto por direccion medido, no un grid: la direccion 1->0 sale neta positiva y la 0->1 muy negativa

| magnitud | valor |
|---|---:|
| corrected_cv | 0.7722071668 |
| delta vs V03 | -0.0050042857 |
| celdas cambiadas | 95 |
| rescate / damage / net | 50 / 45 / +5 |

## 8. Gate y decision

- delta -0.010159 frente a 0.003 exigido
- pliegues positivos 1 de 4 exigidos
- gate pasado: **False**

**CASE = A.**

