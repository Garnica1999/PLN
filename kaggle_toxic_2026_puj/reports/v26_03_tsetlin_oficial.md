# V26-03: Tsetlin Machine con pyTsetlinMachine, sobre word12 binario

Generado por `experiments/v26_03_tsetlin_oficial.py` (V26-03 (Tsetlin Machine con pyTsetlinMachine publicado, sobre word12 binario)).

**Ninguna submission.**

## 1. La implementacion usada

| campo | valor |
|---|---|
| libreria | `pyTsetlinMachine` |
| version | 0.6.6 |
| licencia | MIT |
| commit | `d6c1cf0e4aaa4a8ae2f2818ba27878fb89d31dc5` (2025-04-01) |
| repositorio | https://github.com/cair/pyTsetlinMachine/ |
| clase | `MultiClassTsetlinMachine` |
| python | 3.14.7 |
| numpy | 1.26.4 |
| plataforma | Linux x86_64 |
| modificaciones al codigo de la libreria | ninguna |

el makefile del proyecto usa -Werror y con gcc moderno falla por variables sin usar. Se recompilo con los mismos flags de optimizacion quitando -Werror, que es un cambio de build y no de codigo fuente.

**Por que es legal**: las reglas de la competicion prohiben transformers y LLM, y exigen que los modelos de vectorizacion se entrenen desde cero. pyTsetlinMachine es un modelo de reconocimiento de patrones entrenado desde cero, sin datos externos ni pesos preentrenados, luego cumple.

La implementacion casera de `v26_02_tsetlin.py` queda **REJECTED** y sus resultados no
son evidencia del rendimiento del algoritmo.

## 2. Smoke test canonico

| magnitud | valor |
|---|---:|
| exactitud en XOR de train | 1.0000 |
| F1 clase 0 | 2.0000 |
| F1 clase 1 | 2.0000 |
| segundos | 0.052 |
| dos ajustes consecutivos identicos | True |

pyTsetlinMachine no expone random_state: usa fast_rand() de la libreria C. Dos ajustes consecutivos en el mismo proceso pueden diferir, y esa irreproducibilidad se reporta en vez de ocultarla.

## 3. El control lineal sobre la MISMA matriz binaria

Benchmark historico de referencia: **0.751393**.

| modelo | corrected_cv | dOOF vs V03 | train F1 | gap | rescates | danos | neto |
|---|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression(binario) | 0.748064 | -0.029147 | 0.9903 | +0.2422 | 119 | 208 | -89 |
| V03 (referencia) | 0.7772114525 | +0.0000000000 | -- | -- | 0 | 0 | +0 |

La diferencia entre el control y V03 es el **coste de la binarizacion**. La diferencia
entre el Tsetlin y el control es la **ganancia del mecanismo**, que es la unica que
responde a la pregunta de la familia.

## 4. El Tsetlin

| K | corrected_cv | dOOF vs control | dOOF vs V03 | train F1 | gap | sd pliegues | rescates | danos | neto | segundos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.583397 | -0.164668 | -0.193815 | 0.6575 | +0.0741 | 0.0270 | 289 | 647 | -358 | 146.5 |

**Corte en K=200**: no mejora al control (-0.164668), neto de celdas no positivo (-358)

## 5. Complementariedad con V03

| K | coincidencias | fraccion | rescates | danos | ambos malos | correlacion |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 3513 de 4449 | 0.7896 | 289 | 647 | 468 | 0.5513 |

## 6. F1 por etiqueta y estabilidad entre pliegues

| K | etiqueta | F1 TM | F1 control | F1 V03 | rescates | danos |
|---:|---|---:|---:|---:|---:|---:|
| 200 | HS | 0.5754 | 0.7285 | 0.7617 | 133 | 245 |
| 200 | TR | 0.6131 | 0.8147 | 0.8157 | 61 | 186 |
| 200 | AG | 0.5617 | 0.7010 | 0.7542 | 95 | 216 |

| K | F1 por pliegue | sd |
|---:|---|---:|
| 200 | 0.5669 | 0.5802 | 0.6105 | 0.6123 | 0.5409 | 0.0270 |

## 7. Reproducibilidad

- K=200: primera 0.583397, repetida 0.580404, diferencia 0.002993.
- Mismo numero de cambios: False. Mismas predicciones: False.
- Causa: pyTsetlinMachine usa fast_rand() de la libreria C y no expone random_state.

## 8. Coste

- Tiempo total: 60.8 s. Memoria maxima: 316 MB. Submissions: 0.

