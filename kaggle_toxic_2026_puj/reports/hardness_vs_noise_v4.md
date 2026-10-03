# Dureza de ejemplos frente a ruido de etiqueta (v4, TRACK B)

Generado por `experiments/28_hardness_v4.py` el 2026-10-02. Todas las señales son **out-of-fold**: la probabilidad de V03 para un ejemplo procede de un modelo que no lo vio en entrenamiento (§9).

## 1. Variables calculadas

| Variable | Definición | Uso |
|---|---|---|
| `hardness_score` | cercania a la frontera de decision, `1 - |(p - umbral) * (2y - 1)|` (principal) | ponderacion de ejemplos (V4-B03) |
| `error_score` | magnitud media de la violacion del umbral | severidad del error |
| `bce_score` | entropia binaria cruzada de la etiqueta verdadera (descriptiva) | contraste de definiciones |
| `p_true` | probabilidad OOF asignada a la etiqueta verdadera | separacion facil/dificil |
| `error` | error OOF con los umbrales congelados de V03 | variable de control |
| `nn_disagreement` | el vecino mas cercano (cosimilitud TF-IDF) tiene otra etiqueta | ambiguedad local |
| `knn_disagreement` | desacuerdo ponderado con los tres vecinos | ambiguedad local suavizada |
| `model_disagreement` | fraccion de los canales OOF no redundantes que fallan en esa etiqueta | dificultad del ejemplo |
| `cluster_contradictorio` | pertenece a un cluster de texto repetido con etiquetas distintas | ruido textual |
| `label_rarity` | frecuencia de la etiqueta verdadera en train | rareza |

## 2. Distribucion de la dureza y eleccion de la definicion

- `hardness_score` (media): **0.7298**, mediana 0.7292, p90 0.8910.
- Ejemplos con al menos una etiqueta errada: **441** (29.74 %).
- Ejemplos que fallan las tres etiquetas: **98** (6.61 %).

**Correlaciones con el numero de etiquetas erradas** (Pearson):

| Medida | Correlacion | Lectura |
|---|---|---|
| `p_true` (probabilidad de la etiqueta verdadera) | -0.841 | menor confianza, mas error |
| `error_score` (violacion del umbral) | 0.811 | medida directa del error |
| `bce_score` (entropia binaria cruzada) | -0.136 | **anticorrelacionada** |
| `hardness_score` (cercania a la frontera) | 0.345 | correlacion positiva y monotona |

Hallazgo metodologico que condiciona V4-B03 (la spec §11 propone ponderar por `OOF_loss`): la dificultad **por etiqueta** si se ordena bien (`p_true` frente al error de esa etiqueta: r = -0.795 / -0.81 / -0.801), pero **agregar las tres etiquetas rompe el orden**. La BCE media por numero de errores es 0 errores -> 1.3408, 1 errores -> 1.1625, 2 errores -> 0.9974, 3 errores -> 0.5768: los ejemplos que mas fallan tienen la BCE **menor**, porque una violacion fuerte en una etiqueta se compensa con confianza alta en las otras dos. Por eso la correlacion de la BCE media con el numero de errores (-0.136) es mucho menor que la de la cercania a la frontera (0.345). Conclusion operativa: para ponderar ejemplos se usa la distancia al umbral en la direccion de la etiqueta verdadera (monotona en el error) y, si se quiere priorizar el error, `error_score`; la BCE agregada no debe usarse como peso en este corpus.

## 3. Dos poblaciones: dificil coherente frente a posible ruido

Umbral de corte: dureza >= percentil 80 = 0.8327.

| Poblacion | n | % | error medio por etiqueta | en cluster contradictorio |
|---|---|---|---|---|
| Dificil coherente | 296 | 20.0 % | 0.351 | 0.000 |
| Posible ruido | 43 | 2.9 % | 0.054 | 1.000 |

## 4. Error dentro y fuera de los clusters contradictorios

Tasa de error OOF de V03 y delta con IC95 de bootstrap pareado por tweet (2000 remuestreos):

| Etiqueta | Error dentro | Error fuera | Delta | IC95 |
|---|---|---|---|---|
| HS | 0.046 | 0.218 | -0.172 | [+0.000, +0.116] |
| TR | 0.023 | 0.113 | -0.090 | [+0.000, +0.079] |
| AG | 0.093 | 0.190 | -0.097 | [+0.022, +0.191] |

## 5. Desacuerdo entre modelos en las dos poblaciones

| Poblacion | `model_disagreement` medio | `nn_disagreement` medio |
|---|---|---|
| Dificil coherente | 0.414 | 0.479 |
| Posible ruido | 0.103 | 0.612 |

## 6. Los 15 ejemplos mas duros

| id | etiquetas | p_true (HS/TR/AG) | error | desacuerdo modelos | contradiccion | texto |
|---|---|---|---|---|---|---|
| 22330 | 111 | 0.99/1.00/0.99 | 0 | 0.03 | 0 | @booty_celli zorra CALLATE te mato |
| 21231 | 111 | 0.99/1.00/0.99 | 0 | 0.00 | 1 | @Hxrry_again CLLATE PERRA |
| 22142 | 111 | 0.98/1.00/0.99 | 0 | 0.10 | 0 | @ojvr__ Acho cllate joda perra |
| 23174 | 111 | 0.98/0.99/0.98 | 0 | 0.00 | 0 | @victorACDZ @yael_diaz Tu cllate alv perra envidiosa |
| 20609 | 111 | 0.98/0.99/0.99 | 0 | 0.05 | 1 | @ReSidentG1RL @BlendFreshon Cllate perra |
| 23411 | 111 | 0.98/0.99/0.98 | 0 | 0.00 | 1 | @juulivillalba Tu eres una zorra |
| 22982 | 111 | 0.98/0.99/0.98 | 0 | 0.00 | 1 | @jocksan51 @MemoTorresL Cllate perra |
| 22955 | 111 | 0.98/0.99/0.97 | 0 | 0.00 | 1 | @William43591799 Tu eres una perra |
| 24147 | 111 | 0.98/0.99/0.98 | 0 | 0.00 | 0 | @AnttoC_ callate si sos re zorra |
| 20347 | 111 | 0.98/0.99/0.98 | 0 | 0.00 | 1 | @DIOSAYT @unaineira_04 @A_S_B_L_A_C_K Cllate perra |
| 20461 | 111 | 0.98/0.99/0.98 | 0 | 0.03 | 0 | @medicengaribay Cllate zorra que l alcohlico aqu eres t. |
| 23927 | 111 | 0.98/0.99/0.98 | 0 | 0.00 | 0 | @TaquitoDeCecina Cllate perra trolls forevaa |
| 20937 | 111 | 0.98/0.99/0.97 | 0 | 0.00 | 0 | @lizbethsm19 Cllate perra si ahorita te tomaste una ultra |
| 20981 | 111 | 0.97/0.99/0.98 | 0 | 0.10 | 0 | PUTA ANA CLLATE ZORRA. |
| 23450 | 111 | 0.97/0.99/0.98 | 0 | 0.00 | 1 | @maritehgoldst @info7mty Cllate pinche perra |

## 7. Lectura (sin modificar etiquetas)

El 43 de 1483 ejemplos de desarrollo (2.9 %) pertenece a un cluster de texto repetido con etiquetas contradictorias y se clasifican como posible ruido; 1144 ejemplos son faciles. En las tres etiquetas la tasa de error es **menor** dentro de los clusters contradictorios que fuera (deltas -0.172, -0.090, -0.097): el modelo clasifica bien los ejemplos ambiguos, probablemente porque la anotacion mayoritaria del cluster (000) es la que el lexicon del insulto aprende. En consecuencia, **el ruido de anotacion por duplicacion no es una componente material del error de V03** en este corpus. La via de reponderar o descontar ejemplos contradictorios tiene por tanto un techo de impacto pequeno sobre la F1 global: actua sobre menos del 3 % de los ejemplos. La via con recorrido real es la dificultad general (29.7 % de los ejemplos falla alguna etiqueta) y la ambiguedad local medida por el vecino mas cercano, que es dificultad del modelo y no ruido de anotacion.

Este informe **no** corrige ni elimina etiquetas (§90): solo cuantifica que parte del error es coherente con ambiguedad textual y que parte es dificultad del modelo ante ejemplos no contradictorios.
