# P66 - Las cinco formas de fusion en el espacio de scores

Generado por `experiments/p66_02_fusion_espacio_scores.py`. Fecha 2026-09-29T15:49:01+00:00.

## Los cinco mecanismos, y lo que pregunta cada uno

| mecanismo | formula | pregunta |
|---|---|---|
| blend de probabilidad | `p = (1-a)p_V03 + a p_C` | mezcla ponderada |
| blend de logit | `z = z_V03 + a z_C` | evidencia aditiva |
| fusion LR | `LogisticRegression([z_V03, z_C])` | aprender la mezcla |
| **residual** | `z = z_V03 + b*(z_C - (a + b0*z_V03))` | **quitar lo redundante** |
| sensible a la longitud | residual con coeficiente por longitud | pesa mas en los largos |

P5 y P6 **no son nuevos** y se ejecutan como **controles**: V6-05 y P59-09 ya midieron un
alpha en espacio de probabilidad. Reproducir eso aqui es lo que hace interpretable a P8,
porque si P8 mejora donde la misma informacion en forma de P5 no, la diferencia es la
**residualizacion** y no el candidato.

## El protocolo nested, que es el punto entero

```text
outer_train -> inner_train     ajustar a, b0, la regresion y la normalizacion
outer_train -> inner_val       elegir alpha o beta por F1 con los umbrales de V03
outer_validation               EVALUAR SOLO. Nunca selecciona
```

La diferencia es entre una fusion cuyos parametros se eligieron sobre las mismas filas
contra las que se puntua, que siempre parece positiva, y una cuyos parametros se
congelaron antes de tocar la outer.

## La escala, antes de fusionar (P04)

| candidato | min p_C | max p_C |
|---|---:|---:|
| MI_arbol | 0.004069 | 0.987075 |
| negacion | 0.013227 | 0.998479 |
| word_1_3 | 0.011911 | 0.999041 |
| cohesion | 0.023161 | 0.997094 |
| huecos_1_2 | 0.013972 | 0.996575 |
| contraste | 0.014787 | 0.998576 |
| huecos_6_10 | 0.015829 | 0.998987 |
| **p_V03** | 0.014371 | 0.998639 |

Todas son probabilidades validas, luego la fusion en espacio de probabilidad es
admisible y **no hace falta calibrar antes**.

## Resultados sobre los candidatos reales

| candidato | mecanismo | delta vs V03 | pliegues | neto |
|---|---|---:|---:|---:|
| MI_arbol | blend_prob | +0.0033068923 | 3/5 | +15 |
| MI_arbol | residual | +0.0026743167 | 4/5 | +0 |
| negacion | length_aware | +0.0024316708 | 3/5 | +9 |
| MI_arbol | blend_logit | +0.0022432693 | 3/5 | +15 |
| MI_arbol | length_aware | +0.0021626530 | 3/5 | +1 |
| negacion | residual | +0.0011549795 | 3/5 | +5 |
| contraste | length_aware | +0.0007490637 | 2/5 | +2 |
| huecos_1_2 | residual | +0.0003505310 | 3/5 | +2 |
| negacion | blend_logit | +0.0002260211 | 2/5 | +6 |
| cohesion | residual | +0.0000325761 | 1/5 | +2 |
| contraste | residual | +0.0000000000 | 0/5 | +0 |
| word_1_3 | residual | -0.0001400584 | 2/5 | +0 |
| contraste | blend_logit | -0.0002087275 | 2/5 | +4 |
| word_1_3 | length_aware | -0.0003612173 | 2/5 | -1 |
| contraste | blend_prob | -0.0003687937 | 0/5 | -1 |
| huecos_1_2 | length_aware | -0.0003989428 | 2/5 | -4 |
| huecos_1_2 | blend_prob | -0.0005821223 | 2/5 | +0 |
| huecos_1_2 | blend_logit | -0.0006553566 | 3/5 | +3 |
| word_1_3 | blend_prob | -0.0006600168 | 1/5 | -1 |
| huecos_6_10 | blend_prob | -0.0007298816 | 2/5 | -2 |

## Los mismos mecanismos con el score del CANDIDATO BARRAJADO

Esta es la fila que decide si una mejora es **del candidato** o de la **mecanica**.

| candidato | mecanismo | delta real | delta barajado | brecha de atribucion |
|---|---|---:|---:|---:|
| MI_arbol | blend_prob | +0.0033068923 | -0.0060147443 | +0.0093216366 |
| MI_arbol | blend_logit | +0.0022432693 | -0.0063023874 | +0.0085456567 |
| MI_arbol | fusion_lr | -0.0025748748 | -0.0062881092 | +0.0037132344 |
| MI_arbol | residual | +0.0026743167 | -0.0014390421 | +0.0041133588 |
| MI_arbol | length_aware | +0.0021626530 | -0.0009323121 | +0.0030949651 |
| negacion | blend_prob | -0.0011104651 | -0.0100357467 | +0.0089252816 |
| negacion | blend_logit | +0.0002260211 | -0.0066457637 | +0.0068717848 |
| negacion | fusion_lr | -0.0047512375 | -0.0087245807 | +0.0039733432 |
| negacion | residual | +0.0011549795 | -0.0054335321 | +0.0065885116 |
| negacion | length_aware | +0.0024316708 | -0.0073732756 | +0.0098049464 |
| word_1_3 | blend_prob | -0.0006600168 | -0.0030627989 | +0.0024027821 |
| word_1_3 | blend_logit | -0.0016174424 | -0.0055140851 | +0.0038966427 |
| word_1_3 | fusion_lr | -0.0112336458 | -0.0100146903 | -0.0012189555 |
| word_1_3 | residual | -0.0001400584 | +0.0038763165 | -0.0040163749 |
| word_1_3 | length_aware | -0.0003612173 | -0.0025979168 | +0.0022366995 |
| cohesion | blend_prob | -0.0010938505 | -0.0068003573 | +0.0057065068 |
| cohesion | blend_logit | -0.0009644551 | -0.0042640342 | +0.0032995791 |
| cohesion | fusion_lr | -0.0085040049 | -0.0067000231 | -0.0018039818 |
| cohesion | residual | +0.0000325761 | -0.0070646241 | +0.0070972002 |
| cohesion | length_aware | -0.0015089878 | -0.0065419228 | +0.0050329350 |
| huecos_1_2 | blend_prob | -0.0005821223 | -0.0064097916 | +0.0058276693 |
| huecos_1_2 | blend_logit | -0.0006553566 | -0.0067732812 | +0.0061179246 |
| huecos_1_2 | fusion_lr | -0.0085230557 | -0.0074595281 | -0.0010635276 |
| huecos_1_2 | residual | +0.0003505310 | -0.0064731429 | +0.0068236739 |
| huecos_1_2 | length_aware | -0.0003989428 | -0.0061395457 | +0.0057406029 |
| contraste | blend_prob | -0.0003687937 | -0.0018160015 | +0.0014472078 |
| contraste | blend_logit | -0.0002087275 | -0.0007130585 | +0.0005043310 |
| contraste | fusion_lr | -0.0074527010 | -0.0087287438 | +0.0012760428 |
| contraste | residual | +0.0000000000 | -0.0017878950 | +0.0017878950 |
| contraste | length_aware | +0.0007490637 | -0.0031585575 | +0.0039076212 |
| huecos_6_10 | blend_prob | -0.0007298816 | -0.0039354437 | +0.0032055621 |
| huecos_6_10 | blend_logit | -0.0023602121 | -0.0047123692 | +0.0023521571 |
| huecos_6_10 | fusion_lr | -0.0047093408 | -0.0064282522 | +0.0017189114 |
| huecos_6_10 | residual | -0.0028447979 | +0.0004009382 | -0.0032457361 |
| huecos_6_10 | length_aware | -0.0029694813 | -0.0039028402 | +0.0009333589 |

## P16, informacion unica mas alla de V03

| candidato | corr con z_V03 | AUC candidato | AUC del RESIDUO | AUC del margen |
|---|---:|---:|---:|---:|
| MI_arbol | +0.9294 | 0.555824 | 0.565387 | 0.229125 |
| negacion | +0.9921 | 0.542904 | 0.453645 | 0.229125 |
| word_1_3 | +0.9963 | 0.550512 | 0.514599 | 0.229125 |
| cohesion | +0.9971 | 0.545381 | 0.441306 | 0.229125 |
| huecos_1_2 | +0.9910 | 0.550435 | 0.498082 | 0.229125 |
| contraste | +0.9988 | 0.548418 | 0.482872 | 0.229125 |
| huecos_6_10 | +0.9931 | 0.545601 | 0.418585 | 0.229125 |

El AUC del **residuo** es la medida clave: si desaparece, el candidato es redundante con
V03 y no hay nada que fusionar.

## Veredicto

```text
mejor real       MI_arbol|blend_prob con +0.0033068923 y 3/5
mismo, barajado  -0.0060147443
brecha           +0.0093216366

MI_arbol|blend_prob llega a +0.0033068923 con 3 de 5 pliegues, y la brecha de atribucion es +0.0093216366. Supera el gate de magnitud pero no el de consistencia

DECISION: MARGINAL
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p66_02_fusion_espacio_scores.py`. Sin push.*