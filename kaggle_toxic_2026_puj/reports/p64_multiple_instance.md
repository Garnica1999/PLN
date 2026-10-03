# P64 - Modelo multiple-instance clasico y agregador aprendido

Generado por `experiments/p64_05_multiple_instance.py`. Fecha 2026-09-29T14:15:07+00:00.

## Por que esto NO es un rerun de P59-07

P59-07 cambio la UNIDAD DE REPRESENTACION a una ventana local y salio por el maximo,
y dio -0,0006246152 con neto -1. Pasaron dos cosas ahi y solo una fue el concepto.

**El concepto se probo. Lo que nunca se probo fue el mismo concepto con una unidad que
puede alcanzar las filas que importan.** P59-07 registro la razon de forma explicita:
los tweets son cortos, mediana de 15 tokens, luego una ventana de 32 deja mas de una
ventana en el 9,6 por ciento de las filas. El fracaso fue diagnosticado como
**COBERTURA**.

La segmentacion por puntuacion llega al 50,6 por ciento, 5,29 veces mas. Este script
entrena el modelo de segmento-como-instancia sobre segmentos de puntuacion y no sobre
ventanas fijas, y no cambia nada mas. Si la respuesta sigue siendo negativa, sera
negativa sobre el CONCEPTO, que es una afirmacion mucho mas fuerte que la de P59.

## Los tres mecanismos, que son distintos

| brazo | mecanismo |
|---|---|
| MI_seg | el segmento es la instancia del clasificador y el score es el maximo |
| MI_lineal | evidencia + contribucion + margen + longitud al LogisticRegression de V03 |
| MI_arbol | el mismo bloque a un arbol de profundidad 3 |

El bloque denso tiene **88 columnas**: 48 de
evidencia, 36 de contribucion, 3 de margen y 1 de longitud.

### Por que se permite el brazo no lineal

La regla es pasar primero por LogisticRegression y solo probar un modelo no lineal
pequeno si hay evidencia de que la senal existe y un lineal no la alcanza. Esa
evidencia esta **medida y no supuesta**: P64-03 y P64-04 lo
midieron por debajo del margen mientras las lecturas distribucionales muestran una
estructura interna grande y real. Una sola configuracion, declarada antes, sin
busqueda alrededor:

```text
{"l2_regularization": 1.0, "learning_rate": 0.06, "max_depth": 3, "max_iter": 120, "min_samples_leaf": 20, "random_state": 42}
```

## Resultados

| brazo | corrected_cv | delta vs V03 | pliegues | neto |
|---|---:|---:|---:|---:|
| MI_seg | 0.7222777580 | -0.0549336945 | 0/5 | -400 |
| MI_lineal | 0.7630276292 | -0.0141838233 | 0/5 | -73 |
| MI_arbol | 0.7680427726 | -0.0091686799 | 1/5 | -26 |
| MI_aleatorio | 0.7516577509 | -0.0255537016 | 0/5 | -104 |
| V03_margen | 0.7586603806 | -0.0185510720 | 1/5 | -128 |

### Por estrato de longitud

| bin | n | F1 MI_seg | F1 MI_lineal | F1 MI_arbol | F1 MI_aleatorio | F1 V03_margen |
|---|---:|---:|---:|---:|---:|---:|
| CORTA | 167 | 0.9117 | 0.9407 | 0.9403 | 0.9383 | 0.9335 |
| MEDIA | 602 | 0.7573 | 0.7848 | 0.7850 | 0.7738 | 0.7730 |
| LARGA | 547 | 0.6255 | 0.6655 | 0.6732 | 0.6470 | 0.6733 |
| MUY_LARGA | 167 | 0.5163 | 0.5584 | 0.5568 | 0.4809 | 0.5592 |

## Las brechas, que son las unicas preguntas

```text
MI_seg    contra el control de anchura: -0.0293799929
MI_lineal contra el margen de V03:      +0.0043672486
MI_arbol  contra el margen de V03:       +0.0093823920
MI_arbol  contra MI_lineal:              +0.0050151434
```

La ultima responde a la regla lineal contra no lineal: si el arbol solo mejora frente
a un lineal, la senal existe pero es lineal, y una senal lineal que un lineal no
expresa ya esta en V03.

## Veredicto

```text
el modelo no lineal mejora al lineal en +0.0050151434 y supera al margen en +0.0093823920, luego la senal existe y es no lineal, aunque no llega al gate de +0.0015

LECTURA: NO_LINEAL_UTIL
DECISION: COMPLEMENTARY
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p64_05_multiple_instance.py`. Sin push.*