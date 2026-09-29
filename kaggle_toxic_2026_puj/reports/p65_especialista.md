# P65 - Especialista de tweets largos

Generado por `experiments/p65_04_especialista.py`. Fecha 2026-09-29T14:50:24+00:00.

## Que hay de nuevo, frente al precedente

**P63 y P64 midieron el MARGEN** por rebanada de longitud: casi nueve centesimas de macro
F1 en MUY_LARGA que ningun umbral alcanza. **Ninguno entreno un MODELO** en los tweets
largos. Un margen medido y un modelo ajustado son objetos distintos, y el segundo puede
cambiar que caracteristicas importan en un tweet largo.

## El diseno, con cada parte fijada antes de medir

```text
global        V03 exactamente como es
especialista  un segundo V03 ajustado solo en LARGA y MUY_LARGA, 714 filas
puerta        determinista y CONGELADA ANTES DE VER NADA: 16 tokens o mas
```

La puerta **no se afina, no se aprende, no es un selector y no se elige maximizando F1**.
Dieciseis tokens es la frontera que los propios cortes de P63 ya usaban entre MEDIA y
LARGA, luego la puerta se **hereda** en vez de elegirse, y heredarla es la unica manera
de mantener la afirmacion honesta: una puerta elegida despues de ver el resultado es un
parametro ajustado vestido de parametro fijo.

## Los tres controles, que son el punto real de este script

| control | que separa |
|---|---|
| reentrenado | especializacion de redistribucion |
| aleatorio | especializacion de menos filas |
| solo_largo | referencia like-for-like por estrato |

Si el especialista solo bate al global, queda un hueco. Si **no** bate al reentrenado, la
longitud nunca fue el motivo y la familia cierra sobre la base mas fuerte posible.

## Resultados

| brazo | corrected_cv | delta vs V03 | pliegues | neto |
|---|---:|---:|---:|---:|
| global | 0.7772114525 | +0.0000000000 | 0/5 | +0 |
| especialista | 0.7691517953 | -0.0080596573 | 2/5 | -48 |
| reentrenado | 0.7691517953 | -0.0080596573 | 2/5 | -48 |
| aleatorio | 0.7654406468 | -0.0117708057 | 1/5 | -30 |

### Por estrato

| bin | n | global | especialista | reentrenado | aleatorio |
|---|---:|---:|---:|---:|---:|
| CORTA | 167 | 0.9480 | 0.9480 | 0.9480 | 0.9480 |
| MEDIA | 602 | 0.7964 | 0.7964 | 0.7964 | 0.7964 |
| LARGA | 547 | 0.6834 | 0.6577 | 0.6577 | 0.6577 |
| MUY_LARGA | 167 | 0.5606 | 0.5818 | 0.5818 | 0.4700 |

### Las tres preguntas

```text
el especialista supera al reentrenado: +0.0000000000
el especialista supera al aleatorio:   +0.0037111485
el especialista supera al global:      -0.0080596573
en LARGA, global 0.6834 contra especialista 0.6577
en MUY_LARGA, V03 0.5606 contra especialista 0.5818
```

La primera es la importante: si el reentrenado en las mismas filas llega a lo mismo,
entonces lo que se ha medido no es especializacion sino redistribucion.

## Veredicto

```text
El especialista NO supera al reentrenado: +0.0000000000. Es decir, ajustar un V03 sobre las mismas 714 filas, sin ningun criterio de longitud, consigue lo mismo o mas. Y el global es -0.0080596573. Lo que se ha medido no es que los tweets largos necesiten un modelo distinto, es que tener menos filas cambia el ajuste. Y en MUY_LARGA, que es donde P63 midio el margen de +0,0890, el especialista deja F1 en 0.5818 frente a 0.5606 de V03. La familia del especialista se cierra con el control mas fuerte de las cinco que han hecho falta en esta fase, porque separa las dos explicaciones y las separa a favor de la que no interesa

LECTURA: REDISTRIBUCION_NO_ESPECIALIZACION
DECISION: CLOSED
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p65_04_especialista.py`. Sin push.*