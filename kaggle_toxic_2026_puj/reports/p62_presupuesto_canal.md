# P62 - Cuanto canal de interaccion puede permitirse V03

Generado por `experiments/p62_07_presupuesto_canal.py`. Fecha 2026-09-29T05:37:16+00:00.

## Que deja abierto P62-01 y P62-02

Las dos pistas cerradas midieron su propia causa, y fue una sola:

```text
pista L   coste del bloque  +0,011585   valor de la estructura  +0,002014
pista M   coste del bloque  +0,000443   valor de la estructura  -0,010793
```

La senal de interaccion es real y vale alrededor de +0,002, y el obstaculo es que
anadir un bloque a un espacio de 21.000 dimensiones con 1.186 filas cuesta mas de lo
que la categoria repara. Eso es una afirmacion sobre un REGIMEN, y una afirmacion
sobre un regimen se mide: con dos puntos no se puede llamar ley.

## El diseno

Para un canal de un tamano dado, se mide lo que CUESTA y lo que VALE:

- **brazo ruido**: un bloque de K rasgos SIN SENTIDO. Lo que hace es coste y nada
  mas. Esa es la curva de coste.
- **brazo real**: los K tokens de interaccion mas enriquecidos, elegidos con las
  filas de entrenamiento del pliegue. La diferencia entre los dos brazos en cada K es
  el valor del contenido a ese presupuesto.

**No es una rejilla de hiperparametros.** K es la variable independiente de una
medicion, el clasificador es el de V03 en todos los brazos, y no se selecciona ningun
modelo por su puntuacion. Se reporta la curva entera.

Los rasgos reales se ordenan por enriquecimiento ABSOLUTO, calculado fold-safe, para
no quedarse solo con los que favorecen la clase positiva. Elegirlos por F1 seria
seleccion de rasgos, que esta cerrada.

## La curva

| K | ruido | delta ruido | real | delta real | valor del contenido |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7772114525 | +0.0000000000 | 0.7772114525 | +0.0000000000 | +0.0000000000 |
| 16 | 0.7780667140 | +0.0008552614 | 0.7764050796 | -0.0008063729 | -0.0016616344 |
| 32 | 0.7774930914 | +0.0002816389 | 0.7735938371 | -0.0036176154 | -0.0038992543 |
| 64 | 0.7759317704 | -0.0012796821 | 0.7729974033 | -0.0042140492 | -0.0029343672 |
| 128 | 0.7754964521 | -0.0017150004 | 0.7667815925 | -0.0104298600 | -0.0087148596 |
| 256 | 0.7762787289 | -0.0009327236 | 0.7684648850 | -0.0087465675 | -0.0078138440 |
| 512 | 0.7704720617 | -0.0067393908 | 0.7694731115 | -0.0077383410 | -0.0009989502 |
| 1024 | 0.7682894416 | -0.0089220109 | 0.7703508310 | -0.0068606215 | +0.0020613894 |

## Lectura

```text
NINGUN presupuesto da al canal real una mejora sobre V03. El valor del contenido es positivo en varios K, luego la senal existe, pero el coste de anadir el bloque es mayor en todos ellos. La familia queda cerrada por la aritmetica del regimen, no por dos negativos sueltos
```

## Decision

```text
CLOSED. Ningun presupuesto da al canal una mejora sobre V03
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_07_presupuesto_canal.py`. Sin push.*