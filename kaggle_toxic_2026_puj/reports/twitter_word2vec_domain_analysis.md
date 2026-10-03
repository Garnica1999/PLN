# Dominio Twitter frente a general, con el protocolo corregido y control de azar

Generado por `experiments/p71_10_downstream_corregido.py`. Fecha 2026-09-29.

## Los tres defectos que tenia mi downstream

```text
UNO  el vectorizador se ajustaba sobre TODO el desarrollo, no dentro de cada pliegue
     la forma correcta reproduce el OOF de V03 a 0,777211 EXACTO; el atajo daba 0,778004
DOS  usaba C=1.0 sin class_weight, cuando V03 usa C=2.0 con balanced
TRES la escala del bloque denso no estaba igualada a la del disperso
     y normalize() sobre el concatenado NO es el arreglo: encoge el TF-IDF unas 20 veces
```

Verificacion: la reconstruccion da `0.7772114525`
y el canonico es `0,7772114525165721`. Reproduce:
`True`.

## El barrido de escala, con el control de azar

| escala | twitter | general | AZAR |
|---:|---:|---:|---:|
| 0.00 | 0.777211 | 0.777211 | 0.777211 |
| 0.05 | 0.776423 | 0.775921 | 0.776896 |
| 0.10 | 0.775406 | 0.773801 | 0.777384 |
| 0.20 | 0.772744 | 0.770983 | 0.772363 |
| 0.50 | 0.764054 | 0.760003 | 0.756933 |
| 1.00 | 0.749167 | 0.748361 | 0.725440 |

**V03 sola: `0,777211`.**

## La lectura que decide

### Uno: la fusion NUNCA bate a V03

En todas las escalas, de 0,05 a 1,00, el resultado esta **por debajo** de `0,777211`. El
mejor punto es `0,776423` en twitter a escala 0,05, que es `-0,0008`. Anadir el bloque denso
siempre cuesta.

### Dos: donde menos cuesta, los embeddings son indistinguibles del ruido

| escala | twitter menos azar | general menos azar |
|---:|---:|---:|
| 0.00 | +0.000000 | +0.000000 |
| 0.05 | -0.000473 | -0.000975 |
| 0.10 | -0.001978 | -0.003583 |
| 0.20 | +0.000381 | -0.001380 |
| 0.50 | +0.007121 | +0.003070 |
| 1.00 | +0.023727 | +0.022921 |

**A las escalas bajas, donde la fusion hace menos dano, los bloques reales rinden IGUAL O
PEOR que el ruido:** twitter `-0,000473` y `-0,001978`; general `-0,000975` y `-0,003583`.

El exceso positivo aparece solo a escala alta (`+0,023727` en twitter a 1,00), pero ahi la
fusion esta en `0,749`, veintiocho puntos por debajo de V03. **Que el ruido destruya mas no
es que el embedding aporte.**

## La hipotesis del vocabulario, medida

| corpus | vocabulario | tokens por fila | cubiertos | fraccion | filas sin ningun token |
|---|---:|---:|---:|---:|---:|
| twitter | 77,657 | 18.2 | 12.49 | 68.6% | 70 |
| general | 100,097 | 18.2 | 11.99 | 65.9% | 72 |

La escasez existe: **68,6 por ciento de cobertura en twitter y 65,9 en general**, con 70 y 72
filas que no tienen ni un token en vocabulario. Pero **no es la causa limitante**: el bloque
ALEATORIO se comporta igual, y a el no le falta cobertura ninguna.

## La causa real, dicha sin rodeos

```text
la tarea es LEXICA: son n-gramas concretos los que senalan toxicidad
un embedding promediado de 200 dimensiones es una bolsa de palabras COMPRIMIDA
no puede codificar QUE n-gramas estan presentes, y el TF-IDF de 24.646 si

el cuello no es el corpus. Es el pooling y la dimension.
```

v5 ya lo habia visto con corpus generales. Esto lo confirma con un corpus de dominio.

## El control de dominio, respondido

Twitter bate a general en **todas** las escalas, pero por cantidades que van de `+0,0005` a
`+0,0071`, y en las escalas bajas el ruido se cuela entre los dos. **No hay senal de dominio
que se pueda defender.**

## Veredicto

```text
CLOSED
```

No porque el corpus sea malo: el corpus esta validado, con periodo cerrado y cero
solapamiento con train. Sino porque **la representacion promediada no puede competir con una
representacion lexica dispersa en una tarea lexica**, y eso ya lo habia medido v5.

> V03 remains the best verified leaderboard result.

*Sin push.*