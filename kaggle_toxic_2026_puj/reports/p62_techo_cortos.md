# P62 - El fallo de los tweets cortos: techo de informacion o margen real

Generado por `experiments/p62_10_techo_cortos.py`. Fecha 2026-09-29T05:45:54+00:00.

> **DIAGNOSTICO.** No entrena, no crea candidatos, no usa test, holdout ni
> leaderboard. Es una medicion de identificabilidad sobre el OOF canonico.

## Que queda sin atacar

Cinco experimentos han medido y refutado las dos hipotesis naturales del eje de
representacion, y queda una anomalia medida que nadie ha tratado:

```text
TR corto, error de alta confianza    0,4167
TR largo,  error de alta confianza    0,0433
F1 de V03, rebanada corta menos larga  -0,2864 en TR
F1 de V03, sintaxis rica menos pobre  +0,1070, V03 es MEJOR con estructura
```

Las dos cosas juntas son extranas y apuntan a una sola explicacion: V03 no falla
porque ignore la gramatica, falla cuando no hay texto. Si eso es cierto, la region
corta es un TECHO DE INFORMACION. Si no, hay margen y la fase paro antes de tiempo.

## Como se distingue

Una afirmacion de techo es una afirmacion sobre identificabilidad, y eso se mide sin
entrenar nada. Dentro de la region corta pueden ser ciertas dos cosas a la vez, y
tienen consecuencias opuestas:

- **el ORDEN es bueno y solo la linea esta mal puesta**: es un problema de
  decision, que V8-09 y P59-T ya cerraron, luego es un techo en la practica.
- **el ORDEN es malo**: la representacion no separa las clases ni dentro de la
  region, luego es un techo de informacion genuino.

La medida es el AUC de la puntuacion de V03 calculado DENTRO de la region, y el
segundo control independiente es el techo de grupo por mayoria en los casi
duplicados, que es el limite duro.

La region corta es hasta 6 tokens de spaCy, la MISMA definicion que
uso la EDA linguistica, declarada alli antes de que existiera ningun candidato. Aqui
no se re-recorta para que convenga a un resultado.

## Los resultados por region

| et | region | n | F1 de V03 | error | error alta conf. | AUC interno | techo moviendo el corte |
|---|---|---:|---:|---:|---:|---:|---:|
| HS | corta | 136 | 0.9412 | 0.0809 | 0.0588 | 0.9615 | 0.9418 |
| HS | larga | 1347 | 0.7322 | 0.2264 | 0.1344 | 0.8508 | 0.7412 |
| TR | corta | 136 | 0.9704 | 0.0368 | 0.0294 | 0.9872 | 0.9708 |
| TR | larga | 1347 | 0.7795 | 0.1180 | 0.0794 | 0.9242 | 0.7851 |
| AG | corta | 136 | 0.9318 | 0.0882 | 0.0735 | 0.9430 | 0.9318 |
| AG | larga | 1347 | 0.7213 | 0.1967 | 0.1158 | 0.8566 | 0.7246 |

## El techo de grupo dentro de la region corta

| et | grupos uniformes | grupos contradictorios | filas cortas positivas en grupos contradictorios |
|---|---:|---:|---:|
| HS | 26 | 11 | 3 (3.2 por ciento) |
| TR | 29 | 8 | 3 (3.6 por ciento) |
| AG | 25 | 12 | 3 (3.5 por ciento) |

## Veredicto

```text
AUC medio dentro de la region corta        0.963900
margen hasta el techo moviendo el corte   +0.000323

V03 ORDENA bien la region corta, con un AUC interno de 0.9639, y el margen hasta el techo moviendo solo el corte es de +0.0003, luego es un problema de donde se dibuja la linea y no de que dice el texto. Eso es un techo EN LA PRACTICA: la unica palanca que lo moveria es el umbral, que V8-09 y P59-T ya cerraron. Ninguna representacion lo ataca
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_10_techo_cortos.py`. Sin push.*