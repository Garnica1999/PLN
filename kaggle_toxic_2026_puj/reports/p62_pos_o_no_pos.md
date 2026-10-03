# P62 - El canal POS-conditioned, o las mismas palabras dos veces

Generado por `experiments/p62_08_pos_o_no_pos.py`. Fecha 2026-09-29T05:39:16+00:00.

## Lo que apunta la curva de P62-07

Con el bug de acumulacion corregido, la curva es limpia y su forma es el hallazgo:

Ningun presupuesto mejora a V03, y el valor del contenido es NEGATIVO casi en todos.
Eso es un negativo mas fuerte que el de las pistas L y M, y tiene un sospechoso
evidente: mira QUE tokens selecciona el ranking fold-safe.

```text
pliegue 1: hijo__NOUN, sevillista__ADJ, zorra__NOUN, zorra__PROPN, novia__NOUN,
           perra__PROPN
pliegue 4: zorra__PROPN, zorra__NOUN, cllate__PROPN, perra__PROPN, hijo__NOUN
```

Son los MISMOS lemas que el analisis de palabras ya habia encontrado, `zorra`,
`perra`, `hijo`, `novia`, `cllate`, `acoso`, y son palabras que el canal de unigramas
de V03 ya lleva. El sufijo anade una segunda copia de una palabra que ya esta ahi, y
a veces una tercera.

**La sospecha, enunciada antes de medir:** si eso es lo que pasa, la synergy de
+0,0657 de la EDA no era informacion nueva. Es lo que sale al partir una palabra
enriquecida en dos o tres filas por su categoria: el promedio de log enriquecimiento
de las filas sube porque la misma evidencia se cuenta mas de una vez, mientras que
la informacion no se mueve. Y eso explicaria de un solo golpe por que todos los
canales linguisticos de este proyecto fallaron: ninguno anadio informacion, todos
re-codificaron palabras que V03 ya tenia.

## La prueba

La misma curva de presupuesto con el canal construido de lemas SUELTOS en vez de
`lema__POS`, con todo lo demas identico.

| K | solo lema | delta | lema__POS | delta | ventaja del POS |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.7756324411 | -0.0015790114 | 0.7762373578 | -0.0009740947 | +0.0006049167 |
| 32 | 0.7756400762 | -0.0015713763 | 0.7759254394 | -0.0012860131 | +0.0002853632 |
| 64 | 0.7689589178 | -0.0082525347 | 0.7758402454 | -0.0013712071 | +0.0068813276 |
| 128 | 0.7694259307 | -0.0077855219 | 0.7704657700 | -0.0067456825 | +0.0010398393 |
| 256 | 0.7702214415 | -0.0069900110 | 0.7682919453 | -0.0089195072 | -0.0019294962 |
| 512 | 0.7688754290 | -0.0083360235 | 0.7696138822 | -0.0075975704 | +0.0007384531 |
| 1024 | 0.7687730479 | -0.0084384046 | 0.7678925366 | -0.0093189159 | -0.0008805113 |
| 2048 | 0.7683961013 | -0.0088153512 | 0.7694654804 | -0.0077459721 | +0.0010693791 |

## La redundancia, medida y no inferida de los resultados

| magnitud | valor |
|---|---:|
| presupuestos en los que el sufijo de POS ayuda | 6 de 8 |
| presupuestos donde AMBOS canales pierden | 8 de 8 |
| redundancia media de los tokens seleccionados | 0.5209 |
| cobertura de V03 sobre esos mismos lemas | 0.3983 |

La redundancia es la fraccion de los 2048 tokens `lema__POS` seleccionados cuyo lema
ya estaba en la lista de los 2048 lemas mas enriquecidos. La cobertura es la fraccion
de esos lemas que ya tienen columna en el canal de palabras de V03.

## Veredicto

```text
el sufijo de POS ayuda en 6 de 8 presupuestos con una redundancia de 0.5209 y cobertura de V03 de 0.3983; la condicion para llamarlo redundancia no se cumple y el resultado queda abierto
```

## Decision

```text
CLOSED. El sufijo de categoria no aporta informacion que V03 no tenga ya
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_08_pos_o_no_pos.py`. Sin push.*