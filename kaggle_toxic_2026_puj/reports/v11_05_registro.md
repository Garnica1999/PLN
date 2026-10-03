# V11-05: reconstruccion de las filas V11 del registro

Generado por `experiments/v11_05_registro.py` (V11-05 (reconstruccion del registro v11)).

## 1. Por que existe

Al commitear v10 en a30a467 y traerla a la v11 por merge, el experiments.csv del worktree de v11 retrocedio a una version anterior y las filas V11 se perdieron del registro. Los informes JSON remained intactos, de modo que las filas se reconstruyen derivandolas de ahi en vez de teclearlas.

Reconstruirlas a mano habria reintroducido el riesgo de errata que acababa de costar un
rato, asi que este script las **deriva de los informes**, que son la fuente de verdad. Es
reproducible: volver a ejecutarlo da el mismo registro.

Lo que no hace, a proposito: **no inventa cifras**. Cada numero sale de leer el informe, y
si una clave no existe alli la celda queda vacia en vez de rellenarse con una suposicion.

## 2. Claves extraidas por experimento

| experimento | claves extraidas | claves ausentes | informe |
| --- | --- | --- | --- |
| V11-00 | 13 | ninguna | v11_00_reconciliacion.json |
| V11-01 | 5 | ninguna | v11_01_reproduccion_v9_02.json |
| V11-02 | 3 | ninguna | v11_02_auditoria_legalidad.json |
| V11-03 | 5 | ninguna | v11_03_auditoria_cumplimiento.json |

## 3. Resultado

- filas reconstruidas: **4**, `V11-00`, `V11-01`, `V11-02`, `V11-03`
- registro final: **223 filas**, 223 claves unicas, 23 campos
- filas malformadas: **0**

## 4. El fallo que causo esto, y que conviene no repetir

Mi primer reanexado escribio las filas a mano, envolver cada campo entre comillas dobles
sin escapar las comas internas. El resultado fueron **4 filas malformadas**, que el
registro aceptaba pero con el numero de campos equivocado. Solo se detecto porque se
comprobro el numero de campos por fila, que es una comprobacion que ya se hacia en otros
sitios y que aqui falto.

Fecha: 2026-09-26T21:50:34+00:00. Duracion: 0.0 s.
