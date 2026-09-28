# V26 / V27 - Reproducibilidad

Niveles: `R0` no reproducible, `R1` reproducible con limitaciones, `R2` reproducido dentro de tolerancia, `R3` reproduccion exacta y `R4` reproduccion independiente.

## R3 - Reproduccion exacta, en todos los experimentos medidos en V27

| Experimento | Control | Reproduce | Delta | Decision |
|---|---|---|---|---|
| H12 (`V27 01 H12 fiabilidad por modelo y clase`) | 0.6097782 | True | +0.000000 | H12 CLOSED |
| H11 (`V27 02 H11 restricciones monotonas`) | 0.6097782 | True | -0.000355 | H11 CLOSED |
| H03 (`V27 05 H03 percentil`) | 0.6097782 | True | +0.000272 | H03 MARGINAL |
| H02 (`V26 01 representacion afin de la evidencia`) | 0.6097782 | True | -0.000001 | H02 CLOSED |

Los tres controles reproducen el canonico **0,6097782** antes de que el candidato se ejecute, y los tres se detienen si no lo hacen.

## Que se verifica en cada experimento

| Comprobacion | Estado |
|---|---|
| Baseline intacto | V20 con hash de submission sin cambio |
| Mismos folds | `[33868, 33868, 33867, 33866, 33867]`, los de V20 |
| Mismo artefacto de blend | `v20_f3b_oof_candidato.npy` sin modificar |
| Misma semilla | `cfg.SEED`, sin semilla nueva para seleccionar |
| Misma generacion de features | `matriz_279()` contra el inventario de V25 |
| Misma configuracion del meta-modelo | `HIERBA` del champion, importada |
| Mismos thresholds | cadena de cinco umbrales con maximo acumulado |

## Lo que NO se declara

- **No se declara R4**: no hubo reproduccion independiente por un tercero ni con codigo separado. Todos los controles leen el mismo artefacto que el candidato, que es lo correcto para garantizar una comparacion justa, pero no es verificacion independiente.

## Una limitacion que si se declara

El diagnostico de mecanismo de H11 reporto `0/0 splits` en su primera version por contar una clave inexistente de `dump_model()`. Se corrigio en `experiments/v27_03_h11_mecanismo.json`, y el artefacto original apunta ahi con una nota de edicion. **El delta, los folds y la decision de H11 no se modificaron**: lo que estaba mal era la medida, no el experimento.
