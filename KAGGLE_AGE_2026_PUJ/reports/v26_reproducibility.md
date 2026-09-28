# V26 · Reproducibilidad

Clasificacion del punto 36 del brief. Los cinco niveles son `R0` no reproducible, `R1` reproducible con limitaciones, `R2` reproducido dentro de tolerancia, `R3` reproduccion exacta y `R4` reproduccion independiente.

## R3 — Reproduccion exacta del baseline

| Comprobacion | Estado |
|---|---|
| Baseline intacto | V20, hash de submission sin cambio, artefactos no modificados |
| Mismos folds | [33868, 33868, 33867, 33866, 33867], los mismos de V20 |
| Mismo data hash | `v20_f3b_oof_candidato.npy` sin modificar |
| Misma semilla | `cfg.SEED`, sin semilla nueva para seleccionar |
| Misma generacion de features | `matriz_279()` verificada bit a bit contra el inventario de V25 |
| Misma configuracion del meta-modelo | `HIERBA` importada del champion, sin tocar |
| Mismos thresholds | cadena de cinco umbrales con maximo acumulado, identico a V20 |
| **Reproduccion del control** | **0.6097782** frente a 0.6097782, coincide: **True** |

**Nivel declarado: R3, reproduccion exacta.** No se declara R4 porque no hubo una reproduccion por un tercero ni con codigo independiente: el control se leyo del mismo artefacto que usa el candidato, que es la forma correcta de garantizar que la comparacion sea justa, pero no es una verificacion independiente.

## Del candidato

- Delta frente a V20: **-0.000001** (gate +0.003).
- Folds positivos: **2/5** (gate 4/5).
- Decision: **H02 CLOSED**.

## Lo que NO se declara

- No se declara R4: no hubo reproduccion independiente.
- No se declara R2 sin mas: el control es exacto, luego R3 es el nivel correcto y mas fuerte que R2.
