# V35 via B - Sonda pre-ejecucion del anclaje de `lambda1`

**Fecha**: 2026-09-30. **Estado**: exploracion metodologica previa (NO es un grid), sobre filas
de **TRAIN** del fold 0 (el 90 % del ES-train). Fold-val, test y leaderboard intocados.

## Anclaje derivado

- `lambda1 := mediana(|z_j| > 0)` al final de la pasada 1 = **0.23993793862904017**
- `mediana / alpha` = 6.8044 (alpha = 0.035262336465737745); fraccion con `|z| <= alpha` = 0.144558
- Cuantiles de `|z|`: {"q1": 0.00010707276487313874, "q25": 0.063655343376633, "q50": 0.23993793862904017, "q75": 0.6777203538570445, "q99": 7.910165751736505}
- `sha256` del snapshot de `z`: `ec1d1a7b13454f8b7309446576318487b9325df018c5b454b228a6a336630f66`

## Los tres casos, con las mismas reglas de update y los mismos ordenes

| Caso | lambda1 | ll_train (p1 -> p3) | ||w||_2 (p1 -> p3) | escasez efectiva (p3) |
|---|---|---|---|---|
| lambda1_0 | 0.0 | 10.086314 -> 19.174421 | 29.1949 -> 60.9836 | 1.0000 |
| lambda1_alpha | 0.035262336465737745 | 10.083241 -> 19.172685 | 29.0289 -> 60.8635 | 0.9924 |
| lambda1_mediana | 0.23993793862904017 | 10.058343 -> 19.159444 | 28.2746 -> 60.2372 | 0.9807 |

## Los dos flags tecnicos, resueltos con medicion

1. *"`lambda1 = alpha` es un no-op"* exige `mediana(|z|) > alpha`. Medido: se cumple (razon 6.8044).
2. *"un `lambda1` fijo acota `||w||_2`"*: con `lambda1` = mediana la razon max/min de la norma en las 3 pasadas es 2.1304, luego el L1 NO acota la norma por si solo (solo acota si z permanece acotado).
   Es exactamente lo que la puerta comprueba con G8 sobre `||w||_2`.

## Lectura

- El anclaje por mediana es utilizable y el caso `lambda1 = alpha` es efectivamente no-op en este setting.
- La divergencia se **amortigua pero no se elimina**: la hipotesis de acotacion sigue siendo la que G8 debe poner a prueba, y el resultado sigue siendo el de baja probabilidad deporte que declaro `ml-architect`.
- El runner de V35-B repetira estos tres casos **dentro** de si como sondas no-gate (patron de la sonda (j) de D-085) para que la decision no descanse en nada externo.
