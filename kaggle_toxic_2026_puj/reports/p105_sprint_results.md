# P105 - RONDA 4 (S2/S7a + S4/S7b + S7c): resultados del sprint

Fecha: 2026-10-14. Pre-registro: `reports/p105_preregistro.md` (gate A3;
control V03 exacto en ambas particiones; piso de deteccion declarado;
presupuesto 1 h CPU). Alcance confirmado: SOLO S2-residual/S7a, S4/S7b y S7c,
en ese orden.

## 1. Resultados

| Item | Que probo | Canonica | Particion 47 | Veredicto |
|---|---|---|---|---|
| P105-01 (S2/S7a) | normalizaciones alternativas de la union | V1 (L2 por bloque): **identidad con V03** (diff 0.0; 0 celdas); V2 (L2 conjunta): **-0.017517** (1/5; 282 celdas; IC95 [-0.028724,-0.006328]); V3 (norma media): **-0.034369** (1/5; 373 celdas; IC95 [-0.048696,-0.020160]) | V2 -0.018905; V3 -0.036038 | GATE_FAILED: V2/V3 efecto NEGATIVO detectable; V1 es identidad (no compite) |
| P105-02 (S4/S7b) | canal fonetico es (b/v, ll/y, h, c/k/qu, s/z) como char adicional | **-0.000363** (1/5; 96 celdas; IC95 [-0.006092,+0.005504]; piso +0.005798) | +0.000414 (sin celdas) | GATE_FAILED; sin senal detectable |
| P105-03 (S7c) | peso char por etiqueta (malla 0.5-1.5, nested) | **-0.001125** (2/5; 67 celdas; IC95 [-0.005996,+0.003508]; piso +0.004752) | -0.000178 | GATE_FAILED; sin senal detectable |

Control en los tres: reproduce V03 exacto en la canonica (diff proba 0.00e+00;
ccv 0.7772114525165721) y en la particion 47 (0 celdas; diff 0.00e+00).

## 2. P105-01 (S2/S7a) - detalle

- **V1 (L2 por bloque) es identidad matematica con V03**: los bloques de V03 ya
  son unitarios por fila (norm='l2' de sklearn); el experimento lo confirma con
  diff 0.0 y 0 celdas. Queda documentado que la propuesta "L2 por bloque" del
  triaje ES la normalizacion actual.
- V2 (L2 conjunta sobre bloques crudos) y V3 (reescalado a la norma media)
  perjudican con IC95 enteramente negativo: romper el balance implicito
  (cada bloque unitario) del hstack de V03 no ayuda.

## 3. P105-02 (S4/S7b) - detalle

- Ejemplos del folding (verificados): vacaciones->bakasiones, quiero->kiero,
  caballo->kabayo, huevo->uebo, mucha->mucha.
- Resultado: delta -0.000363, bajo el piso de deteccion (+0.005798) e IC95 que
  incluye cero; la particion 47 da +0.000414 con 96 celdas: sin senal
  detectable en ninguna lectura. char_wb ya absorbia la variacion que el
  folding captura.

## 4. P105-03 (S7c) - detalle

- Seleccion inestable: HS 1.25/1.5 (sube el peso char); TR 0.5-1.25; AG
  0.75-1.5; ningun peso se repite consistentemente.
- Resultado: -0.001125, IC95 que incluye cero, 67 celdas: sin senal
  detectable. La libertad de peso por etiqueta no aporta sobre el peso 1.0
  fijo de V03.

## 5. Cierre

- Sin candidato: los tres items fallan el gate A3 completo. La serie de rondas
  acumula ~15 experimentos sin candidato.
- V03 permanece congelado (corrected_cv 0.7772114525165721; LB 0.78306).
- Recomendacion de slots del leaderboard privado SIN CAMBIOS: V03 +
  v08_meta_v2 (LB 0.78163).
- Presupuesto: ~15 min de CPU de 1 h (231.2 + 81.6 + 94.7 s, regeneracion
  incluida).
- Sin test; sin holdouts; sin leaderboard; sin submissions; artefactos
  historicos intactos.

## 6. Artefactos

- Scripts: `experiments/p105_01_normalizacion_bloques.py`,
  `p105_02_fonetica.py`, `p105_03_pesos_bloque_etiqueta.py`,
  `p105_90_registrar.py`.
- Resultados: `reports/p105_01_normalizacion_bloques.{json,md}`,
  `p105_02_fonetica.{json,md}`, `p105_03_pesos_bloque_etiqueta.{json,md}`,
  `p105_preregistro.md`.
- Registro: filas P105-01/02/03 en `experiments.csv` (504 filas).
