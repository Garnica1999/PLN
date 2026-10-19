# P93-R4 - Acuerdo humano final (decisiones bloqueadas de C1 y C2)

Fecha: 2026-10-13. Unidad: celda (tweet, etiqueta); 176 celdas
candidatas (HS 23, TR 87, AG 66); 127 tweets unicos. Las decisiones
finales quedan BLOQUEADAS; el gold NO se consulta en esta fase.

## 1. Recepcion y bloqueo

- C1: archivo completo (176 filas, 9 columnas sin text/nota inicial;
  2 cambios de decision: P93S1-0052, P93S1-0108).
- C2: archivo de CORRECCIONES (16 filas: 12 cambios + 4 casos que
  permanecen en UTD); las 160 celdas restantes conservan su respuesta
  inicial (reconstruccion documentada en PROVENANCE).
- Decisiones iniciales de la ronda 1 preservadas (verificadas contra los
  archivos archivados).

## 2. Distribuciones finales y UTD

| etiqueta | celdas | UTD C1 | UTD C2 |
|---|---|---|---|
| HS | 23 | 0 | 0 |
| TR | 87 | 0 | 2 |
| AG | 66 | 0 | 2 |

## 3. Acuerdo final por etiqueta (numeradores/denominadores)

| etiqueta | bruto 3cat | binario cond. | dice pos | dice neg | kappa 3cat | kappa bin | alpha cat | alpha falt |
|---|---|---|---|---|---|---|---|---|
| HS | 23/23 | 23/23 | 6/6 | 40/40 | 1.000 | 1.000 | 1.000 | 1.000 |
| TR | 85/87 | 85/85 | 158/158 | 12/12 | 0.847 | 1.000 | 0.847 | 1.000 |
| AG | 63/66 | 63/64 | 52/53 | 74/75 | 0.909 | 0.968 | 0.909 | 0.968 |

Matrices 3x3 finales (filas C1, columnas C2):

- HS: NN=20 NP=0 NU=0 | PN=0 PP=3 PU=0 | UN=0 UP=0 UU=0
- TR: NN=6 NP=0 NU=0 | PN=0 PP=79 PU=2 | UN=0 UP=0 UU=0
- AG: NN=37 NP=1 NU=2 | PN=0 PP=26 PU=0 | UN=0 UP=0 UU=0

## 4. Acuerdo conjunto

- Celdas con acuerdo exacto: 171/176.
- Tweets cuyas celdas candidatas coinciden todas: 122/127.
- Cambios de decision desde la ronda inicial: C1_HS=1, C1_TR=1, C2_AG=8, C2_HS=1, C2_TR=3

## 5. Clasificacion de resultados humanos

- FINAL_AGREEMENT_BINARY: 171 celdas.
- FINAL_DISAGREEMENT: 5 celdas.
- Cola de pendientes (FINAL_DISAGREEMENT + POLICY_AMBIGUITY): 5 celdas -> p93r4_pending_cases.csv (tercera revision
  propuesta; sin desempate automatico).

## 6. Cegamiento y procedencia (limitaciones)

- El cegamiento de la RONDA 1 sigue sin poder verificarse completamente
  (P93-R3: sin registros de acceso; manifest con recuentos agregados;
  carpeta compartida).
- La ronda 2 incluyo DELIBERADAMENTE las respuestas de A/B: las decisiones
  finales NO son independientes de esas respuestas; no se declara
  independencia estadistica entre rondas.
- El gold permanece oculto: no se afirma ninguna correccion de etiquetas
  ni error confirmado. La coincidencia C1-C2 no prueba que el gold sea
  incorrecto.
- La reconstruccion de C2 (160 celdas sin cambio) asume que el archivo de
  correcciones contiene TODOS los cambios; no hay evidencia de otros.
