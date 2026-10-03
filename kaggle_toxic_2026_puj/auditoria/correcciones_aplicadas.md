# Correcciones aplicadas por la auditoría integral (2026-10-02)

Documento de cierre de la auditoría. Cada corrección indica el hallazgo de
origen, la acción aplicada y la evidencia. Sin commit: queda a revisión de
Carlos.

## 1. Registro de experimentos (`experiments.csv`)

- C1. Reparadas las 43 filas malformadas (P67 a P77) que partían el CSV por
  comas sin comillas. El archivo pasa de 411 a 450 filas, todas de 23
  columnas, sin ids duplicados y sin filas malformadas.
- C2. Añadidas 35 filas para experimentos que no tenían fila propia,
  incluidos P78, P82 y P83 (con contenido curado) y, además, P79, P80, P81 y
  P82-02. Los ayudantes (integración, cierre, auditoría, submission, registro)
  quedan clasificados en el inventario; su ausencia de fila propia es de
  diseño.
- C3. Etiquetas corregidas: la fila P69-01 declara que su "reproducción R3" es
  re-scoring trazable sobre el OOF histórico, no reentrenamiento; la fila
  P72-03 declara que su delta es contra el MÁXIMO de las nulas y no contra la
  media (A queda por encima de la media y solo 1 de 10 nulas lo supera).

## 2. Uso del holdout

- C4. Registrado el sexto uso no declarado del holdout v2
  (`experiments/v33_01_validacion.py`, 2026-09-27, mismo valor que P7-01) en
  `reports/holdout_evaluaciones_v2.csv`, con nota en `config/holdout_v2.json`.
  El consumo real es 6 de 5 y no se añaden más evaluaciones.

## 3. Codificación

- C5. `submissions/leaderboard_log.csv` convertido de cp1252 a UTF-8; pandas y
  los scripts que lo leen con UTF-8 funcionan. Las seis celdas del notebook
  que lo leían con cp1252 pasan a utf-8-sig. El test oficial sigue en cp1252
  y no se toca.

## 4. Erratas en informes congelados

- C6. Añadida fe de erratas al informe vigente de P82 (A7 contaminada; el
  determinismo bit a bit no queda verificado; H3 sobre E*) y al informe de la
  sonda (el envío sí ocurrió; manda el registro del leaderboard, LB 0.68150).
  Los informes originales no se reescriben.

## 5. Notebook consolidado

- C7. Secciones 122, 124 y 125 actualizadas con las cifras corregidas; nueva
  sección 126 con las correcciones aplicadas y las erratas; lecturas del
  registro en UTF-8. El notebook compila celda a celda.

## 6. Lo que NO se corrige y por qué

- El código de experimentos cerrados no se reescribe: la trazabilidad exige
  que cada script siga siendo el que produjo su informe. Las correcciones
  sobre ellos van como errata, registro o nota. Si se quiere un A7 correcto en
  P82, es un experimento nuevo con su re-ejecución.
- Los 154 hallazgos D1 del escáner (funciones sin llamador real) quedan
  clasificados; los confirmados de P83 ya se conectaron. Reescribir 376
  scripts históricos no es viable ni deseable.
- Las transversales A-D de P73-P77 (hiperparámetros contra pre-registro,
  early stopping, fugas, cruce del escáner) siguen declaradas como no
  ejecutadas.
- La ejecución end-to-end del notebook completo no se ha corrido (pesada y
  con artefactos ausentes preexistentes).

## 7. Reejecuciones de experimentos (2026-10-02)

- **P69.** La cadena completa de entrenamiento v4 se reejecuto (27
  label_consistency, 28 hardness, 29 reweighting; 2 s, 2 s y 60 s) y los tres
  artefactos (`label_consistency_v4.npz`, `hardness_v4.npz`,
  `oof_weighting_v4.npz`) quedaron **byte a byte identicos**. Despues se
  reejecuto `p69_01` con la puerta de cumplimiento corregida: la puerta
  excluye ahora el registro del holdout (`holdout_evaluaciones`), su lista de
  identificadores (`holdout_v2_ids`, que solo sirve para excluirlo) y el
  generador de submission `p69_10` (posterior a la corrida original, con su
  propio informe). Resultado: 10 de 10 en cumplimiento, nivel R3 y los mismos
  numeros (0.7854160387; delta +0.0082045862; 5 de 5 pliegues). La afirmacion
  "reproduccion R3" queda ahora verificada CON reentrenamiento, no solo con
  re-scoring.
- **P78.** Errata anadida al addendum: el rango real de E* es 8-17 (el JSON
  declaraba 10-17) y la trayectoria efectiva del learning rate difiere por
  brazo.
- **P82.** El A7 se corrigio para comparar solo posiciones finitas (antes el
  maximo sobre NaN daba NaN y la igualdad se declaraba con `equal_nan` sobre
  mascaras). La corrida completa se reejecuto (marca
  `20261002T132340Z_p563891`, EXIT 0, ~96 minutos) y da **23 de 23 CONFIRMADO**:
  A7 compara 891 posiciones finitas con variacion maxima 0.0 y patron de NaN
  identico, y las medias de los cinco brazos coinciden exactamente con las
  almacenadas (B1 0.639659, B2 0.637554, B3 0.615163, B4 0.617194, B5
  0.560747). El informe vigente es
  `reports/p82_rnn_atencion_20261002T132340Z_p563891.json`; el anterior se
  conserva con su fe de erratas.
- **Escaner estatico e inventario.** Ambos se re-ejecutaron sobre el codigo
  corregido. El escaner excluye ahora el patron visitante de `ast.NodeVisitor`
  (los `visit_*` se despachan por nombre y no son funciones desconectadas):
  D1 baja de 152 a 146 y las ALTA de 56 a 50. El inventario se mantiene en 483
  experimentos, 376 scripts, 0 malformadas y 33 ayudantes sin fila propia.
