# P102 - Resultados del sprint (triage de 18 ideas)

Fecha: 2026-10-14. Pre-registro: `reports/p102_preregistro.md`.
Gate A3 (unico criterio de aceptacion): delta canonico >= +0.003; >= 4/5
pliegues; IC95 de bootstrap por clusters excluyendo cero; delta positivo en la
particion 47; >= 200 celdas cambiadas vs V03. Umbrales V03 congelados.
Sin test; sin holdouts; sin leaderboard; sin submissions.

## 1. Resultados

| Item | Que probo | Mejor resultado | Veredicto |
|---|---|---|---|
| P102-01 (D4) | variantes char sin 'wb' (con/sin case) sobre el arnes REP | V2 triple: +0.004466 (3/5; IC [-0.001664,+0.011226]; 91 celdas; 47 +0.002446) | GATE_FAILED (senal marginal) |
| P102-02 (C1-corner) | promedio V03+LinearSVC+ComplementNB con umbrales recalibrados | 0.773370 (-0.003841; IC [-0.011717,+0.004407]; 47 -0.009448) | GATE_FAILED |
| P102-03 (B2-corner) | regla dura TR=0 si HS=0, aislada | -0.000646 canonica; -0.004097 en 47 (20/21 violaciones) | GATE_FAILED |
| P102-04 | micro-barrido de umbrales sobre el OOF congelado | argmax canonico +0.003895 (IC incluye cero; 83 celdas); 47 -0.002002 | GATE_FAILED |
| P102-05 (D1-corner) | emojis como tokens | sin material: texto oficial 100% ASCII (0/1731 filas no-ASCII) | NO_MATERIAL |

## 2. Notas metodologicas (lo aprendido, con controles)

- **Harness validado**: en P102-01 y P102-02 el control M2 reprodujo el OOF
  congelado con diferencia exacta 0.0 (canonica y particion 47), incluida la
  re-creacion de folds por hash. Los resultados de las variantes son
  confiables dentro de la maquinaria del proyecto.
- **Contraccion de promedios, confirmada y corregida**: el promedio con
  umbrales congelados de V03 da 0.743469 (perdida ~-0.034); recalibrar los
  umbrales tras el promedio recupera +0.0299 (0.773370). El mecanismo que
  v08_p1 midio queda cuantificado; aun asi el promedio no alcanza a V03
  (el miembro ComplementNB, 0.6866, y el SVC, 0.7715, arrastran).
- **Regla dura TR**: con la invariante exacta en train (TR=1 & HS=0 = 0
  casos), aplicarla a V03 perjudica porque destruye celdas TR correctas en
  filas donde HS es falso negativo (20 celdas, todas con dano neto).
- **Umbrales**: el optimo local por etiqueta difiere entre particiones
  (canonica: HS 0.410/TR 0.475/AG 0.450; particion 47: HS 0.365/TR 0.505/
  AG 0.490) -> no hay punto estable distinto de V03.

## 3. Conclusion del sprint

- **Sin candidato**: los 4 items ejecutados fallan el gate A3 y el quinto no
  tiene material. Coherente con la banda historica del proyecto (deltas
  locales < +0.02 no transfieren; los tres "casi" del sprint mueren en la
  particion 47 o en IC que incluye cero).
- **V03 permanece congelado** como referencia. Recomendacion de slots del
  leaderboard privado sin cambios: **V03 + `v08_meta_v2`** (mejor no-V03
  enviada; 0.78163). Si una futura candidata pasara el gate A3 completa,
  sustituiria el slot 2.
- Sin rescates ni barridos de consuelo; sprint cerrado.

## 4. Artefactos

- Scripts: `experiments/p102_01_char_variantes.py`, `p102_02_promedio_
  recalibrado.py`, `p102_03_regla_dura_tr.py`, `p102_04_barrido_umbrales.py`,
  `p102_90_registrar.py`.
- Resultados: `reports/p102_01_char_variantes.{json,md}`,
  `p102_02_promedio_recalibrado.{json,md}`, `p102_03_regla_dura_tr.{json,md}`,
  `p102_04_barrido_umbrales.{json,md}`, `p102_preregistro.md`.
- Registro: filas P102-01..04 en `experiments.csv` (496 filas).
- Memoria: episodios P102 en Graphiti (grupo `main`).
