# P103 - RONDA 2 (N5 + N6): resultados del sprint

Fecha: 2026-10-14. Pre-registro: `reports/p103_preregistro.md` (gate A3;
control V03 exacto en ambas particiones; piso de deteccion declarado;
presupuesto 3 h CPU). Alcance confirmado: SOLO N5 y N6; N1, N2 y N8-A/B/C
descartados (datos externos no permitidos por el profesor; N2 no se consulta
por delta esperado ~0 y riesgo de reglas).

## 1. Resultados

| Item | Que probo | Canonica | Particion 47 | Veredicto |
|---|---|---|---|---|
| P103-01 (N5) | espacio TF-IDF por etiqueta (20 configs) con seleccion anidada | +0.002389 (4/5 pliegues; 64 celdas; IC95 [-0.002405,+0.007250]; piso deteccion +0.004828) | +0.001362 (114 celdas) | GATE_FAILED; sin senal detectable |
| P103-02 (N6) | word dropout (p 0.05-0.15; 1-2 copias) con seleccion anidada | -0.006873 (1/5; 173 celdas; IC95 [-0.015445,+0.001793]) | -0.004265 (178 celdas) | GATE_FAILED; el dropout perjudica |

Control en los dos experimentos: reproduce V03 exacto en la canonica
(diff proba 0.00e+00; ccv 0.7772114525165721) y en la particion 47 (0 celdas
distintas; diff proba 0.00e+00).

## 2. P103-01 (N5) - detalle

- Malla pre-registrada de 20 configuraciones por etiqueta; seleccion anidada
  (CV interna de 3 pliegues; criterio F1 de la etiqueta @0.5).
- Direccion observada (NO adoptada; por debajo del piso de deteccion):
  HS prefiere max_df 0.9 y word (1,3) (3/5 pliegues); AG prefiere char (2,5)
  (3/5); TR inestable (4 configuraciones distintas en 5 pliegues).
- El delta puntual (+0.002389) queda por debajo del piso de deteccion
  (+0.004828) y el IC95 incluye cero (pct_mejora bootstrap 0.841): se reporta
  como "sin senal detectable", sin forzar conclusion.

## 3. P103-02 (N6) - detalle

- Malla corta pre-registrada (6 configuraciones); aumento solo del train del
  pliegue; copia vacia conserva el original; semillas deterministas.
- Seleccion por pliegue inestable (p/copias distintos en cada pliegue) y
  resultado negativo en ambas particiones: el dropout de tokens perjudica a
  la representacion TF-IDF + LR. Coherente con V6-08 (aumento negativo),
  V47-01 (TRAIN_AUGMENTATION = CLOSED) y P94-R16 (EDA ALREADY_TESTED).

## 4. Poder estadistico y honestidad

- se del bootstrap por clusters: 0.002463 (N5) y 0.004398 (N6); deltas
  minimos detectables al 95%: +0.004828 y +0.008619 respectivamente.
- Ningun efecto esperado de N5/N6 alcanzaba el piso; ambos se reportan como
  "sin senal detectable" (IC95 incluye cero en los dos).

## 5. Cierre

- Sin candidato: los dos items fallan el gate A3 completo.
- V03 permanece congelado (corrected_cv 0.7772114525165721; LB 0.78306).
- Recomendacion de slots del leaderboard privado SIN CAMBIOS: V03 +
  v08_meta_v2 (mejor no-V03 enviada, LB 0.78163). Ninguna candidata nueva
  pasa el gate.
- Presupuesto: ~3.5 min de CPU de 3 h (P103-01 105.4 s; P103-02 104.4 s).
- Sin test; sin holdouts; sin leaderboard; sin submissions; artefactos
  historicos intactos.

## 6. Artefactos

- Scripts: `experiments/p103_01_vectorizadores_por_etiqueta.py`,
  `p103_02_word_dropout.py`, `p103_90_registrar.py`.
- Resultados: `reports/p103_01_vectorizadores_por_etiqueta.{json,md}`,
  `p103_02_word_dropout.{json,md}`, `p103_preregistro.md`.
- Registro: filas P103-01/02 en `experiments.csv` (498 filas).
