# Comparacion de candidatos finales (con submissions experimentales)

Fecha: 2026-09-25. Fuente: `reports/final_candidates.csv` y
`reports/holdout_evaluaciones_v2.csv` (4/5 evaluaciones usadas: 2 de P7 y
2 de esta fase). Las submissions experimentales de los 3 candidatos se materializaron despues en `experiments/25_submissions_candidatos.py`.

## Tabla comparativa

| candidato | corrected CV | CV historico | holdout | F1 HS/TR/AG (holdout) | delta vs V03 (dev) | complementariedad | complejidad | evidencia | Submission file |
|---|---|---|---|---|---|---|---|---|---|
| FIN-01-C-REF-44 | 0.7874 | no medido (protocolo v2) | no evaluado | - | +0.0102 | ensemble_gain +0.0089 (IC [+0.0019,+0.0160]) vs M2 a 0.5; desacuerdo 0.139 | 44 features + 3 modelos binarios | PROMISING | no generada |
| FIN-02-C-EMO | 0.7809 | no medido (protocolo v2) | 0.8364 (por etiqueta {"HS": 0.8692, "TR": 0.84, "AG": 0.8}) | HS 0.869/TR 0.840/AG 0.800 | +0.0037 | ensemble_gain +0.0035 (IC [-0.0014,+0.0084]) vs M2 a 0.5; desacuerdo 0.111 | 6 features + 3 modelos binarios | no adoptado | toxic_submission_v09_candidate_a.csv |
| FIN-03-C-EMO-margenes | 0.7798 | no medido (protocolo v2) | no evaluado | - | +0.0026 | ensemble_gain +0.0051 (IC [-0.0018,+0.0120]) vs M2 a 0.5; desacuerdo 0.134 | 12 features + 3 modelos binarios | no adoptado | no generada |
| FIN-04-C-EMO-SYN | 0.7797 | no medido (protocolo v2) | 0.8336 (por etiqueta {"HS": 0.8558, "TR": 0.8533, "AG": 0.7917}) | HS 0.856/TR 0.853/AG 0.792 | +0.0025 | ensemble_gain +0.0043 (IC [-0.0004,+0.0096]) vs M2 a 0.5; desacuerdo 0.125 | 9 features + 3 modelos binarios | no adoptado | toxic_submission_v11_candidate_c.csv |
| FIN-05-C-SYN | 0.7793 | no medido (protocolo v2) | no evaluado | - | +0.0021 | ensemble_gain +0.0022 (IC [-0.0017,+0.0065]) vs M2 a 0.5; desacuerdo 0.081 | 6 features + 3 modelos binarios | no adoptado | toxic_submission_v10_candidate_b.csv |
| FIN-06-C-REF-44-CH | 0.7782 | no medido (protocolo v2) | no evaluado | - | +0.0010 | ensemble_gain +0.0096 (IC [+0.0022,+0.0172]) vs M2 a 0.5; desacuerdo 0.174 | 80 features + 3 modelos binarios | COMPLEMENTARY | no generada |
| FIN-07-C-EMO-SYN-TGT-CYT | 0.7777 | no medido (protocolo v2) | no evaluado | - | +0.0005 | ensemble_gain +0.0058 (IC [-0.0020,+0.0138]) vs M2 a 0.5; desacuerdo 0.146 | 15 features + 3 modelos binarios | no adoptado | no generada |
| V03 | 0.7772 | 0.7989 | 0.8285 (HS 0.8571 | TR 0.8493 | AG 0.7789) [P7-01] | - | +0.0000 | referencia | 3 modelos binarios (M2 union, 24.646 features TF-IDF) | referencia | no generada |
| META-BASES-RIDGE | 0.7874 | no medido (protocolo v2) | 0.8465 (HS 0.8624 | TR 0.8571 | AG 0.8200) [P7-02] | - | +0.0102 | ensemble_gain +0.0089 (IC [+0.0019,+0.0160]); LB v08 0.78163 (no transferido, D-020) | 44 features + 3 modelos binarios | adoptado en P6; el LB no supera V03 | no generada |

## Criterios de seleccion

- PROMISING: corrected CV >= 0.7772 + 0.005 = 0.7822 y mejora en >= 3/5
  pliegues frente a V03. Solo lo cumple `FIN-01-C-REF-44`, que es la
  reproduccion del meta v08 adoptado en P6 (ya evaluado en P7-02).
- COMPLEMENTARY: nivel <= referencia pero fusion con V03 con
  `ensemble_gain >= 0.0015` e IC bootstrap > 0. Lo cumple
  `FIN-06-C-REF-44-CH`, pero su fusion alcanza 0.7814 en el desarrollo,
  por debajo del meta v08 (0.7874), y el meta ya no transfirio al LB.
- El resto queda como evidencia negativa documentada.

## Decision del screening (sin submissions)

- **No hay candidato nuevo PROMISING**: el mejor candidato nuevo sobre
  probabilidades de V03 + canal es `FIN-02-C-EMO` (0.7809 en el desarrollo
  y 0.8364 en el holdout) frente a 0.7874 / 0.8465 del meta v08.
- Ninguno de los candidatos nuevos alcanza el nivel del meta adoptado y el
  meta adoptado ya fallo en el LB (D-020): se mantiene **V03** como modelo
  final y se recomienda **no gastar sondas** con los candidatos de canales.
- Las dos evaluaciones nuevas del holdout (`FIN-02-C-EMO`, `FIN-04-C-EMO-SYN`)
  quedan registradas; el holdout conserva 1 evaluacion libre (4/5 usadas).
  Regla de seleccion de finalistas: se eligio el mejor candidato de cada
  conjunto de canales distinto para no gastar dos evaluaciones en variantes
  del mismo candidato (por eso `FIN-03-C-EMO-margenes`, variante de `FIN-02`
  con +6 features, no se evaluo en el holdout).

## Materializacion de submissions experimentales (2026-09-25)

- C-EMO: `submissions/toxic_submission_v09_candidate_a.csv` (sha256 8e5a95d3bd6a...; positivos HS/TR/AG 462/351/427)
- C-SYN: `submissions/toxic_submission_v10_candidate_b.csv` (sha256 92caa41e379f...; positivos HS/TR/AG 460/351/432)
- C-EMO-SYN: `submissions/toxic_submission_v11_candidate_c.csv` (sha256 066d46ca12e9...; positivos HS/TR/AG 456/348/428)
- Artefactos experimentales sin sonda de LB: la decision documentada no cambia (V03 sigue siendo el modelo final y el screening no encontro evidencia para adoptar los candidatos).
- No se re-evaluo nada: las definiciones y los hiperparametros congelados salen de `reports/final_candidates.csv` y se verifico que reproducen su corrected_cv.
