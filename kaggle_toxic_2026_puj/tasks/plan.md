# PLAN v2 - KAGGLE_TOXIC_2026_PUJ (Advanced Research)

Aprobado por Carlos el 2026-09-25. Objetivo: encontrar una fuente de senal nueva
que supere el LB 0.78306 de V03, con evidencia experimental por encima del
numero aislado.

## Reglas del proyecto

- V03 congelada en `experiments/baseline_v03_frozen/`; no se modifica. Todos los
  experimentos nuevos se comparan contra ella.
- Protocolo de validacion v2 (D-016):
  - Grupos = near-duplicado (Fase 0) UNION pseudo-autor (primera mencion).
  - Holdout congelado: `StratifiedGroupKFold(7)` sobre los grupos; 1 pliegue es
    holdout (~14 %), 6/7 es desarrollo. Se congela ANTES de mirar resultados;
    maximo 5 evaluaciones al holdout en todo el proyecto, solo finalistas.
  - `corrected_cv` (desarrollo, grupos fusionados) es la metrica de seleccion;
    `historical_cv` se reporta para continuidad. Baseline de referencia en el
    desarrollo: la propia V03 reevaluada.
  - Contingencia: si el corrected_cv tiene std entre pliegues > 0.025, pasar a CV
    repetida (3 semillas x 5 pliegues); si aun no discrimina, holdout al 20 %.
- Regla de adopcion: corrected_cv >= +0.005 y mejora en >= 3/5 pliegues; o bien
  complementariedad demostrada (`ensemble_gain` >= 0.0015 con IC bootstrap > 0).
- Todo OOF; prohibido usar test para fit/vocabulario/seleccion/umbrales.
- Presupuesto LB: maximo 3 sondas, solo finalistas, cada una con hipotesis.
- Entregables academicos (notebook y FINAL_REPORT) se actualizan si un candidato
  supera a V03.

## Paquetes y gates

| Paquete | Contenido | Gate de salida |
|---|---|---|
| P0 | Congelar V03; holdout v2; OOF corregido de bases; gap analysis v2; backlog v2; D-016 | Gate P0 publicado con numeros |
| P1 | Label dependency suave (cadenas en 3 ordenes) + meta-modelo regularizado | corrected_cv o complementariedad |
| P2 | Target-insult distance; target profiling; colectivo + agresion (FN de AG) | idem + mejora en rebanadas |
| P3 | POS/dependencias/morfologia como representacion SPARSE | idem |
| P4 | Cita / atribucion / negacion (H13), canales autor vs citado | idem |
| P5 | Emocion H5 + lexicones documentados + NB-SVM + multicanal ortografico | idem |
| P6 | Late fusion / stacking v2 + bootstrap por tweet | adopcion pre-registrada |
| P7 | Finalistas -> holdout -> auditoria -> 2-3 sondas LB -> entregables | GO/NO-GO |

## Lineas cerradas (no repetir sin hipotesis nueva)

Reduccion de capacidad (v06/v07 peores), ensembles indiscriminados (v04
0.77510), Optuna masivo (v05: OOF 0.8039 -> LB 0.77602), post-proceso duro
(-0.0056), DL secuencial desde cero (0.7949/0.7410/0.7364), concatenacion densa
de features (-0.009 a -0.032).

## Criterios de cierre de linea

Una linea se cierra cuando la hipotesis fue probada, las variantes razonables
fueron evaluadas, no hay mejora y el resultado es estable (documentado en
RUNLOG + experiments.csv + DECISIONS).
