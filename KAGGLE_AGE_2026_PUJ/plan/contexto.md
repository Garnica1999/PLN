# Contexto para el siguiente agente (V17)

## Donde esta cada entregable de P0
- `reports/v17_gap_audit.md` - informe de la auditoria de brechas (8 secciones, 42 familias inventariadas).
- `reports/v17_gap_audit.csv` - 9 columnas exactas, 1 fila de notacion + 42 filas de familia. Las fronteras
  se escriben `18-24__25-34` porque el guion vertical delimita columnas.
- No se genero ningun otro artefacto. No se toco V12, V15, V16, `experiments.csv`, `PROJECT_STATE.md`,
  `DECISIONS.md`, `RUNLOG.md` ni `FINAL_REPORT.md`: P0 es documental y la memoria se actualiza al cierre.

## Reconciliacion de la referencia (leer la seccion 2 del informe antes de nada)
- **Canonico para V17: 0,6005**, la escala *pooled* de `metricas_globales`
  (`scripts/v13_p1b_fusion_tokens.py:132-142`).
- **NO es un redondeo distinto.** Son dos estimadores: pooled 0,6005017473 frente a media-por-fold
  0,6004103057, diferencia 9,14e-5, mayor que la unidad de redondeo. Los dos reproducen exactamente y
  comparten el vector por fold `0,6007/0,6050/0,6032/0,6004/0,5928`.
- Motivo: el macro-F1 de la confusion acumulada no es la media de los macro-F1 por fold (efecto Jensen).
- Usar 0,6004 daria **+0,0001 regalado** en cada delta y rompe el emparejamiento con A2 (0,6011) y con
  el IC95 de `bootstrap_diferencia`, que tambien es pooled.
- Ojo: 0,6007 es la **accuracy** de V12, no la macro-F1. Y 0,62182 / 0,62188 son dos leaderboards de
  dos artefacts distintos con 93,88 % de acuerdo (D-036, cerrado, no es un problema abierto).

## Reparto de las familias
42 familias: 30 cerradas o vigentes, 3 abiertas de las que 2 son las nuevas (39, 40, 41) y 1 es la
reconciliacion. De las diez pistas: **cerradas 3** (P6 residual, P8 perfil de confianza, P10
decodificacion local), **parcialmente probadas 5** (P1, P2, P4, P7, P9), **genuinamente nuevas 2**
(P3, P5). P0 es esta fase. P0' esta a medias: la reconciliacion esta hecha, la reproduccion de A2 no.

## Las tres conclusiones que mandan
1. **La referencia es 0,6005 pooled.** Ver arriba.
2. **El especialista de frontera no fallo: fallo la representacion.** Medido en modo inerte sobre
   `features/oof_E028a_pares.npy`: el especialista sparse por par saca **0,6864** y **0,7218** dentro de
   `18-24|25-34` y `25-34|35-49`, frente a los **0,4963** y **0,4974** de V12 y a los **0,4944** y
   **0,4942** del especialista de V13 P3a sobre las 279 variables. Gap de **+0,19 y +0,22** sin usar.
   Y los 5 margenes de E-028a **ya son entradas de V12** (`MEMBER_FILES`), que sigue fallando el par.
3. **P2 tal como esta escrita en el brief seria repetir P3a.** Sobre las 279 variables es sinonimo
   (mismo protocolo anidado, resultado medido: peor que V12). La version nueva exige la representacion
   dispersa. Orden propuesto en la seccion 8.1: P0' -> P1.1 -> P2SPARSE -> P3.1 -> P5 -> P2.3.

## Lo siguiente
- **P0'**: reproducir A2 con `scripts/v16_arquitectura.py` sin editarlo. Si no reproduce, STOP.
- **P1.1** primero: es el diagnostico barato que decide si P2 y P3 se cierran sin entrenar.
- Empezar todo script con `autotest_numerico()` y usar `PARES` de
  `scripts/v13_p1b_fusion_tokens.py:76` para las etiquetas de frontera (ver riesgos de la seccion 7).

## No verificado (marcado en el informe, no suponer)
- La parte (b) de D-019 (enrutamiento por margen del par-SVM con tau) **no es reproducible**: ni el
  script ni el JSON contienen ese codigo. Solo el resumen 0,0000 consta en dos informes.
- El brief de V17 completo (53 secciones) **no esta en el repositorio**. P6 a P10 solo se auditaron
  contra su nombre de una linea en `plan/00_intent.md` seccion 4.
- `reports/tables/e028_combinaciones.csv` se cita pero **no existe**.
