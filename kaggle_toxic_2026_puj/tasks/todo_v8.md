# Tareas de la fase v8

## Hecho

- [x] Worktree y rama aislados, `data/` enlazado, `tools/` copiado, verificador a 0 errores.
- [x] P0: auditoria programatica de las diez pistas y de las veintidos tecnicas del material
      de clase. Nueve tecnicas genuinamente nuevas. Tres falsos positivos propios corregidos.
- [x] P0: registrar que P2 es el estimador de Mosteller y Wallace y llega con precedente
      adverso, y que P3 tiene respaldo doble.
- [x] D-057 y D-058 escritas. `experiments.csv` llega a 190 filas.
- [x] P1: cinco modelos, identidad de pipeline verificada celda a celda (0.000e+00).
- [x] P1: nueve comprobaciones de integridad del fichero, todas correctas.
- [x] P1: fiche `submissions/toxic_submission_v08_p1_fold_ensemble.csv` escrita y verificada.

## Bloqueado a la espera del usuario

- [ ] **Enviar la fiche de P1 y pasar el numero del leaderboard.** Parada obligatoria: no se
      crean variantes de umbral, de pesos ni de semilla antes de conocerlo.

## Siguiente, segun el resultado

- [ ] Si P1 mejora: P9 (desacuerdo entre pliegues) y P10 (segunda semilla), como maximo.
- [ ] Si P1 no mejora: cerrar P1 sin variantes y ejecutar P3 (estabilidad de features).
- [ ] Despues de P3: P2 (log-count ratio), con el precedente de E2-15 ya medido.
