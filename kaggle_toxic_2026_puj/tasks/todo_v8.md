# Tareas de la fase v8

## Hecho

- [x] Worktree y rama aislados, `data/` enlazado, `tools/` copiado, verificador a 0 errores.
- [x] P0: auditoria programatica de las diez pistas y de las veintidos tecnicas del material
      de clase. Nueve tecnicas genuinamente nuevas. Tres falsos positivos propios corregidos.
- [x] P1: cinco modelos, identidad de pipeline verificada celda a celda (0.000e+00).
- [x] P1: fichero escrito y verificado con nueve comprobaciones de integridad.
- [x] **P1 enviado y medido: 0.77470, diferencia -0.00836. Cerrado sin variantes.**
- [x] A2: analisis de la resolucion del leaderboard. Ruido 0.01524, equivalencia a 19.2
      aciertos netos, diez de once sondas indistinguibles.
- [x] A2: premisa propia revisada. El movimiento positivo de v02 a v03 fue ruido.
- [x] A2: filtro de detectabilidad adoptado, estimable antes de enviar.
- [x] `experiments.csv` llega a 191 filas. D-057, D-058 y D-059 escritas.

- [x] **P3, estabilidad de features: negativo y monotono.** -0,0033, -0,0046 y -0,0189. El
      filtro de detectabilidad falla en los tres. Sin submission y sin sonda.
- [x] Conclusion registrada: el criterio mide forma del texto, no pertenencia a una clase.

## Siguiente

- [ ] **P2, ponderacion por razon logaritmica de recuentos**, con el precedente de E2-15 ya
      medido entre -0,016 y -0,039. Es el estimador de Mosteller y Wallace de 1964.
- [ ] Aplicar el filtro de detectabilidad **antes** de considerar cualquier envio, siempre.
- [ ] P4 a P7 solo si el filtro de detectabilidad dice que su cambio es resoluble.
