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

- [x] **P2, razon logaritmica de recuentos: cerrado, negativo.** -0,0585 con suavizado
      pequeno y +0,0047 con el grande; el control de conteos da -0,075. Filtro falla en las
      tres. Sin submission y sin sonda. Precedente adverso de E2-15 confirmado.
- [x] Premisa propia corregida antes de ejecutar: el ratio por etiqueta ya estaba probado.
- [x] No se barre el suavizado, porque el resultado positivo es un punto de una superficie
      sin explorar y no una senal.

## Siguiente

- [ ] **Decidir si P4 a P7 merece ejecutarse.** Las cuatro mueven la capa de decision en la
      banda de cien a doscientas celdas, donde el filtro de detectabilidad exige mas del 60 %
      de acierto. P4 es la unica con mecanismo propio, pero su techo lo acotan tres
      predecesores fallidos: cadenas, powerset y apilamiento.
- [ ] Si se ejecuta alguna, aplicar el filtro de detectabilidad **antes** de cada envio.
