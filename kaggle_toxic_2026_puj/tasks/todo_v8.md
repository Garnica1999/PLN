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

- [x] **P4, regularizacion conjunta: cerrado, negativo.** -0,0015 y -0,0068, con el mecanismo
      funcionando: el coseno entre cabeceras sube de 0,738 a 0,922. Compartir estructura entre
      las tres etiquetas es posible y resta.
- [x] Identidad de la solucion confirmada por el valor del objetivo, no por el coseno.
- [x] **Patron de fase documentado**: P1, P3 y P4 mueven la decision hacia peor en las filas
      que tocan, con tasa de acierto de 0,46, de 0,39 a 0,44 y de 0,42 a 0,39.

## Siguiente, y es una decision del usuario

- [ ] **P5, P6 y P7 quedan sin razon para ejecutarse.** Las tres solo mueven la capa de
      decision en la banda de cien a doscientas celdas, donde el filtro exige mas del 60 % de
      acierto, y las tres lineas medidas en esa banda han dado tasa por debajo del azar. La
      decision es cerrar la fase o ejecutarlas, y conviene tomarla con el dato encima.
