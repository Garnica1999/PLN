# V11-09: cierre de entregables

Generado por `experiments/v11_09_cierre_entregables.py` (V11-09 (cierre de entregables)).

## 1. Que faltaba y por que

Al cerrar la fase con V11-08, siete de los once entregables pedidos tinham contenido
real. **Cuatro no se produjeron, y no por olvido.**

| entregable | linea | tipo | motivo del vacio |
| --- | --- | --- | --- |
| v11_gap_audit.md | P0 | documento | La auditoria de brechas SI se ejecuto: es V11-00, con 25 de 25 comprobaciones superadas. Este documento es un puntero a su informe canonico, no una co |
| v11_calibration.csv | P4 | tabla | La linea de calibracion ya estaba medida y cerrada en v10 con V10-04, que dio NO ACCIONABLE porque la calibracion mejora el ECE de 0.0583 a 0.0174 y n |
| v11_threshold_stability.csv | P8 | tabla | La parametrizacion estable de umbrales ya estaba medida y cerrada en v10 con V10-06, que no supero su gate: no hay parametrizacion a la vez igual de b |
| v11_error_intersection.csv | transversal | tabla | La interseccion de errores es una medida entre DOS o mas candidatos, y en v11 solo hay uno vivo, V03. La interseccion de un conjunto consigo mismo es  |

El patron es el mismo en tres de los cuatro: **la linea correspondiente ya estaba
medida y cerrada en v10**, de modo que en v11 no se ejecuto ningun experimento propio y
no hay cifras nuevas que escribir.

El cuarto es distinto: la interseccion de errores es una medida **entre dos o mas
candidatos**, y en v11 solo hay uno vivo, V03. La interseccion de un conjunto consigo
mismo es el propio conjunto, luego la tabla seria trivial.

## 2. La regla aplicada

**Un entregable ausente se confunde con uno olvidado. Se escribe con una fila que documenta el motivo del vacio, porque documentar una decision es mas honesto que callarla.**

Por eso los tres CSV se escriben con cabecera y **una fila que documenta el motivo**,
y la auditoria de brechas se escribe como **puntero** a su informe canonico.

## 3. Lo que no se hizo, y por que

**No se copiaron cifras de v10 a tablas de v11. Seria inventar trazabilidad: el lector veria resultados de v10 etiquetados como de v11, sin forma de distinguirlo al abrir el fichero.**

Ese es el motivo de que los tres CSV **no tengan cifras**: vacios con el motivo
escrito es mas honesto que llenos con numeros que no son de aqui.

## 4. Estado final de los once entregables

**11 de
11 presentes.** Ausentes: ninguno.

Fecha: 2026-09-26T22:09:00+00:00. Duracion: 0.0 s.
