# V11: auditoria de brechas (P0)

Generado por `experiments/v11_09_cierre_entregables.py` (V11-09 (cierre de entregables)).

## 1. Este documento es un puntero, y hay que decir por que

La auditoria de brechas **si se ejecuto**: es **V11-00**, con 25 de 25 comprobaciones superadas. Su informe canonico es
**`reports/v11_00_reconciliacion.md`**, con las cinco
secciones de detalle, y su registro estructurado esta en
`reports/v11_00_reconciliacion.json`.

Este documento existe con el nombre que pidio el plan, y **no copia el contenido**: dos
copias de la misma observacion divergen, y cuando divergen ya no se sabe cual es la
verdadera. El informe de V11-00 es la fuente; este solo indica donde esta.

## 2. Que preguntaba la auditoria

Antes de medir nada nuevo, comprobar que **las tres referencias de las que depende la
v11 son coherentes y reproducibles**: la metrica corregida de V03, su huella de
submission, sus umbrales congelados, la integridad del holdout2, y las cifras de
desplazamiento de prevalencia que V9-02 reporto y que la v11 iba a dar por buenas.

## 3. Los tres hallazgos, y que hacer con cada uno

### Hallazgo 1, propio y sin consecuencia: la comprobacion buscaba la misma clave

buscaba la clave en el informe y en el registro, y en dos experimentos no se llaman igual: en V10-05 es `gate_a` en el registro y `gate_a_superado` en el informe, y en V10-10 el informe anida el veredicto en `gate_b_de_mejora.superado`. Se corrigio la comprobacion, no el dato.

### Hallazgo 2, propio y con consecuencia real: el registro retrocedio

el commit de v10 y el merge con el que se trajo a v11 **retrocedieron el `experiments.csv`** y las cuatro filas V11 se perdieron del registro. V11-05 lo resuelve derivando las filas de los informes JSON en lugar de teclearlas, con la regla de que una clave ausente queda vacia y no rellenada.

### Hallazgo 3, de un artefacto historico: el nombre del campo no significa lo que dice

el campo `factor_vs_train` de V9-02 usa como denominador la prevalencia del **desarrollo** de 1483 filas y no la del **train completo** de 1731, porque el codigo define `prevalencia_train = y_dev.mean(axis=0)`. La diferencia es de 0.0002, pequena, pero es el tipo de detalle que hace fallar una cadena de resultados. **V9-02 es internamente consistente y no tiene defecto; lo que no significa lo que dice es el nombre del campo.** Ver D-079.

## 4. Lo que la auditoria NO pudo comprobar

**No pudo comprobar la transferencia a la prueba real**, y no es un hueco de la
auditoria: es una limitacion de las reglas, que V11-02 clasifico como ambigua y NIVEL 2
elevo a prohibida. Auditar si el mecanismo mejora en esta prueba concreta exigiria
usar la prueba, que esta prohibido. La pregunta queda sin respuesta verificable, y se
dice aqui para que no se lea mas de lo que vale.

## 5. Veredicto

Las tres referencias son coherentes y reproducibles. La auditoria **no encontro ningun
defecto en V03 ni en el holdout2**: la huella de V03 es identica a la congelada y el
holdout2 conserva su integridad con sus tres evaluaciones sin gastar. El unico
hallazgo de fondo es el nombre del campo de V9-02, que se documento y serodea usando la
prevalencia absoluta en lugar del factor.

Fecha: 2026-09-26T22:09:00+00:00. Duracion: 0.0 s.
