# Auditoria V1-V37 · Fase 4 - Reconciliacion del notebook consolidado

## Que se hizo

Anadido un capitulo **PARTE III - AUDITORIA DEL HISTORIAL V1-V37** al final de
`notebooks/99_entrega_consolidado.ipynb`, sin tocar ninguna celda existente.

| Metrica | Antes | Despues |
|---|---|---|
| Celdas totales | 326 | **330** (+4) |
| Markdown / codigo | 163 / 163 | 165 / 165 |
| `nbformat` | 4.5 | 4.5 (sin cambio) |
| Errores guardados | 0 | 0 |
| IDs unicos | si | si |

**Verificado**: las 326 celdas originales son **byte-idénticas** a las de `git`. Solo se actualizó el
título de los metadatos para reflejar que la entrega incluye la auditoría.

## Las 4 celdas añadidas

1. **markdown** (109 líneas): qué se auditó, con qué restricciones, la tabla de los 13 hallazgos con su
   severidad y estado, y dos secciones de detalle para los dos hallazgos que cambian una conclusión
   publicada (K-01 y A-01). Incluye una sección explícita de **qué NO encontró** la auditoría.
2. **código**: lee `experiments/auditoria_v1v37_inventario.json` y muestra el recuento por tipo de
   registro y por familia. **Ninguna cifra de la auditoría está escrita a mano.**
3. **markdown** (15 líneas): dónde está el detalle, y cómo regenerar el inventario.
4. **código**: comprueba las correcciones aplicadas contra el estado actual de `experiments.csv`.

La celda de código se **ejecutó de verdad** y produce:

```
filas del registro            : 197
integridad de la evidencia    : 0 filas con artefacto ausente
correcciones aplicadas        : 197 x 22, ninguna fila descuadrada
V37-DIAG-POS duracion_seg     : 89.5 (reconstruido del JSON)
V37-DIAG-POS accuracy_mean    : '' (vacio = correcto)
```

## Deuda documental que queda declarada

El notebook llega hasta **V35**. Las fases **V36** (cierre de L4, D-088) y **V37** (diagnóstico
`P(POS|edad)`, D-089) están documentadas en `reports/v36_e1_*.md` y `reports/v37_diagnostico_pos.md`
y en el acta D-090, pero **no tienen sección propia en el notebook**.

Queda registrado como pendiente de reconciliación, y **no** se ha escrito aquí porque el brief de la
auditoría prohibía alterar el contenido histórico en silencio: escribir esas dos secciones exige
decidir si se resumen o se reproducen, y eso es una decisión de Carlos.

## Script y su autoejercicio

`scripts/auditoria_v1v37_notebook.py`, **ASCII puro**, ruff limpio, `--verificar` con **5/5**.

El autoejercicio detectó y corrigió **4 defectos reales** durante la escritura:

1. **249 bytes no-ASCII** en el script: el markdown con tildes estaba embebido en el código. Se movió a
   `plan/auditoria/notebook_capitulos.md` (que sí admite tildes) y el script lo lee. El título del
   notebook se compara con una versión normalizada para no reintroducir el problema.
2. **La cabecera del `.md` se contaba como un tercer capítulo**, así que se insertaban 6 celdas en vez
   de 4. Se descarta todo lo anterior al primer separador.
3. **La idempotencia fallaba**: el escaneo comparaba la celda *completa* contra un marcador de *una
   línea*, luego nunca encontraba el bloque y las celdas crecían 4 por pasada (326 → 338). Ahora
   compara la primera línea, y retira la celda de código emparejada junto a su markdown.
4. **Dos checks mal escritos** que medían lo que no decían: uno comparaba contra la aplicación
   anterior, otro contaba capítulos buscando un texto que el capítulo visible no contiene.

La lección de D-086 vuelve a funcionar: **el autoejercicio detecto 4 defectos antes de tocar el notebook
real**. El notebook quedó intacto hasta que los 5 checks pasaron.

## Estado

Fase 4 **cerrada**. El notebook consolidado tiene su capítulo de auditoría, con las cifras leídas del
artefacto congelado y no escritas a mano.
