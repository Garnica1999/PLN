# P93-R - Auditoria de cegamiento del paquete de ronda 1

Fecha: 2026-10-13. Resultado: LIMPIO.

## 1. Comprobaciones ejecutadas

- Escaneo de tokens prohibidos en los archivos visibles (ids de origen,
  referencias a A/B, gold, candidatos, discrepancias, OOF, V03, umbrales,
  folds, componente, holdout): sin hallazgos.
- Columnas del CSV exactas y celdas no textuales restringidas (vacias o v2).
- Anotaciones vacias en las tres etiquetas y en las dos columnas de notas.
- Orden de filas = permutacion registrada (semilla 20261027); ids P93R-* asignados con semilla 20261026.
- Sin clave interna ni labels originales en el paquete compartible.
- Guia v2.1 identica al original (hash verificado en el manifiesto).

## 2. Lo que la ronda 1 NO revela

- Que los tweets fueron seleccionados por coincidencia A/B distinta del gold.
- Las respuestas y notas de A/B (visibles solo en la ronda 2, tras entregar C).
- El gold original (oculto en ambas rondas).
