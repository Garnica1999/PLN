# Auditoria V1-V37 · Fase 1 - Inventario congelado

## Scripts y artefactos de esta fase

- `scripts/auditoria_v1v37_inventario.py` (nuevo; no modifica ningun script de experimento).
- `reports/tables/auditoria_v1v37_inventario.csv` (195 filas x 24 columnas).
- `reports/tables/auditoria_v1v37_huecos.csv` (0 filas).
- `experiments/auditoria_v1v37_inventario.json`.
- `plan/auditoria/00_entorno_y_alcance.md`.
- `config/hardware.json`.

Modo `--verificar`: **6/6 caminos** en 0,1 s, no escribe artefactos reales. `ruff`: limpio. ASCII puro.

## Resultado del inventario

**195 filas no equivalen a 195 experimentos.** La clasificacion por evidencia da:

| Tipo de registro | Filas | Que es |
|---|---|---|
| `EJECUTADO` | **148** | hubo corrida; 113 con macro-F1 en el CSV y 35 cerradas por gate/verbal |
| `DOCUMENTAL` | 21 | auditoria, informe, cierre, compliance, submission; no entrena |
| `PREREGISTRO` | 15 | pre-registro congelado, sin ejecucion |
| `ENMIENDA` | 10 | enmienda pre-ejecucion declarada, sin ejecucion propia |
| `BLOQUEADO` | 1 | `V35-E1-INTENTO-1`, `UnboundLocalError`, evidencia en cuarentena |

Documentos = `DOCUMENTAL` + `PREREGISTRO` + `ENMIENDA` = **46**. Corridas reales = 148 + 1 abortada.

## Distribucion por familia

| Familia | Filas |
|---|---|
| V1-V4 fundacional (E-*, B*, ADV-*) | 37 |
| V5-V12 fases PH* | 76 |
| V13-V17 busqueda residual | 13 |
| V18-V23 frontera y ordinal | 12 |
| V24-V29 seleccion y ensemble | 15 |
| V30-V31 campeon y tracks | 14 |
| V32-V33 reapertura y FTRL | 10 |
| V34-V36 falsacion FTRL y L4 | 16 |
| V37 diagnostico POS | 2 |

El peso del historial esta en V5-V12 (76 filas, 39 %) y en V1-V4 (37, 19 %): 116 de 195 filas son
las fases iniciales, que el notebook cubre por narracion y no por ID.

## Hallazgos de la fase 1

### H-01 (corregido) Fila `V37-DIAG-POS` con 21 campos en vez de 22

La fila escrita al cierre del ciclo V37 tiene 21 campos: el texto de `decision` nunca se escribio, y
`duracion_seg` quedo con la lista de artefactos mientras `decision` tomo el valor `89.5` (los
segundos reales). Cualquier lector ve desde `accuracy_mean` los campos desplazados una posicion.

- **Origen**: escritura de la fila, en el cierre anterior.
- **No se perdio ningun dato**: `duracion_seg=89.5` se reconstruyo desde
  `experiments/age_linguistic_14_diagnostico_pos.json` (campo `segundos`) y los 8 artefactos se
  verificaron uno a uno en disco.
- **Correccion aplicada**: fila reconstruida con las 22 columnas tipadas; `decision` con el cierre;
  `accuracy_mean` vacio (V37 no mide accuracy).
- **Verificacion**: `experiments.csv` con 195 filas x 22 columnas, ninguna fila de largo erroneo, y
  el diff contra la copia previa muestra un unico cambio, el de esta fila.
- **Regla que lo evita**: el script de inventario ahora valida el largo de cada fila y falla con
  `RuntimeError` en el camino C1. Este defecto no habria sobrevivido a la auditoria.

### H-02 (documental) 95 filas citaban rutas que no existian tal cual

El registro usa ` | ` como separador, pero las filas de ADV-* y PH* usan `, `. Ademas, varias filas
citan un CSV de `reports/` sin su carpeta, y la columna incluye prosa ('figuras v25_x.png y
v25_y.png', 'DECISIONS.md (D-087').

**No es perdida de evidencia.** Con un resolvedor que separa por ambos delimitadores y prueba el
nombre del fichero en los subdirectorios conocidos: **de 262 artefactos citados, 230 existen y 0
huecos reales**. Los 32 restantes iniciales eran globs (`lexicons/**`), carpetas, o referencias
textuales.

Ademas `features/` y `models/` estan en `.gitignore` y no aparecen en el indice de Codebase; el
inventario los comprueba contra el arbol real, no contra el grafo.

### H-03 (riesgo de lectura, no defecto) La cobertura documental es asimetrica

- 23 de 195 filas (12 %) tienen su ID citado literalmente en el notebook.
- 172 filas no aparecen por ID. No es automaticamente un hueco: el notebook resume por fase y usa
  otra nomenclatura ('PHASE - V13 ...'). Falta el mapeo semantico, que es trabajo de la fase 4.
- Las ultimas fases sí están narradas, pero sin ID: `D-081` y `D-086` aparecen, `D-088` y `D-089` no.
- `V36` y `V37` no aparecen en absoluto en el notebook.

### H-04 El notebook conserva 35 celdas de 163 con contador de ejecucion, 0 errores

Ninguna salida de error guardada, lo que es consistente con las validaciones `--verificar` y
`--humo`. Pero la mayoria del codigo del notebook no tiene salida persistida: la reproducibilidad
del notebook hay que juzgarla por el codigo y por los artefactos, no por sus salidas.

## Estado

Fase 1 **cerrada**. Inventario congelado y reproducible. Las fases 2 (correccion de ejecucion) y 3
(parametros y justificacion) se delegan por familia a los agentes especialistas.
