# P7 - Auditoria del notebook 10_final_pipeline.ipynb

- Celdas totales: 63 (codigo 32, markdown 31)
- Celdas con error en salida: 0; con Traceback: 0

## Cobertura por familia

| familia | codigo | resultados | decision | estado |
|---|---|---|---|---|
| V03 (referencia) | no | si | si | documentado_sin_codigo |
| v2 (correccion de protocolo) | si | si | si | reproducible_en_notebook |
| v4 (holdout) | no | si | si | documentado_sin_codigo |
| v5/v5.1 (embeddings) | si | no | si | documentado_sin_codigo |
| v6 (tracks cerrados) | no | si | si | documentado_sin_codigo |
| v7 (umbral anidado) | no | no | si | documentado_sin_codigo |
| v8 (tecnicas) | no | no | no | ausente |
| finales (FIN/meta v08) | no | si | si | documentado_sin_codigo |

## Numeros clave presentes en SALIDAS del notebook

- 0.777211 (corrected_cv V03): presente
- 0.78306 (LB V03): presente
- 0.7989 (OOF historico con umbrales): presente
- 0.7927 (OOF historico 0.5): presente
- 0.8285 (holdout V03): presente
- 0.8465 (holdout meta): presente
- 0.78163 (LB meta v08): presente
- 0.8163 (holdout meta36 (rechazado)): AUSENTE

## Nota
Estructura real: secciones 1-18 cubren v1, v2, v4 y v5/v5.1 con celdas de verificacion ejecutables (comprueban artefactos y cifras, no reentrenan modelos). NO existen secciones de v6, v7 ni v8: esas fases viven solo en reports/ (documentadas con cifras, sin celda de notebook). Estado global: 'documentado y verificable en sus cifras; no reproducible end-to-end desde el notebook; incompleto para v6-v8'.
## Ejecucion en vivo (auditoria P6, worktree aislado)

- `jupyter nbconvert --execute` sobre el notebook final: **FALLA** en la celda 30
  (AssertionError: 'El proyecto declara 12 sondas de leaderboard'): el log final
  tiene 15 filas (12 de v2 + 3 de v8). La validacion EBD-26 (GO, 4.6 s) es
  ANTERIOR a las sondas de v8 y no se repitio tras anadirlas.
- Consecuencia: el notebook no es ejecutable end-to-end en el estado final (si
  lo era antes de v8). Correccion B-14.
- La seccion 11 (markdown, celda 29) tambien declara '12 sondas' en prosa.
