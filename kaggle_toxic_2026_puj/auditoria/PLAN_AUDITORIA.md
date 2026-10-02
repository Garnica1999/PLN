# PLAN DE AUDITORIA INTEGRAL Y UNIFICACION - kaggle_toxic_2026_puj

Estado general: EN EJECUCION.
Fecha de inicio: 2026-10-02.
Worktree canonico: `/home/carlos/Git/PLN-v67`, rama `proyecto-kaggle-toxic-2026-p67`.
Bitacora viva: este documento se actualiza al cerrar cada fase.

## 1. Objetivo

1. Unificar en el proyecto P67 los contenidos unicos de las 29 worktrees del
   repositorio, sin borrar ni reescribir historia.
2. Auditar todos los experimentos: anomalias, hiperparametros mal configurados,
   experimentos sin ejecutar, calibracion, fugas de datos y codigo desconectado.
3. Declarar y documentar cada experimento en el notebook consolidado
   `notebooks/10_final_pipeline.ipynb`, en espanol, con la enie y sin simbolos
   de otros alfabetos.
4. Proponer, con literatura publicada, el camino hacia 0.80 de F1 en el
   leaderboard externo, respetando las reglas vigentes.

## 2. Reglas no negociables

- V03 permanece congelado: `corrected_cv = 0.7772114525165721`, umbrales
  HS 0.44, TR 0.51, AG 0.46, LB 0.78306. Frase obligatoria en los informes:
  `V03 remains the best verified leaderboard result.`
- Prohibido afirmar que Word2Vec, la RNN, la atencion o el bayesiano superan
  a V03. Prohibido tocar `modelos/twitter_domain/w2v_twitter_100d.npz`
  (sha256 `302c994d13b75f8f5fbd440eea2c4f4113997e2581524021923b92f8244d705a`).
- El holdout v2 (248 filas) esta agotado: 5/5 evaluaciones. No se leen sus
  filas ni se anaden evaluaciones. El test no se usa para entrenar.
- Sin submissions nuevas sin autorizacion explicita de Carlos.
- Un experimento = un script nuevo y completo. Prohibido editar o borrar
  scripts de experimentos anteriores. Prohibido `rm`.
- No se reescribe historia git ni se hace push sin autorizacion.
- Codigo en ASCII. Documentos en espanol con acentos y enie, sin simbolos de
  otros alfabetos ni emojis; verificacion con `tools/verificar_caracteres.py`.
- Cada subagente escribe su resumen (maximo 20 lineas) en
  `auditoria/bitacora.md` al terminar.

## 3. Fases

| Fase | Nombre | Estado | Evidencia |
|---|---|---|---|
| F0 | Preflight y asegurado del estado sucio | COMPLETADA | commit `1026ff1` |
| F1 | Unificacion de workspaces en P67 | EN CURSO | `auditoria/manifiesto_unificacion.csv` |
| F2 | Inventario unificado de experimentos | PENDIENTE | por definir |
| F3 | Auditoria por experimento (fugas, hiperparametros, calibracion, vacios, conexion codigo-corrida) | PENDIENTE | por definir |
| F4 | Correccion de experimentos en vuelo (P83) y re-verificacion | PENDIENTE | por definir |
| F5 | Notebook consolidado con cada experimento documentado | PENDIENTE | por definir |
| F6 | Sugerencias hacia 0.80 con literatura publicada | PENDIENTE | por definir |
| F7 | Verificacion final GO/NO-GO y cierre | PENDIENTE | por definir |

## 4. Criterios de exito

- `auditoria/inventario_experimentos.csv` cubre el 100 por ciento de los
  scripts de `experiments/` y de las filas de `experiments.csv`, con estado y
  evidencia por experimento.
- Cada hallazgo de la auditoria tiene severidad, evidencia con archivo y
  linea, y decision: corregido, documentado o descartado con motivo.
- El notebook consolidado ejecuta sin errores y documenta cada experimento.
- Las sugerencias hacia 0.80 citan fuentes verificables y distinguen lo
  permitido por las reglas de lo que exigiria cambiarlas.
- `tools/verificar_caracteres.py` termina con 0 errores.

## 5. Bitacora

- 2026-10-02: F0 completada. Estado sucio de P67 (P82 y P83, 187 archivos)
  asegurado en el commit `1026ff1`. Verificado: el verificador oficial de
  caracteres admite acentos y enie (solo persigue alfabetos no latinos).
- 2026-10-02: F1 en curso. Importados 67 archivos unicos de la rama de
  auditoria v8 (`audit/`, `reports/audit_*`) y 85 artefactos del rerun
  (`audit/out_p06/`, scripts p06). Copiada la fuente de fastText (commit
  `1142dc4`, 7.2 MB sin `build/`) a `tools/fastText`, ignorada por git como
  en el diseno original. Manifiesto con 153 filas en
  `auditoria/manifiesto_unificacion.csv`. Sin conflictos ni sobrescrituras.
