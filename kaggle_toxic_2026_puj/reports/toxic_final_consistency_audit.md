# TOXIC - P89-00: auditoria final de consistencia y cierre documental

Fecha: 2026-10-09. Script: `experiments/p89_00_auditoria_final_consistencia.py`.
Ciclo de auditoria y correccion documental: **sin entrenamiento, sin modelos, sin
submissions y sin cambios de modelo**. La unica modificacion del ciclo es la
correccion del bloque P61 duplicado del notebook (`p89_01`, ver seccion 5).

## 1. Estado de V03 (FROZEN, verificado)

| artefacto | sha256 | estado |
|---|---|---|
| OOF canonico | `559265816ce4f6312c87f008f90324f0...` | exacto |
| submission (submissions/) | `7d7478f271e6edfd51afff85bad3eaa0...` | exacto |
| submission (baseline congelada) | `7d7478f271e6edfd51afff85bad3eaa0...` | exacto |
| modelo joblib | `263ed5bd0fc0de7cabfbaa85e88fca6f...` | exacto |
| train | `3d40e7345b588860cf4744914be88ddc...` | exacto |
| particion dev/holdout | `e5b62e5f22c4c63a8c43df4019969504...` | exacto |

`corrected_cv = 0.7772114525165721` (del OOF canonico), umbrales
`[0.44, 0.51, 0.46]`, LB `0.78306` (CHAMPION.md, P46). V03 no se ha tocado.

## 2. Candidato A_cost_sensitive_downweight (conservado)

```text
candidate_status   PROMISING_UNCONFIRMED (5 de 6 componentes)
external_transfer  NEGATIVE (LB 0.77522 frente a 0.78306, delta -0.00784)
research_status    EXPERIMENTAL_RESEARCH_CLOSED
governance_status  PENDING
```

Evidencia congelada verificada: reproduccion R3 (diferencia 3,2e-11); particion
canonica `+0.0082045862`; segunda particion `+0.0105253130`;
tercera particion `+0.0048825807`; permutacion p = `0.000999`;
bootstrap por grupos con IC95 que incluye el cero; submission de 1.154 filas
(1155 lineas con cabecera) en disco, sin pipeline propio.
No se cambia a ADOPTED ni INVALIDATED; no se modifica ningun artefacto; no se
genera otra submission.

Observacion (doctrina del proyecto): la submission del candidato no aparece en
`leaderboard_log.csv`, igual que el envio V33-03 (P57: "el log no se modifica: es
un registro de mediciones"). Su LB esta registrado en el postmortem, el informe
completo, la auditoria de oportunidades y D-288. No se modifica el log historico.

## 3. Cierres P85-01 y P88-00 (verificados)

- **P85-01** (SGNS corregido sobre corpus Twitter/X): caso `C`,
  fusion `-0.0691724132` frente a V03; cierre de la familia de
  embeddings en D-290; seccion 131 del notebook.
- **P88-00** (LightGBM sobre la matriz TF-IDF de V03): `DUPLICATE` de
  `P61-01`; sin entrenamiento nuevo; D-293; seccion 134.

## 4. Auditoria de oportunidades (consistente)

`reports/toxic_remaining_legal_opportunities.md/.csv`: 20 lineas
auditadas; reparto `{'BLOCKED_BY_RULES': 5, 'CLOSED': 14, 'DUPLICATE': 1}`; **0 ACTIONABLE**;
`reports/toxic_final_research_gap.md` presente. D-288 a D-293 presentes con sus
titulos. Reaperturas posteriores legitimas: solo la autorizada de P85 (ejecutada);
P86, P87 y P88 cerraron sin reabrir. Revision humana de TR = `LOW_EXPECTED_VALUE`
(D-292); near-duplicados train->test = `QUARANTINED`.

## 5. Notebook consolidado (verificado y corregido)

- Secciones 130 a 134 presentes (auditoria de oportunidades, P85, P86, P87, P88).
- **Discrepancia detectada y corregida**: el bloque de la fase P61 (secciones 81 a
  88, 24 celdas) aparecia DUPLICADO. La copia huerfana (celdas 265-288) contenia
  las dos celdas de la primera version con bugs (el SyntaxError del f-string de
  cuantiles y la clave 'pool' inexistente); la copia gestionada entre los
  marcadores `P61-NOTEBOOK-INICIO/FIN` tiene las versiones corregidas. `p89_01`
  elimino la copia huerfana: 443 celdas, 0 cabeceras
  duplicadas, marcadores intactos. Registrado en D-294.
- Observaciones sin accion (historicas y consistentes): la seccion 96 ("P62 -
  Cierre") aparece despues de la 99 (la correccion se escribio antes que el
  cierre); la numeracion salta de la 31 a la 33. No son discrepancias de
  contenido y no se modifican.
- Las auditorias de novedad no se presentan como entrenamientos (declaran "no se
  ejecuta nada" en el notebook).

## 6. Cumplimiento por operacion y fichero concreto

Comprobacion AST de los 12 scripts post-cierre (tokens de ruta
sensibles fuera de docstrings + imports + escrituras):

| script | tokens de ruta sensibles | imports prohibidos |
|---|---:|---|
| `p84_06_notebook_completo.py` | 0 | ninguno |
| `p84_07_notebook_auditoria_oportunidades.py` | 0 | ninguno |
| `p85_01_sgns_twitter_corregido.py` | 7 | ninguno |
| `p85_02_integrar_resultado.py` | 0 | ninguno |
| `p86_00_auditoria_novedad_fm.py` | 5 | ninguno |
| `p86_01_integrar_notebook.py` | 0 | ninguno |
| `p87_00_gobernanza_anotacion_tr.py` | 8 | ninguno |
| `p87_01_integrar_notebook.py` | 0 | ninguno |
| `p88_00_auditoria_novedad_lgbm.py` | 5 | ninguno |
| `p88_01_integrar_notebook.py` | 0 | ninguno |
| `p89_00_auditoria_final_consistencia.py` | 55 | ninguno |
| `p89_01_correccion_duplicado_p61.py` | 0 | ninguno |

Todos los tokens son etiquetas de contadores (p. ej. `holdout_filas_usadas`,
`toxic_test_abierto: False`, `submissions: 0`), la verificacion de hash de la
submission congelada del campeon, o (solo en `p87_00`) la lectura de
`holdout_v2_ids.csv` como metadatos de particion para acotar el alcance de la
revision. En `p89_00` los 55 tokens son su propio vocabulario de comprobacion
(patrones, claves y rutas de verificacion), no operaciones. Cero lecturas de
`toxic_test.csv`, cero lecturas del leaderboard, cero escrituras de submission,
cero imports prohibidos.

```text
labels de test utilizadas = 0        (toxic_test.csv no tiene etiquetas: 1154 x 2)
test utilizado para tuning = 0       (ningun script post-cierre lee el test)
holdout reutilizado para tuning = 0  (solo metadatos de particion en P87-00)
leaderboard utilizado para tuning = 0
datasets externos etiquetados = 0    pseudo-labeling = 0
embeddings preentrenados = 0         transformers = 0        LLM = 0
submissions nuevas desde el cierre = 0  (log: 23 sondas, ultima 2026-10-01)
push = 0
```

Distincion registrada: el notebook usa el texto de test de forma legitima en dos
lugares (la validacion de formato de `toxic_test.csv` (celda de datos) y la
reconstruccion byte a byte de la submission congelada de V03 desde su joblib
(celda de reproducibilidad)), nunca para desarrollar o ajustar el modelo.

## 7. Discrepancias

```text
detectadas en este ciclo   1 (bloque P61 duplicado en el notebook)
corregidas                 1 (p89_01; D-294)
pendientes                 0
```

## 8. Estado canonico final

```text
V03 = FROZEN                       corrected_cv 0.7772114525165721 | LB 0.78306
A_cost_sensitive_downweight        PROMISING_UNCONFIRMED
external_transfer                  NEGATIVE (0.77522)
governance_status                  PENDING
P85-01                             CLOSED
P88-00                             DUPLICATE
research_status                    EXPERIMENTAL_RESEARCH_CLOSED
```

La investigacion de modelado solo debe reabrirse si aparece una fuente de datos
legal nueva, cambian las reglas de la competicion o se presenta una hipotesis
materialmente diferente con justificacion empirica. El cierre se limita al
espacio de mecanismos, datos y restricciones auditado.

> V03 remains the best verified leaderboard result.
