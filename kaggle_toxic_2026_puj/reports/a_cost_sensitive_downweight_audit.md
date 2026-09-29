# Auditoria de A_cost_sensitive_downweight

Generado por `experiments/p69_01_auditoria_downweight.py`. Fecha 2026-09-29T17:48:42+00:00.

## La pregunta de este ciclo, y ninguna otra

```text
existe evidencia reproducible y auditable de que
A_cost_sensitive_downweight mejora V03?
```

No se disena una tecnica. No se amplia el rango de pesos. No hay grid. No se usa el
test, ni el holdout, ni el leaderboard. No se usa este resultado para elegir
hiperparametros nuevos.

## Por que este caso no se puede cerrar sin auditoria

```text
corrected_cv     0,7854160387     (V03: 0,7772114525165721)
delta_vs_V03     +0,0082045862
pliegues         5 de 5 positivos
```

Es el mejor resultado de ponderacion del proyecto. **No se rechazo porque la mejora
fuera negativa**: se rechazo por el criterio 6, el del artefacto de prediccion. Es un
fallo procedimental, no experimental, y esa diferencia es la razon de esta auditoria.

## ITEM 3 - La procedencia completa

| campo | valor |
|---|---|
| experiment_id_original | V4-X01 (candidato A) |
| nombre_en_el_gate | A_cost_sensitive_downweight |
| clave_del_oof | combo__downweight_violacion_x_clase |
| mecanismo_segun_el_codigo | downweight coste-sensible por violacion del umbral, combinado con un barrido anidado del m |
| script_que_reporta | experiments/36_candidatos_v4.py |
| script_que_construye | experiments/30_supervised_weighting_v4.py |
| el_reportador_NO_calcula_el_candidato | True |
| lo_carga_de | features/oof_weighting_v4.npz |
| fecha_del_registro | 2026-09-25 |
| dataset | desarrollo (particion dev), 1483 filas |
| folds | pliegues_corregidos con grupos near-duplicado fusionados, 5 pliegues |
| seed | 42 |
| representacion | TF-IDF word(1,2) min_df=2 sin sublinear_tf + char_wb(3,5) min_df=2, ajustado por pliegue |
| clasificador | LogisticRegression(C=2.0, max_iter=1000, solver=liblinear) |
| class_weight | barrido anidado de multiplicadores (0.8, 1.0, 1.2, 1.5), 3 rondas |
| thresholds | [0.44, 0.51, 0.46] |
| score_reportado | 0.7854160387 |
| delta_reportado | 0.0082045862 |
| pliegues_mejorados | 5 |
| veredicto_externo | no existe: nunca se envio al leaderboard |

### La cadena de generacion

| orden | script | commit | artefacto |
|---|---|---|---|
| 1 | `experiments/27_label_consistency_v4.py` | `01ad7ea` | `features/label_consistency_v4.npz` |
| 2 | `experiments/28_hardness_v4.py` | `01ad7ea` | `features/hardness_v4.npz` |
| 3 | `experiments/30_supervised_weighting_v4.py` | `c96cb0f` | `features/oof_weighting_v4.npz` |
| 4 | `experiments/36_candidatos_v4.py` | `5d06ce7` | `reports/v4_candidates.json` |

**El dato que lo cambia todo:** el script que reporta el candidato NO lo calcula.
`36_candidatos_v4.py` lo **carga** de `features/oof_weighting_v4.npz`, que produce
`30_supervised_weighting_v4.py`, que a su vez necesita `features/hardness_v4.npz`.
`features/` esta en `.gitignore`, luego **ninguno de los tres artefactos viaja entre
worktrees**. En este worktree no existia ninguno.

## ITEM 4 - La hipotesis original, reconstruida del codigo

```text
margen_favor = (p_V03 - umbral) * (2*y - 1)      >0 acierto, <0 error
violacion_l  = clip(-margen_favor, 0, None)      por etiqueta
error_score  = violacion_l.mean(axis=1)          POR EJEMPLO, media de 3
z            = (error_score - mediana) / IQR
peso         = clip(1 - 0.5*z, 0.5, 1.5)         POR EJEMPLO
```

Y la comprobacion de que la formula del artefacto es la del codigo:
desviacion maxima frente al OOF canonico `1.484e-08`.

| propiedad | valor |
|---|---|
| depende_de_p_V03 | True |
| depende_de_y | True |
| depende_del_margen | True |
| depende_de_label | NO para el peso: se promedia sobre las 3 etiquetas |
| unidad | por tweet (ejemplo), NO por par tweet-etiqueta |
| mediana_de_error_score | 0.0 |
| IQR | 0.016266 |
| peso_minimo | 0.5 |
| peso_maximo | 1.0 |
| peso_medio | 0.86337 |
| n_con_peso_mayor_que_1 | 0 |
| n_con_peso_menor_que_1 | 441 |
| n_con_peso_igual_1 | 1042 |
| n_en_el_suelo_0_5 | 371 |
| suma_de_pesos | 1280.38 |
| suma_neutra | 1483 |
| fraccion_de_peso_eliminada | 0.13663 |

### El signo, que es lo que importa

**La regla solo baja pesos.** Ningun ejemplo sube por encima de 1, porque
`error_score >= 0` y su mediana es 0. Quien recibe menos de 1 es quien V03 falla en al
menos una etiqueta; quien recibe 1 es quien V03 acierta las tres.

La regla retira el **13.7 por ciento** del peso total del entrenamiento.

## ITEM 5 - Por que fallo el criterio 6

```text
categoria      A_falta_de_artefacto
criterio       6_artefacto_de_prediccion_completo
submission     NO EXISTE
```

**Que faltaba exactamente**: un CSV de submission. El candidato NUNCA se convirtio en prediccion sobre el conjunto de prueba, luego no hay artefacto que auditar ni enviar

**Por que no se creo**: la directiva 12 de la fase v4 prohibia generar artefactos nuevos durante la auditoria. La ausencia del fichero es una decision de gobernanza, no un fallo del experimento

### Descartando las otras causas, una por una

| causa posible | aplica? |
|---|---|
| B_artefacto_corrupto | no aplica: no hay artefacto que pueda estar corrupto |
| C_no_trazable | el OOF SI era trazable, pero vivia en features/, ignorado por git |
| D_score_no_reproducible | es lo que esta auditoria comprueba, no la causa |
| E_codigo_no_recuperable | el codigo SI es recuperable y sin drift |
| F_inconsistencia_reporte_artefacto | hay una: P58 describe el mecanismo como 'meta', y el codigo no construye ningun meta-modelo. Se documenta abajo |

### La inconsistencia documental que hay que corregir

el CSV de P58 describe el mecanismo como 'meta con downweight coste-sensible por violacion del umbral'. El codigo de 30_supervised_weighting_v4.py NO construye ningun meta-modelo: entrena LogisticRegression directamente sobre la representacion de V03 con sample_weight

**Consecuencia**: la etiqueta 'meta' puede inducir a creer que el candidato pertenece a la familia de stacking, que ya fue rechazada por transferencia. No pertenece.

**Fuente autoritativa**: el codigo, no la etiqueta.

## ITEM 10 - Ficha de la reproduccion

| | historico | reproducido |
|---|---:|---:|
| corrected_cv | 0.7854160387 | 0.7854160387 |
| delta vs V03 | +0.0082045862 | +0.0082045862 |
| pliegues positivos | 5 de 5 | 5 de 5 |
| deltas por pliegue | 0.002243 | 0.001592 | 0.003378 | 0.023607 | 0.010356 | 0.002243 | 0.001592 | 0.003378 | 0.023607 | 0.010356 |
| rescate | 99 | 99 |
| damage | 74 | 74 |
| neto | +25 | +25 |

**Nivel de reproducibilidad: `R3`.** Estado: **REPRODUCIBLE**.

Sobre la escala del proyecto: `R2` es reproducido dentro de tolerancia y `R3`
exactamente. **No se declara R3 solo porque el score sea cercano**: este candidato usa
`LogisticRegression` con `liblinear`, que es determinista dado el mismo input. Se
exige coincidencia de score, delta, pliegues, rescate y dano a la vez, y se cumple:
`True`, `True`, `True`, `True`.

## ITEM 14 - Bootstrap pareado

```text
delta de acierto por ejemplo   +0.0056192403
IC95 pareado                   [-0.0002247696, +0.0116880198]
historico                      [-0.0002250000, +0.0116880000]
incluye el cero                True
```

## ITEM 13 - Rescate, dano y neto

| etiqueta | V03 ok / cand ko | V03 ko / cand ok | ambos ok | ambos ko | rescate | damage | neto |
|---|---:|---:|---:|---:|---:|---:|---:|
| HS | 23 | 42 | 1144 | 274 | 42 | 23 | +19 |
| TR | 19 | 14 | 1300 | 150 | 14 | 19 | -5 |
| AG | 32 | 43 | 1174 | 234 | 43 | 32 | +11 |
| GLOBAL (celdas) | 74 | 99 | 3618 | 658 | 99 | 74 | +25 |

## ITEM 16 - El criterio de resolucion

```text
SI   reproduccion_valida
SI   cinco_de_cinco_folds
NO   bootstrap_compatible_con_mejora
SI   delta_positivo

RESOLUCION: MARGINAL
```

## ITEM 11 - Compliance

- `OK` transformers = 0
- `OK` BERT = 0
- `OK` RoBERTa = 0
- `OK` LLM = 0
- `OK` embeddings preentrenados de terceros = 0
- `OK` acceso a red = 0
- `OK` solo se lee el train oficial y el OOF
- `OK` no se lee el holdout para entrenar
- `OK` no se lee el leaderboard
- `OK` no se envia ninguna submission

**10 de 10, 0 fallos.**

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p69_01_auditoria_downweight.py`. Sin push.*