# V20-00: auditoria de si la familia de interacciones de scores ya se ejecuto

Alcance: worktrees `PLN-v7`, `-v8`, `-v11`, `-v12`, `-v13`, `-v14`, `-v15`, `-v16`, `-v17`,
`-v18`, `-v19`, `-v20`, subdirectorio `kaggle_toxic_2026_puj`, sobre `experiments/`, `reports/`,
`DECISIONS.md` y `experiments.csv`. Metodo: lectura directa de codigo, no busqueda por nombres.

Nota de estructura: los worktrees son copias acumulativas. `PLN-v20/.../experiments/` contiene los
146 ficheros de los worktrees anteriores mas los suyos, asi que la cobertura se resolvio sobre el
worktree `v20` y se comprobo por conteo de ficheros en los demas (v7=78, v8=98, v11=119, v12=126,
v13=129, v14=131, v15=134, v16=138, v17=141, v18=142, v19=146). `PLN-v20` reporto 145 ficheros en
`experiments/` mas lo que ya existia; ningun worktree anterior tiene un script que falte en `v20`.

## 1. Las tres cosas que hay que separar

1. **Productos de features de TEXTO.** Factores que son proyecciones o conteos del texto.
   Ejemplo canonico: `v9_05_proyecciones_supervisadas.py:101`.
2. **Productos dentro de un meta-modelo.** V2 y V13 (stacking) usan columnas de scores, margenes y
   entropias como entrada de un clasificador ya entrenado. Un producto ahi no es V20.
3. **La formulacion de V20.** Scores OOF de V03 (HS, TR, AG) multiplicados entre si, o combinados
   para cambiar la decision de UNA sola etiqueta, con V03 como base.

## 2. Tabla por formulacion

| Formulacion | Fase | Script y linea | Estado | Que midio o por que no aplica |
|---|---|---|---|---|
| `HS_score * TR_score` | V6-B | `experiments/v6_02_label_factorization.py:188` | ejecutada, pero NO es la formulacion de V20 | `proba_tr = proba_hs * cond_tr_hs1`. El segundo factor NO es el score de TR de V03: es un modelo NUEVO entrenado solo en filas HS=1 (lineas 181-187). El primero tampoco es el OOF de V03: es un HS reentrenado por este codigo (lineas 167-170). B1-TR-condicional: 0.766495, delta **-0.010716**, IC95 [-0.01843, -0.003339], 1 rescue / 35 danio, 1/5 pliegues, Gate 1 no superado. V03 si es la base y solo se toca la columna TR (lineas 342-344). |
| `HS_score * AG_score` | ninguna | - | nunca formulada | Ningun script multiplica el score de HS por el de AG. El mas cercano es V6-02 linea 207-210, donde HS multiplica a un condicional de AG, no al score de AG de V03. |
| `TR_score * AG_score` | ninguna | - | nunca formulada | Ningun script multiplica el score de TR por el de AG. |
| `HS_score * TR_score * AG_score` | ninguna | - | nunca formulada | Busqueda explicita del producto triple en los 12 worktrees: cero coincidencias. |
| `x * (1 - y)` | V6-B (parcial) | `experiments/v6_02_label_factorization.py:207-210` | presente pero no es la formulacion de V20 | `proba_ag = proba_hs*cond_ag_hs1 + (1 - proba_hs)*cond_ag_hs0`. Tiene la forma `x*(1-y)`, pero `y` es un condicional de AG entrenado en la rama HS=0, no el score de otra etiqueta. B2-AG-dos-ramas: 0.776379, delta -0.000832, IC95 [-0.007475, +0.005724], 2/5 pliegues, Gate 1 no superado. |
| `logit(x) * logit(y)` | V12-03 (parcial) | `experiments/v12_03_entrenamiento_por_pares.py:257-258` | presente pero no es la formulacion de V20 | `d_tr = arriba_hs * (logitos[:,0] - logitos[:,1])`, `d_ag` analogo. Opera en logit y sobre las tres etiquetas, pero es un **termino de funcion de perdida**, no un post-proceso sobre scores: el multiplicador `arriba_hs` es +1 o -1 segun la etiqueta VERDADERA (linea 256) y la prediccion sale de modelos ajustados conjuntamente. `lambda=0.25` da **-0.002469** con deteccion 0.4915 (minimo 0.5166); la curva es monotona decreciente con maximo en `lambda=0`. H3 cerrada. |
| Equivalentes en log-score | V11-07, V8-08, V13, 33 | `v11_07_correccion_prior.py:112,146`; `v8_08_corrector_margenes.py`; `17_stacking_v2.py`; `33_class_conditional_lm_v4.py:347-349` | presente pero no es la formulacion de V20 | (a) V11-07 usa `logit(p)` pero **aditivo y de una sola etiqueta**: `logit(p) + logit(pi_dest) - logit(pi_origen)`, correccion de prevalencia. (b) V8-08 es un corrector, un **segundo modelo** sobre margenes de V03. (c) V13 es el meta-modelo de stacking. (d) 33 calcula la entropia binaria elemento a elemento de las tres columnas de V03, `-(p*log(p) + (1-p)*log(1-p))`, como columnas de meta: transformacion de cada score consigo mismo, no combinacion de dos etiquetas. Todas categoriza 2. |

## 3. Lo que se descarto de forma expresa, y por que

- **V9-05, `experiments/v9_05_proyecciones_supervisadas.py:100-101`.** Es el unico sitio del
  proyecto donde aparecen literalmente los tres productos cruzados por pares:
  `np.column_stack([p, p[:,0]*p[:,1], p[:,0]*p[:,2], p[:,1]*p[:,2]])`. **No es V20**: `p` son las
  proyecciones de la matriz TF-IDF sobre direcciones discriminantes por etiqueta
  (`proyecciones = np.column_stack([x_tr @ d for d in direcciones])`, lineas 98-99), y las seis
  columnas alimentan una LogisticRegression nueva (linea 110). Categoriza 1: productos de texto,
  no de scores de modelos distintos. Ejecutado el 2026-09-26: SDR 0.706634 contra SVD 0.653145,
  delta frente a V03 **-0.070578**, veredicto "SDR SUPERA A SVD", sin adopcion.
- **V19-02, `experiments/v19_02_restriccion_tr_implica_hs.py`.** Depende del score de otra
  etiqueta para cambiar una decision, pero es una **regla dura** con un gatillo, no un termino
  continuo. Docstring lineas 27-30: "si TR predice 1 entonces HS predice 1", un solo parametro.
  Medido: propagar hacia arriba **+0.000685**, hacia abajo **-0.000646**. JSON: delta medio fuera de
  muestra **+0.0006626** con desviacion **0.0023678**, 3/5 pliegues, `reproducible: false`,
  `cambios_netos_de_celda: -3` frente a `cambios_netos_necesarios: 26.75`, estado
  **REGLA DESCARTADA**.
- **V19-01** es valoracion en forma cerrada de la F1, sin producto. **V19-03** diagnostico.
  **V16** (00 a 03) es prevalencia, generador de submissions, promedio de umbrales y vecinos de
  capacidad: nada de interacciones. **V8** (00 a 13) es auditoria de brecha, ensemble de pliegues,
  log-count ratio sobre la matriz TF-IDF (vease `DECISIONS.md:2349-2384`,Decision 2: "Multiplicar
  la matriz por eso destruye la escala relativa"), C y pesos por etiqueta, corrector de margenes,
  curva de transferencia, clasificadores alternativos: nada de interacciones entre scores.
- **V8-12, `v8_12_seleccion_por_grupo.py:170`.** Unica coincidencia de producto en un script que
  carga el OOF de V03, y es `y[:,0]*4 + y[:,1]*2 + y[:,2]`: codificacion entera de powerset sobre
  las etiquetas **verdaderas**, no scores.

## 4. Comprobacion de cierre

Los 64 scripts que leen el artefacto `features/oof_v03_final.npz` se enumeraron uno a uno y se
busco en cada uno un producto entre columnas de la matriz de scores. Resultado: **cero** en los 63,
y el unico positivo es el de V8-12 linea 170 ya descartado arriba. Entre los 64 estan los scripts de
V6, V7, V8, V11, V12, V13, V15, V16, V17, V18, V19, V4, V5 y el embedding, que es donde vive
practicamente toda la manipulacion de scores del proyecto.

Ademas, `DECISIONS.md` no contiene ninguna entrada que describa la formulacion de V20, y la cadena
`\bV20\b` no aparece ni en `DECISIONS.md` ni en `experiments/`: la fase no esta pre-registrada.

## 5. Terminos buscados, para constancia de cobertura

Productos entre columnas de scores:
`\[:, *0\] *\*`, `\[:, *1\] *\*`, `\[:, *2\] *\*`, `proba[a-z_]*\[:, *[0-2]\]`,
`p_hs *`, `p_tr *`, `p_ag *`, `score_hs *`, `hs_score`, `\* *\[:, *[0-2]\]`, `np.multiply`,
`outer(`, `column_stack` sobre probas, `[:,] *[012][]] * *[:,] *[012]]`.

Triple producto: `\[:, *0\] *\* *\[:,? *1\]? *\* *`, `\[:, *0\] *\* *\[:, *2\]`,
`\[:, *1\] *\* *\[:, *2\]`.

`x * (1 - y)`: `\(1 *- *[a-z_]*proba`, `\(1 *- *p_`, `\(1\.0 *- *[a-z_]*proba`,
`\* *\(1 *- *[a-z_]*(proba|score|p_)`, `\(1 *- *[a-z_]*(proba|score|p_)[a-z_]*\) *\*`.

Log-score: `logit`, `log_odds`, `logit_`, `expit`, `log *\(p *\/`, `log-odds`, `razon de odds`,
`log score`, `log-score`.

Contexto: `interacci`, `no.?lineal`, `condicional a (HS|TR|AG)`, `umbral.*anid`, `depende del score`,
`score de la otra etiqueta`, `puerta`, `routing`, `gate por etiqueta`, `producto de scores`,
`scores.*entre si`, `multiplicar los scores`, `producto punto`, `termino unico.*score`, `logit`.

Artefactos: `experiments.csv` (columna `question` y `hypothesis` de las filas V6-02, V8-*, V11-*,
V12-03, V13-*, V16-*, V19-*), `reports/*.md`, `reports/*.json`, `reports/*.csv`, `DECISIONS.md`,
`RUNLOG.md`, `PROJECT_STATE.md`, `CHAMPION.md`, `FINAL_REPORT.md`.

## 6. Veredicto

La formulacion de V20 es genuinamente nueva: en ningun script de los doce worktrees se multiplican
los scores OOF de V03 (HS, TR, AG) entre si, ni se combina el score de una etiqueta con el de otra
La formulacion de V20 es genuinamente nueva: en ningun script de los doce worktrees se multiplican los scores OOF de V03 (HS, TR, AG) entre si, ni se combina el score de una etiqueta con el de otra mediante un termino continuo para cambiar una sola decision. Lo unico ejecutado y adyacente es
`v6_02_label_factorization.py:188`, que multiplica un HS reentrenado por un condicional de TR
nuevo, con V03 como base y cambiando solo TR, y que ya dio -0.0107 con el mecanismo de escala
diagnosticado en `DECISIONS.md` (D-095). V20 es, en rigor, la version de V6-B con los factores
correctos: los scores que V6-B no tenia.
