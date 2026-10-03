# V32 - E3 (P3) - Pre-registro congelado: challenger FTRL-Proximal sobre TF-IDF

**Fecha**: 2026-09-30. **Estado**: CONGELADO antes de ejecutar (commit `15c2be8`).
**Baseline primario recomputado**: 0.6159732 (referencia secundaria nested 0.6154643).

## La pregunta y la hipotesis

> El aprendizaje online proximal (FTRL-Proximal) sobre TF-IDF aporta informacion
> complementaria al campeon V30 + plug-in que la familia lineal ya usada no captura?

Ultimo candidato de la lista priorizada de la reapertura V32 (P1 D-081, P2 D-082).
FTRL no esta implementado en el repositorio (novedad verificada en el grafo y por grep).

## Materiales congelados (SHA256)

| Artefacto | SHA256 (16) |
|---|---|
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| canonico_ids | `d69ecb339f1dc57d...` |
| manifest_canonico | `8881a9f53e8613aa...` |
| corpus_archivo | `39c6c8270366fe15...` |
| corpus_textos | `53c1a348a7954f3e...` |
| vectorizador | `9bae157614458dbd...` |

## Corpus (identidad verificada, sin reordenar nada)

- Receta: `src.data.load_train()` sobre `data/raw/age_train.csv` (sep `;`, encoding
  `utf-8-sig`, `clean_text` aplicado); texto = columna `text`; orden de filas = orden
  del CSV = orden canonico de `ids.npy` (verificado elemento a elemento) y etiquetas
  = `y_true` canonico (verificado).
- Hash de contenido determinista: longitud (8 bytes LE) + bytes UTF-8 por texto, en orden.
- **Desvio del manifiesto** (auditoria): su `data_hash` (2ada573b27b9cb503cda0bb80344b092b9601a3105fe9b56d2127e261f2edf8e)
  se calculo con `to_numpy().tobytes()` sobre un array de OBJETOS -> serializa punteros
  de memoria; no es reproducible (se observaron `68b9a0f2` y `40342f7c` en dos procesos).
  Se congela en su lugar el hash del archivo + el hash de contenido + la receta.

## Representacion (TF-IDF por fold)

- Config IDENTICA a `VECTORIZADOR` de `v31_02_banco_representaciones.py`: word, (1,2),
  min_df 3, max_df 0.5, sublinear_tf, lowercase, strip_accents None, token_pattern
  `(?u)\b\w+\b`; norm y dtype en defaults de sklearn.
- **Ajuste por fold**: `fit` solo en fold-train (idf, min_df y max_df nunca globales);
  precedente canonico del propio manifest. Verificabilidad: sha256 de config + vocabulario
  ordenado + idf por fold + sonda de refit.

## Brazos

| Brazo | Definicion | Rol |
|---|---|---|
| A1_ftrl | FTRL-Proximal propio (alpha 0.05, lambda1 0.0, lambda2 1.0, beta 1.0, bias regularizado x0=1, sin power_t, 10 pasadas fijas, OvR 6 sigmoides, barajado por epoca con 20261010+fold) | decisional |
| A2_sgd_twin | 6 x SGDClassifier(log_loss, alpha 1e-6, max_iter 10, tol None) con el mismo presupuesto de pasadas y la misma matriz | control de atribucion (decisional) |
| A3_ftrl_barajado | A1 con etiquetas de fold-train permutadas (20261011+fold) | control negativo (no decisional) |

## Sondas obligatorias antes de entrenar (exit 4 si fallan)

1. Vectorizado == lazo ingenuo (problema aleatorio 200x50x6, tolerancia 1e-12).
2. Punto fijo analitico de FTRL con lambda1=lambda2=0 y feature constante (residuo < 1e-9).
3. TotalF1:average=Macro == macro-F1 de sklearn (<= 1e-6).
4. p sin NaN/inf y log_loss finito.

## Regla de fallback (congelada antes de ejecutar, no es grid)

- Criterio: en el fold 0 de A1, sobre las filas de fold-train, log_loss OvR medio tras
  10 pasadas <= 0.8 x el log_loss tras la pasada 1, y sin NaN/inf.
- Si falla: alternativa declarada alpha=0.005 (resto igual) y `config_fallback_usada=True`
  en el JSON de resultados y en el informe.
- Prohibido decidir con fold-val o con el leaderboard.

## Determinismo

- Semillas fijas; OMP/MKL/OPENBLAS = 32 antes de importar numpy; lazo FTRL mono-hilo.
- Sonda: reentrenar fold 0 de A1 y de A2 y comparar bit a bit. Tolerancias: 0 exacto;
  <=1e-9 conforme; >1e-9 con rango publicado y delta_gate > rango; rango >= 0.0008876
  o solape con el gate => no concluyente (entonces no puede ser PROMISING).

## Integracion y gates (mecanismo P6/P7, sin cambios)

- Combinador anidado [L2(v30) || L2(p_brazo)] (12 features); LogReg C=1.0 por fold;
  decision `argmax(p_comb * w)` con los pesos del plug-in CONGELADOS.
- Gates: baseline +-5e-7; delta_int(A1) >= +0.0015; folds+ >= 4/5;
  control favorable (delta A1 > delta A2 Y IC95 pareado de la diferencia sin cero);
  A3 coherente (por debajo de la linea base de la clase mayoritaria); IC95 A1 sin cero;
  sin leakage; determinismo concluyente.
- **Caps**: IC95 de la diferencia con cero -> veredicto capado a MARGINAL; determinismo no
  concluyente -> no PROMISING.

## Arbol de decision

| Resultado | Estado | Siguiente paso |
|---|---|---|
| A3 >= A1 | NO_ATRIBUIBLE | cierre documentado |
| gates completos + IC95 diferencia excluye 0 | PROMISING | segunda particion (semilla 987654) en este ciclo |
| delta > 0 sin gates (o cap) | MARGINAL | no se adopta |
| delta <= 0 | CLOSED | se cierra P3 y termina la reapertura |

## Compliance, novedad y limites

- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin
  leaderboard, sin submission; un script nuevo; campeon intacto. TF-IDF + lineal ya
  usados por el proyecto: no cuenta como embedding vetado.
- `vowpalwabbit 9.11.8` instalado y **no usado** (alternativa disponible descartada).
- AST audit: ausencia de vowpal/vw/torch/transformers/gensim/fasttext y 0 rutas de
  test/holdout/submission. RSS pico con `ru_maxrss` (tope 20 GB). Escritura atomica.
- Asimetria declarada: FTRL regulariza el bias; SGD de sklearn no.
- A2 se aparta de E-011 (max_iter 30/tol 1e-4 -> 10/tol=None) para igualar pasadas.
- El cap por IC95 puede degradar un resultado genuinamente bueno a MARGINAL: decision
  del lead, registrada.
