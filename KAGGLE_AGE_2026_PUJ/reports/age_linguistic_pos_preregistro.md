# Pre-registro congelado - P2: canal POS-secuencia contra el campeon adoptado

**Fecha**: 2026-09-29. **Estado**: CONGELADO antes de ejecutar (el commit y los hashes se
registran en `experiments/age_linguistic_02_preregistro.json`).

Autorizado por **D-076** (etiquetas si, vectores no) y por el brief de la fase.

## La pregunta, una sola

> `pos_seq` (secuencia de etiquetas POS) contiene informacion que el campeon adoptado
> (V30 + plug-in por clase) **no explota de forma equivalente**, y su integracion por
> promedio neutral de probabilidades mejora el macro-F1 corregido?

No se pregunta si POS tiene alguna senal (casi cualquier representacion la tiene): se
pregunta si aporta sobre el campeon, medida contra las decisiones de EL.

## Materiales congelados (SHA256)

| Artefacto | SHA256 (16) |
|---|---|
| anotacion_md_parquet | `dacbc3055ccde0d9...` |
| anotacion_lg_parquet | `81287d541c082ea8...` |
| modelo_md_meta_json | `889a2fd05d217fa7...` |
| modelo_lg_meta_json | `5007184c49f3cf23...` |
| canonico_y_true | `b1c928f9b9f55755...` |
| canonico_fold_idx | `74d18330c40c3598...` |
| canonico_v30_prob | `3d77352486605971...` |
| manifest_plugin | `8d635e6a546b5a0e...` |

Pesos congelados del plug-in (manifest oficial): `[1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]`

Referencia primaria (la que decide): **baseline congelado global = 0.6159732**
(la regla que ejecuta la submission, recomputada hoy sobre la particion canonica).
Referencia secundaria de continuidad: nested 0.6154643.

## Protocolo exacto (sin tuning)

1. **Particion**: `fold_idx` canonico leido del artefacto persistido (StratifiedGroup
   KFold 5 sobre `dup_clusters`, el mismo con el que se valido el plug-in). Nada se
   reagrupa: los folds se LEEN, no se regeneran.
2. **Canal** (por anotador, md y lg): entrada = `pos_seq` del parquet congelado.
   - Vectorizador: `TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=5, sublinear_tf=True, lowercase=True, token_pattern=r'(?u)\S+')`
   - Modelo: `LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')`
   - Vectorizador y modelo se ajustan **solo** en el train de cada fold; el OOF son las
     probabilidades en el fold de validacion. Filas normalizadas.
3. **Medicion 1** (diagnostico, sin gate): macro-F1 del canal solo.
4. **Medicion 2** (diagnostico): rescate unico / compartido / dano / neto del canal
   contra las decisiones del campeon; rescates unicos sobre errores de alta confianza
   del campeon (max `p_v30` >= 0.6).
5. **Medicion 3 (la que decide)**: `p_mix = 0.5 * p_v30 + 0.5 * p_canal` (filas
   normalizadas); decision = `argmax_c(w_c * p_mix_c)`. Delta primario = macro-F1 de la
   decision menos el baseline primario, sobre los mismos folds. **Alfa congelado en 0.5**
   (neutro; no se busca su optimo).
6. **Controles, definidos ANTES de ver el resultado**:
   - `barajado_filas` (primario): `pos_seq` permutada ENTRE documentos (mismo multiset
     de secuencias, sin asociacion fila-texto); el canal se reentrena igual.
   - `barajado_intra` (secundario): etiquetas permutadas DENTRO de cada documento
     (mismo bag por documento, sin orden).
   Semillas de los controles: barajado_filas = 20261010, barajado_intra = 20261011.

## Gates (todos obligatorios, para cada anotador)

- delta >= **+0.0015** sobre el baseline primario;
- folds positivos >= **4/5**;
- IC95 del bootstrap pareado (2.000 remuestreos, semilla 20261010, unidad la fila) sin
  incluir el cero;
- delta real > delta de `barajado_filas` (el control no puede empatar ni ganar).

## Que NO se hara

- No se ajustan C, n-gramas, min_df, vocabulario, alfa ni los pesos del plug-in.
- No se anaden features de longitud, URL o posicion al canal: es POS y solo POS.
- No se toca el test. No se usa el holdout. No se usa el leaderboard. Sin submission
  automatica.

## Arbol de decision pre-registrado

| Resultado | Estado | Siguiente paso |
|---|---|---|
| Gate completo | **PROMISING** | segunda particion agrupada (misma receta, semilla 987654, reejecucion declarada); sin submission sin autorizacion explicita |
| delta positivo pero < gate, o folds < 4/5 | **MARGINAL** | no se adopta; se cierra POS documentado |
| delta <= 0 | **NEGATIVO** | POS se cierra documentado (regla §38 del brief) |

## Limites declarados

- Alfa = 0.5 es el punto neutro; si el resultado es MARGINAL o NEGATIVO, la conclusion
  valida es sobre ESTA integracion (y sobre el canal como fuente), no una refutacion
  universal de 'POS'. El diagnostico de las mediciones 1 y 2 acota la lectura.
- La particion es la canonica; la segunda particion, si se llega a ella, es reejecucion
  (mismos `dup_clusters`, semilla 987654), no una particion independiente.
