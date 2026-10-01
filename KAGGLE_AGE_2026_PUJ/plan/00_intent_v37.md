# 00 · Intencion confirmada · V37 Diagnostico P(POS | edad)

Fecha: 2026-10-01. Confirmada por Carlos en conversacion (sin entrevista adicional: la
peticion fue explicita y quedo cerrada antes de redactar este documento).

## Peticion literal

> "Se podria utilizar entidades no nombradas o la construccion de silver standard para
> mejorar la clasificacion?" -> "La unica que se me ocurre es devolvernos un paso, estudiar
> las probabilidades condicionales de cada tipo de palabra, es lo que se me ocurre, para hallar
> algun patron." -> "Pues si, hagamos la parte de P (POS | edad), prepara un plan.
> Adicionalmente, si se te ocurre algun experimento para este caso que no hallamos realizado es
> valido." -> "Ejecuta el plan, cargalo y ejecutalo paso a paso. Mantenlo actualizado fase a fase."

## Objetivo

Determinar si la distribucion condicional `P(POS | edad)` contiene patron estable y no
redundante en el corpus, y solo si ese diagnostico lo sostiene, evaluar si existe margen real
para un canal predictivo construido sobre esa informacion.

## Metrica de exito (diagnostico)

No es una metrica de leaderboard. Es un veredicto con evidencia:

1. Tabla de `P(tag | edad)` de 6 clases con soporte, cobertura y dispersion entre folds.
2. Estabilidad: desviacion maxima entre folds y acuerdo md vs lg.
3. Potencia discriminativa de la informacion POS marginal: macro-F1 OOF plug-in (Bayes
   cerrado, sin hiperparametros) con el suelo de azar multiclase 0.1667 y la referencia ya
   medida de P2 (canal solo 0.2751) y P1a (bloque solo 0.3997).
4. Veredicto: `PATRON_ESTABLE_NO_REDUNDANTE` o `DIAGNOSTICO_CERRADO`.

## Datos

Solo train. Fuentes congeladas: `features/linguistic/anotacion_md_train.parquet` y
`anotacion_lg_train.parquet` (169.336 x 8), `artifacts/p33_scores/todos/` (y_true, fold_idx,
v30_prob, cand_prob, cand_pred, ids). Etiquetas de train. Sin test, sin holdout, sin
leaderboard, sin submission.

## Restricciones (heredadas, no negociables)

Campeon V30 + plug-in congelado. Prohibidos transformers, LLM, embeddings (preentrenados o
entrenados), datos o etiquetas externas, metadatos, un tweet por fila sin agregar por autor,
cualquier ajuste sobre test/holdout/leaderboard, grillas de hiperparametros, edicion o borrado
de scripts ya ejecutados. Un experimento = un script nuevo. Pre-registro congelado antes de
correr. spaCy `es_core_news_md`/`_lg` autorizado solo para etiquetas (D-076).

## Out of scope

No se entrena ni se modifica el modelo en esta fase. No se cambia el campeon. No se reescriben
actas cerradas. No se toca V30, la representacion, la regla, los pesos, la semilla ni los
hiperparametros.