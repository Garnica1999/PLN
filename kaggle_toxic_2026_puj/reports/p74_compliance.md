# P74: auditoria de compliance

```text
transformers = 0            BERT = 0              RoBERTa = 0
LLM = 0                     classifier externo = 0
external labels = 0         external embeddings = 0
new external corpus = 0     pseudo-labeling = 0
test labels = 0             test training = 0
holdout reuse = 0           leaderboard tuning = 0
submission = 0
dependencias: torch, numpy, sklearn, scipy, pandas
```

## El Word2Vec

Entrenado desde cero por el proyecto (skip-gram con negative sampling implementado a mano en
PyTorch). Corpus externo `mmaguero/covid19-spanishtweets-earlylateapril2020`, licencia CC BY-SA 4.0
con atribucion, periodo cerrado de 2020, cero solapamiento exacto, normalizado y por
near-duplicados (Jaccard maximo 0.142857). Autorizacion del profesor archivada en
`docs/p73_provenance_autorizacion_corpus.md` con el texto verbatim.

## Verificaciones antes de ejecutar

```text
sha1 de la matriz con padding     == el registrado por P73     OK
holdout apartado, no usado        248 filas, 0 usadas por P74  OK
el texto de test no entra en      vocabulario, tokenizer, entrenamiento, embedding, calibracion,
                                  seleccion ni evaluacion      OK
las diez matrices aleatorias      distintas entre si por hash   OK
norma por token en la rama B      desviacion maxima < 1e-6     OK
ninguna matriz aleatoria coincide con la entrenada              OK
```

## Nota de transparencia

La primera comprobacion del hash dio **falsa alarma**: hasheaba `W` sin la fila de padding
(77657 x 100) mientras que P73 registro el hash de la matriz **con** padding (77658 x 100). Se
verifico la causa antes de escalarla y el artefacto esta intacto. Se deja escrito para que el
procedimiento quede documentado.
