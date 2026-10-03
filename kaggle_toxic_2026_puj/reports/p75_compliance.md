# P75: compliance

```text
transformers = 0     BERT = 0     RoBERTa = 0     LLM = 0
external classifier = 0     external labels = 0     external embeddings = 0
new external corpus = 0     pseudo-labeling = 0
test labels = 0     test training = 0     test calibration = 0     test feature selection = 0
holdout reuse = 0     leaderboard tuning = 0     submission = 0
```

El Word2Vec se reutiliza **congelado**, sin reentrenar y sin fine-tuning. Su sha1 coincide con el
artefacto auditado. La TextCNN se entrena supervisada solo con etiquetas de `outer_train`, desde
cero, sin pesos preentrenados de terceros.

El corpus externo siguio usandose solo como texto NO etiquetado, cubierto por la autorizacion
archivada en `docs/p73_provenance_autorizacion_corpus.md`. Precision literal de la auditoria: sin
near-duplicados por encima de Jaccard 0.142857, no "sin ningun solapamiento".

Los umbrales 0.44/0.51/0.46 son parte de la definicion de la metrica y se aplican igual a los tres
brazos. No se recalibran.
