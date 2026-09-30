# P77: compliance

```text
transformers = 0     BERT = 0     RoBERTa = 0     LLM = 0
external classifier = 0     external labels = 0     external embeddings = 0
new external corpus = 0     pseudo-labeling = 0
test labels = 0     test training = 0     test calibration = 0     test feature selection = 0
test threshold tuning = 0     holdout reuse = 0     leaderboard tuning = 0     submission = 0
```

El Word2Vec 100d se reutiliza congelado y su sha1 coincide con el artefacto auditado. La TextCNN
no recibe fine-tuning. El gate se ajusta SOLO con el OOF interno de outer_train y se aplica una
sola vez sobre outer_validation, que nunca participa en su ajuste. Los cortes por cuantil se
calculan SOLO dentro de outer_train. Los umbrales de V03 no se tocan. Del parquet de anotaciones
se lee EXCLUSIVAMENTE la particion 'dev'.
