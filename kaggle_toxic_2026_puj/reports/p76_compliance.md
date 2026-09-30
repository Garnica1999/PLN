# P76: compliance

```text
transformers = 0     BERT = 0     RoBERTa = 0     LLM = 0
external classifier = 0     external labels = 0     external embeddings = 0
new external corpus = 0     pseudo-labeling = 0
test labels = 0     test training = 0     test calibration = 0     test feature selection = 0
holdout reuse = 0     leaderboard tuning = 0     submission = 0
```

El Word2Vec se reutiliza congelado y su sha1 coincide con el artefacto auditado. La TextCNN no
recibe fine-tuning. Los calibradores se ajustan SOLO con `inner_calibration`, derivada de
`outer_train`: **ningun calibrador ve la outer_validation**. Las predicciones de outer_validation
son las de P75.

No se abrio ninguna familia de calibracion mas alla de Platt, ni isotonic, ni beta, ni meta-modelos.
