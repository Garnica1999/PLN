# Decision de P75

**CASO 5: CNN_PERJUDICA**

```text
V03                            0.7772114525
CNN sola (diagnostica)         0.5838755350
alpha primario 0.1  delta -0.0052438938  folds 1/5
gate delta >= 0.003 y >= 4/5 folds y net positivo  ->  NO PASA
rescue 268  damage 813  net -545   (unidades tweet-etiqueta)
finalista                      None
```

## Los dos niveles que no se confunden

```text
Word2Vec + TextCNN produce informacion complementaria   -> solo si los errores muestran correccion especifica
Word2Vec + TextCNN mejora V03                           -> solo si la fusion demuestra transferencia en corrected_cv
```

Las probabilidades del CNN desplazan decisiones de forma netamente negativa. Se cierra la fusion y no se rescata automaticamente: nada de probar alpha mas pequeno, alpha por etiqueta ni stacking sin una hipotesis nueva.

## Estado

```text
V03 = FROZEN
P72 = CLOSED
P73 = COMPLEMENTARY
P74 = EVIDENCIA REPRODUCIBLE
P75 = EJECUTADA
TEXTCNN = WAITING (se cierra aqui salvo que P75 justifique una fase posterior)
test = 0
holdout = 0
leaderboard = 0
submission = 0
```
