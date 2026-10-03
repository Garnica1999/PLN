# Decision de P74

**CASO 1: EVIDENCIA_REPRODUCIBLE**

```text
A gana en 10 de 10 replicas
media del delta +0.0242705458
IC 95 t de Student sobre replicas [+0.0139795245, +0.0345615672]
epsilon 0.01 | dentro de la banda False
control A contra A, misma semilla +0.000000000000
```

Existe evidencia reproducible de que la representacion Word2Vec entrenada desde cero aporta informacion que el lector TextCNN puede utilizar y que **no se explica por una unica realizacion del embedding aleatorio**.

**No** significa que la TextCNN sea competitiva con V03. **No** significa que exista todavia una mejora sobre V03. **No** significa que haya que hacer una submission.

## Estado

```text
V03 = FROZEN
P72 = CLOSED
P73 = COMPLEMENTARY (evidencia direccionalmente consistente y cuantitativamente inestable)
Word2Vec Twitter 100d = FROZEN
P74 = EJECUTADA
TEXTCNN = WAITING
test = 0
holdout = 0
leaderboard = 0
submission = 0
```
