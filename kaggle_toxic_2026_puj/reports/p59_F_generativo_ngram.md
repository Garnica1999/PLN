# P59-F - Modelo generativo de n-gramas: CLOSED_BY_DUPLICATION

Generado por `experiments/p59_11_generativo_ngram.py`. Fecha 2026-09-28T14:35:17+00:00. Sin entrenar.

## Que proponia F

Un modelo generativo de n-gramas que exploits evidencia de secuencia o condicionalidad que un modelo discriminativo sparse no representa igual, con un unico orden pequeno y una unica estrategia de suavizado o backoff fijada de antemano.

## Control

```text
V03 0.7772114525165721 reproduce exacto: True
```

## F1 gap check: EXISTE EQUIVALENCIA FUNCIONAL

### Pierna 1, algebrica

Sobre una representacion de n-gramas fija, todo modelo generativo condicional de clase colapsa a

```text
log P(c | x) = log P(c) + sum_f theta_{f,c}
```

que es la forma de naive Bayes y de log-odds. El unico parametro libre es el estimador de `theta`, y todo estimador de `theta` es una estrategia de suavizado: Laplace, backoff de Witten, priors de Dirichlet, o el complemento de ComplementNB. La representacion no salva nada, porque el colapso vale para cualquier base.

### Pierna 2, la familia ya se ejecuto

| id | que hizo | resultado | cierre |
|---|---|---|---|
| MOD-02_BEST | ComplementNB(alpha=1,0) x3 sobre la union TF-IDF de V03 | 0,6888 frente a 0,7587 del lineal, **-0,069949 y 0 de 5 pliegues** | descartada |
| E2-15-NB-WORD | ratio log-count NB-SVM fold-local hacia la misma LR | **-0,0234, 0 de 5 pliegues**, ganancia de ensemble **-0,0057**, correlacion de residuales con V03 **0,9243** | NO adoptada |
| E2-15-NB-CHAR | lo mismo sobre char | -0,0392 con 0 de 5 pliegues | NO adoptada |
| V37-01 | BM25 en word con el clasificador de V03, k1 1,2 y b 0,75 | **-0,0027860149**, rescates 25 contra danios 31, **neto -6**, correlacion **0,9874** | BM25 = CLOSED, no se prueba sobre char |
| V4-G01-G03 | LR sobre verosimilitudes | candidato D con evidencia fragil | --- |
| V26-03 | Tsetlin Machine | REJECTED | familia cerrada |

Cero filas del registro describen un modelo generativo como tal: la familia existe solo con los nombres de arriba.

### Pierna 3, y esta cierra la especificacion exacta de F

El unico grado de libertad de un modelo generativo de n-gramas sobre representacion fija es el **suavizado**, y F esta especificado justo como un orden y una estrategia de suavizado. `V8-04` midio ese parametro directamente:

```text
suavizado 1e-6  ->  -0,058496   con IC95 [-0,07611, -0,04061], 0 de 5 pliegues
suavizado 1e-3  ->  +0,004745   con IC95 [-0,00708, +0,01714], 3 de 5 pliegues
salto entre dos valores que solo difieren en esa constante:  +0,063241
```

Su propia lectura: **el suavizado sostiene el metodo entero**. Y su decision pre-registra, por escrito: *no se barre el suavizado para buscar un valor mejor, porque seria el rescate por hiperparametro que las reglas prohiben desde la fase v6*. Ademas el mejor punto queda a tres milesimas de superar su filtro de detectabilidad, con el IC95 incluyendo el cero.

F es un punto suelto de un barrido cerrado de antemano.

## Decision

```text
decision  CLOSED_BY_DUPLICATION. 0 experimentos consumidos. No es que el modelo generativo de n-gramas falle en este corpus: es que no es un experimento. Sobre una representacion fija colapsa a la forma de naive Bayes, y su unico grado de libertad, el suavizado, esta pre-registrado como barrido prohibido desde V8-04, donde ademas se midio que sostiene el metodo entero. Sin orden sweep, sin suavizado sweep, sin backoff sweep, sin submission. V03 remains the best verified leaderboard result.
```

> V03 remains the best verified leaderboard result.

*Seccion generada por `experiments/p59_11_generativo_ngram.py`. Sin push.*