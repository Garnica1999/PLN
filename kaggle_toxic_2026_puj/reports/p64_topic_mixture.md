# P64 - Mezcla de temas como descriptor de composicion

Generado por `experiments/p64_08_lda.py`. Fecha 2026-09-29T14:29:15+00:00.

## Lo que encontro la auditoria, y por que se abre la reserva

Hay **cinco** factorizaciones latentes ya medidas en este repositorio, y las cinco
estan cerradas:

| experimento | que era | por que cerro |
|---|---|---|
| DL-01 | TF-IDF -> TruncatedSVD(300) -> MLP | proyeccion neutral |
| V9-05 | SVD(6) frente a compresion supervisada | sin adopcion |
| V25-01 | interaccion de segundo orden de bajo rango sobre V03 | cerrada |
| V38-01 | matriz termino por etiqueta factorizada con SVD | invalida por rango 3 |
| V40-01 | PLSRegression(8) sobre V03 | sobreajusta; no reabierta por V42 ni V44 |

Ese es un priori fuerte en contra de una sexta factoriazacion usada IGUAL, es decir
para REEMPLAZAR la representacion y clasificar desde la proyeccion. Este script no
repite eso.

**Lo que ninguna de las cinco hizo es justo lo que P63 demostro que importa.** P63
midio que la COMPOSICION lleva informacion: los ratios de rasgos morfologicos
batieron a un control aleatorio en 0,1055 aislados y anadieron 0,00835 de margen a V03,
y aun asi no mejoraron V03. Eso es un resultado sobre la FORMA del documento y no sobre
su contenido, y la mezcla de temas es la contraparte tematica de ese bloque de ratios.

Asi que esto es un descriptor y no una proyeccion. LDA se ajusta en el entrenamiento de
cada pliegue, la mezcla documento-tema entra como un bloque denso pequeno, y el
clasificador es el LogisticRegression de V03 con ese bloque anadido.

## Una sola configuracion, declarada antes

```text
lda          {"doc_topic_prior": 0.1, "learning_method": "batch", "max_iter": 20, "n_components": 8, "random_state": 42, "topic_word_prior": 0.01}
count        {"lowercase": true, "max_features": 3000, "min_df": 3}
busqueda del numero de temas: NINGUNA, por regla del encargo
```

## Resultados

| brazo | corrected_cv | delta vs V03 | pliegues | neto |
|---|---:|---:|---:|---:|
| V03_lda | 0.7726501697 | -0.0045612828 | 1/5 | -21 |
| V03_lda_completo | 0.7525751216 | -0.0246363310 | 0/5 | -87 |
| V03_random | 0.7747691669 | -0.0024422856 | 2/5 | -10 |
| V03_margen | 0.7661312520 | -0.0110802005 | 1/5 | -90 |

### Las brechas

```text
la mezcla completa contra el control de anchura:  -0.0221940454
el perfil de 5 rasgos contra el control de anchura: -0.0021189973
el perfil de 5 rasgos contra el margen de V03:    +0.0065189177
```

### Coste, porque una sexta factoriazacion tiene que justificarse

| pliegue | vocabulario | segundos de LDA |
|---:|---:|---:|
| 1 | 753 | 2.1 |
| 2 | 768 | 2.2 |
| 3 | 775 | 2.2 |
| 4 | 774 | 2.1 |
| 5 | 776 | 2.1 |

### El perfil de mezcla por estrato

| estrato | n | `dominante` | `entropia` | `concentracion` | `segundo` | `dispersion` |
|---|---:|---:|---:|---:|---:|---:|
| todas | 1483 | 0.5929 | 0.4978 | 0.5929 | 0.2437 | 0.2037 |
| V03 correcto | 1042 | 0.5960 | 0.4966 | 0.5960 | 0.2431 | 0.2048 |
| V03 incorrecto | 441 | 0.5857 | 0.5006 | 0.5857 | 0.2450 | 0.2012 |
| error de alta confianza | 295 | 0.5797 | 0.5054 | 0.5797 | 0.2481 | 0.1993 |
| CORTA | 167 | 0.6835 | 0.4982 | 0.6835 | 0.1476 | 0.2239 |
| MEDIA | 602 | 0.6187 | 0.4768 | 0.6187 | 0.2433 | 0.2126 |
| LARGA | 547 | 0.5669 | 0.5007 | 0.5669 | 0.2654 | 0.1970 |
| MUY_LARGA | 167 | 0.4946 | 0.5634 | 0.4946 | 0.2695 | 0.1738 |

### Por estrato de longitud

| bin | n | V03 | perfil | mezcla completa | margen |
|---|---:|---:|---:|---:|---:|
| CORTA | 167 | 0.9480 | 0.9484 | 0.9373 | 0.9379 |
| MEDIA | 602 | 0.7964 | 0.7897 | 0.7727 | 0.7825 |
| LARGA | 547 | 0.6834 | 0.6820 | 0.6550 | 0.6806 |
| MUY_LARGA | 167 | 0.5606 | 0.5446 | 0.5107 | 0.5595 |

## Veredicto

```text
la sexta factoriazacion latente no aporta: -0.0021189973 contra el control de anchura y +0.0065189177 contra el margen. Y con esto quedan SEIS factorizaciones distintas medidas y cerradas, que es la mayor evidencia que puede tener una familia cerrada en este repositorio

LECTURA: CERRADA
DECISION: CLOSED
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p64_08_lda.py`. Sin push.*