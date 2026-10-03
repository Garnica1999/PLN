# P68-H1 - Enrutado por estructura del ejemplo

Generado por `experiments/p68_01_enrutado_por_estructura.py`. Fecha 2026-09-29T17:18:53+00:00.

## La hipotesis, y por que no es otra formula de score

P67 cerro la familia de transformaciones genericas del score dos veces: empiricamente,
porque todas daban cuatro celdas de 1483, y algebraicamente, porque la mejor de ellas
resulto ser un `blend_logit` con otro nombre.

Lo que sobrevivio a P67 fue una **medicion**: la poblacion B del gap analysis, con
`139` celdas delegables medidas y no supuestas. Este experimento
cambia **quien decide cada fila**, y lo decide mirando la **estructura del texto**,
nunca el score.

```text
no concatena columnas        P64 y P65, cerrado
no mezcla scores             P66 y P67, DUPLICATE por algebra
no mueve el umbral           P67-01, cerrado
no es un ensemble de pesos   P61, cerrado
cambia QUIEN decide          esto
```

La diferencia no es cosmetica: un blend produce **una** frontera en el plano de los dos
scores. Una seleccion produce una region que puede ser **no convexa**.

## El protocolo

- Features del enrutador: **123 estructurales**, todas del texto.
- Derivadas del score usadas en la variante principal: **0**.
- Ajuste anidado: 3 pliegues internos. La tasa de delegacion se elige en la
  validacion **interna**, nunca en la externa.
- Aislamiento verificado: solo la particion `dev`. Cero filas de `test` y de `holdout`.

## El techo, medido antes de creerse nada

Si se delegara **exactamente** lo delegable, el resultado seria `0.8162985861`, es
decir `+0.0390871336`. Ese es todo el margen que existe.

| variante | corrected_cv | delta |
|---|---:|---:|
| V03 sin tocar | 0.7772114525 | +0.0000000000 |
| enrutado por estructura | 0.7753122224 | -0.0018992302 |
| objetivo barajado | 0.7763249636 | -0.0008864889 |
| delegacion aleatoria | 0.7764776157 | -0.0007338368 |
| enrutado con el margen de V03 | 0.7742814291 | -0.0029300234 |
| techo teorico | 0.8162985861 | +0.0390871336 |

Celdas delegadas: `423`. Acierta `323`, falla `100`, neto `+223`.
Delta por pliegue: `[-0.0054044262, -0.0027527746, -0.0028894375, 0.0040762641, -0.0028180822]`, positivos `1` de 5.

## El veredicto

```text
INVALIDATED

delta -0.0018992302, 1 de 5 pliegues. El enrutado NO bate a sus controles (barajado -0.0008864889, aleatorio -0.0007338368). La estructura del ejemplo no permite decidir a quien delegar
```

## Por que falla, que es lo que hace util un negativo

Si el enrutador no sirve hay que saber **por que**: si no aprende nada, o si aprende la
cosa equivocada. Se entrenan dos modelos con las **mismas features** y dos objetivos
distintos:

```text
objetivo A   a quien delegar      V03 falla Y el candidato acierta
objetivo B   donde V03 falla      dificultad, sin mirar al candidato
```

| etiqueta | AUC a quien delegar | AUC donde V03 falla |
|---|---:|---:|
| HS | 0.5365 | 0.564 |
| TR | 0.5398 | 0.6514 |
| AG | 0.571 | 0.6085 |

**Media: `0.5491` frente a `0.6080`.**

La estructura del texto predice la **dificultad** bastante mejor de lo que predice
**quien gana**. El enrutador no es que no aprenda nada: aprende la cosa equivocada.

Y eso encaja con el detalle por pliegue: de las celdas delegadas, V03 **ya acertaba**
la mayoria en todos los pliegues y etiquetas. El enrutador manda trabajo al experto
donde el ejemplo *parece* dificil, que es donde V03 ya estaba resolviendo y donde el
experto tambien falla.

### Y el aviso que deja el `f1_inner`

En **11 de 15** celdas pliegue-etiqueta el delta de la validacion interna era
**positivo**, y el delta externo salio negativo. Con 800 filas internas divididas en
tres etiquetas, un `+0,004` interno no transfiere. Es la misma leccion de P66 y P67
otra vez: a esta escala, la seleccion se confunde con la senal.

## La prueba de que esto no es un blend

El enrutado cambia la decision de V03 en `39` celdas.
De ellas, `15` estan a **mas de 0,10 del umbral**, es decir fuera
de la banda donde un umbral podria actuar. Cambiar decisiones lejos del umbral es
algo que **ningun umbral puede hacer**, y es la firma de que el mecanismo es otro.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p68_01_enrutado_por_estructura.py`. Sin push.*