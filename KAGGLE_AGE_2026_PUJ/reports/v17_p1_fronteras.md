# V17 · P1 y P1.1 — Matriz por miembro en las fronteras y complementariedad local

Todo es inerte: no se ha entrenado nada. Se releen las matrices OOF del proyecto.

## 1. La interseccion de errores, que es la cifra que decide

| Frontera | n | V12 | especialista disperso | solo V12 | solo especialista | ninguno | techo con enrutamiento |
|---|---|---|---|---|---|---|---|
| 18-24\|25-34 | 57,023 | 0.4963 | 0.6864 | 3,519 | 14,356 | 14,365 | **0.7481** |
| 25-34\|35-49 | 56,577 | 0.4974 | 0.7218 | 2,872 | 15,571 | 12,867 | **0.7726** |

**Lo que dice la tabla.** El especialista no es una version mejorada de V12: es una decision distinta sobre las mismas filas. En la frontera `18-24|25-34` resuelve 14.356 filas que V12 falla y rompe 3.519 que V12 acierta; en `25-34|35-49` resuelve 15.571 y rompe 2.872. La correccion neta posible es de +10.837 y +12.699 filas respectivamente.

El **techo con enrutamiento** es la accuracy que se alcanzaria si se pudiera elegir bien, fila a fila, entre V12 y el especialista: 0,748 y 0,773. Es una cota, no una prevision: exige saber de antemano cual de los dos acierta.

Y ese techo tiene un limite que conviene no olvidar: en las dos fronteras hay alrededor de un 23 a un 25 % de filas en las que **los dos fallan**. Ahi no hay nada que repartir.

## 2. La confianza de V12 separa las filas que el especialista arregla de las que rompe?

Esta es la pregunta que decide si una compuerta es viable, y se mide antes de construirla. Si las filas que el especialista resuelve no se distinguen por la confianza de V12, cualquier compuerta es un salto al aire.

| Frontera | filas ambiguas | margen mediano de las salvables | de las danables | separacion |
|---|---|---|---|---|
| 18-24\|25-34 | 9.9% | 0.4935 | 0.3763 | +0.117 |
| 25-34\|35-49 | 9.1% | 0.4974 | 0.4404 | +0.057 |

**La separacion es positiva en las dos fronteras**: las filas que el especialista arregla tienen, en mediana, menos confianza de V12 que las que rompe. Es exactamente la condicion que necesita el enrutamiento, y convierte P3 de una apuesta a una prueba con mecanismo.

## 3. Los miembros de V12 en las fronteras

La tabla completa esta en `reports/v17_member_boundary_matrix.csv`, con una fila por miembro y frontera. Ordenados por la frontera `25-34|35-49`:

| Miembro | Familia | accuracy | F1 | rescate | danio | neto |
|---|---|---|---|---|---|---|
| E028a_especialista_par | par | 0.7218 | 0.7214 | 15,571 | 2,872 | +12,699 |
| oof_E015_softmax | ordinal | 0.5143 | 0.5669 | 11,095 | 10,137 | +958 |
| oof_E021a_ordinal | ordinal | 0.4789 | 0.5848 | 4,702 | 5,748 | -1,046 |
| oof_E-011 | prob_lineal | 0.4727 | 0.5857 | 4,334 | 5,729 | -1,395 |
| oof_E-009 | prob_lineal | 0.47 | 0.584 | 4,073 | 5,618 | -1,545 |
| oof_E016_margenes | svm_lineal | 0.4629 | 0.5806 | 3,552 | 5,503 | -1,951 |
| oof_E014_margenes | svm_lineal | 0.4611 | 0.5793 | 3,519 | 5,568 | -2,049 |
| oof_E013_margenes | svm_lineal | 0.459 | 0.5769 | 3,587 | 5,755 | -2,168 |
| oof_ADV05_nbsvm | nbsvm | 0.4409 | 0.5638 | 2,992 | 6,189 | -3,197 |
| oof_E-010 | bayes | 0.4033 | 0.5334 | 2,932 | 8,255 | -5,323 |
| oof_E017_probabilidades | prob_lineal | 0.4033 | 0.5209 | 4,620 | 9,940 | -5,320 |
| oof_E028a_pares | ordinal | 0.4009 | 0.4729 | 9,055 | 14,514 | -5,459 |
| oof_E019b_densas_svd | denso | 0.2964 | 0.4021 | 4,770 | 16,141 | -11,371 |
| oof_E021b_rocchio | prototipos | 0.2862 | 0.3969 | 4,198 | 16,144 | -11,946 |

## 4. Advertencia sobre lo que estos numeros son y lo que no son

Todas las accuracies de este informe estan **restringidas al par**: se evalua solo sobre filas cuya etiqueta real cae en las dos clases de la frontera. **No** son macro-F1 de seis clases y **no** se comparan con el 0,6005 de referencia. Convertir una correccion de +10.000 filas dentro de 57.000 en ganancia de macro-F1 global exige contar ademas el efecto sobre las otras cuatro clases, y eso es trabajo de P11, no una promesa de este script.

Figuras: `figuras/v17_p1_rendimiento_por_miembro.png` y `figuras/v17_p1_interseccion_de_errores.png`.

