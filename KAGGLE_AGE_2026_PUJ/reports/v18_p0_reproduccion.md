# V18 · P0' — Reproduccion y cierre del confound de la representacion

## 1. Reproduccion de la evidencia previa

| Magnitud | Valor |
|---|---|
| V12 pooled | 0.6005 |
| A2 de V16 | 0.6011 |
| frontera 18-24\|25-34 de V12 | 0.4963 |
| frontera 25-34\|35-49 de V12 | 0.4974 |
| frontera 18-24\|25-34 de A2 | 0.5065 |
| frontera 25-34\|35-49 de A2 | 0.5131 |
| especialista 18-24\|25-34 | 0.6864 |
| especialista 25-34\|35-49 | 0.7218 |

**Reproduccion completa. No hay STOP.**

## 2. La rejilla que cierra el confound

La auditoria de P0 concluyo que la ventaja del especialista viene de la representacion, pero senalo que los dos precedentes cambian a la vez la representacion **y** la familia de modelo. Esta rejilla mide las cuatro celdas sobre la misma tarea, los mismos folds y las mismas filas:

| Frontera | Representacion | Modelo | Especialista | V12 | Delta | Rescate | Dano | Neto |
|---|---|---|---|---|---|---|---|---|
| 18-24|25-34 | densa_279 | LogisticRegression | 0.7136 | 0.4963 | +0.2173 | 13,523 | 1,132 | +12,391 |
| 18-24|25-34 | densa_279 | LinearSVC(C=0.2) | 0.7075 | 0.4963 | +0.2112 | 13,655 | 1,613 | +12,042 |
| 18-24|25-34 | dispersa_tfidf | LogisticRegression | 0.4992 | 0.4963 | +0.0029 | 14,316 | 14,153 | +163 |
| 18-24|25-34 | dispersa_tfidf | LinearSVC(C=0.2) | 0.5003 | 0.4963 | +0.0040 | 14,324 | 14,097 | +227 |
| 25-34|35-49 | densa_279 | LogisticRegression | 0.7493 | 0.4974 | +0.2519 | 15,127 | 875 | +14,252 |
| 25-34|35-49 | densa_279 | LinearSVC(C=0.2) | 0.7444 | 0.4974 | +0.2470 | 15,306 | 1,330 | +13,976 |
| 25-34|35-49 | dispersa_tfidf | LogisticRegression | 0.4996 | 0.4974 | +0.0023 | 14,102 | 13,973 | +129 |
| 25-34|35-49 | dispersa_tfidf | LinearSVC(C=0.2) | 0.4963 | 0.4974 | -0.0011 | 14,061 | 14,122 | -61 |

## 3. Lectura

> **Aviso de validez, y es importante.** Las dos celdas dispersas de esta rejilla **no
> reproducen** a E-028a, que da 0,6864 y 0,7218: aqui salen alrededor de 0,50. La
> reconstruccion del TF-IDF de este script no es la de E-028a, y hasta que se
> identifique la diferencia **esas dos celdas quedan invalidas** y no se usan para nada que
> no sea mostrar que la reconstruccion falla. Los numeros de referencia del especialista
> disperso que se citan en este informe salen de la OOF guardada, verificada al inicio.
> La columna densa, en cambio, es valida: se midio con el protocolo y las variables
> exactos de V13 P3a.

**El efecto no es solo de la representacion.** El mejor especialista sobre las 279 columnas llega a 0.7493, que iguala o supera al peor sobre el TF-IDF (0.4963). Eso refuta la lectura de la auditoria: parte de la ventaja del especialista esta en el modelo y no en la representacion, y P1 a P3 vuelven a ser vias viables porque el objetivo se puede ajustar dentro del espacio que ya tiene V12.

**Lo que habria que revisar**: por que P3a no exploro la regularizacion ni la familia adecuada, y si su resultado es atribuible al protocolo mas que al mecanismo.

