# P66 - Analisis de hipotesis: la fusion en el espacio de scores

Generado por `experiments/p66_00_audit_gap.py`. Fecha 2026-09-29T15:33:33+00:00.

## Por que esta fase existe, en una frase

P64 y P65 midieron cinco representaciones que baten a **los dos controles**, el de
capacidad y el de la confianza de V03, y ninguna mejoro a V03 al concatenarla. P66
prueba lo unico que no se probo: si esa misma evidencia funciona como una correccion
pequena en el espacio de scores, sin tocar el espacio TF-IDF.

## La auditoria del baseline

```text
V03 corrected_cv  0.7772114525165721
V03 sha256        7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e
registro          364 filas, 364 identificadores
holdout 0 | test 0 | submissions 0 | Kaggle 0 | leaderboard 0
```

## El inventario es PEOR de lo esperado, y se reporta en vez de rodearlo

El encargo dice que los scores OOF de los candidatos deben reutilizarse cuando exista uno
valido. Al comprobar el repositorio, **ninguno de los cinco scripts de P64 y P65 escribio
sus probabilidades a disco**. Publicaron metricas y scores por pliegue y las dejaron en
memoria. Ademas `features/` esta en `.gitignore`, luego los bloques de features que si
escribieron tampoco estan versionados.

```text
artefactos .npz en disco: oof_v03_final.npz
scripts de P64/P65 que guardan probabilidades: 3 de 16
```

**Consecuencia: los scores hay que REGENERARLOS.** La alternativa seria inventarlos, que
es peor. Se reutiliza el codigo de P64 y P65 tal cual, con las mismas semillas y los
mismos pliegues, y la regeneracion tiene que **reproducir el corrected_cv que cada fase
reporto**. Si un numero no cuadra, la regeneracion esta mal y la fase para en vez de
seguir sobre scores no verificados.

Ese es el motivo de que esto sea un paso aparte y no una regeneracion silenciosa dentro
del script de fusion.

## Los siete candidatos, por prioridad

| candidato | delta que reporto su fase | brecha contra el margen | por que |
|---|---:|---:|---|
| MI_arbol | -0.0091686799 | +0.0093823920 | P64 lo encontro por encima del margen de V03 en +0,0093823920, que es la brech |
| negacion | -0.0021826774 | +0.0088975231 | unico candidato con las DOS brechas positivas a la vez, capacidad y margen |
| word_1_3 | -0.0039520688 | +0.0071281317 | la brecha contra el control de anchura mas alta, +0,0596671770 |
| cohesion | -0.0027090062 | +0.0083711943 | seis cantidades, todas medibles, con el bloque ya construido |
| huecos_1_2 | -0.0010515908 | +0.0100286097 | el mas cercano a V03 de todos los candidatos, solo -0,0010515908 |
| contraste | -0.0023851933 | +0.0086950072 | el mas PARCORELECTADO en P65 y el unico con un solo canal de 53 columnas |
| huecos_6_10 | -0.0074391131 | +0.0036410874 | el mas negativo de los siete, y por eso el control de que la fusion no lo arra |

## La tabla de familias

| familia | estado | por quien |
|---|---|---|
| blend lineal de un score local con V03 mediante un alpha | EXACT_DUPLICATE | V6-05 y P59-09 |
| correccion residual sobre el score de V03 | GENUINELY_NEW | ninguno. La palabra residuo aparece en el codigo en otros sent |
| correccion sensible a la longitud, continua y sin puerta | GENUINELY_NEW | el especialista duro se midio en P65-04 y fue BIT A BIT igual  |
| calibracion anidada de probabilidades antes de fusionar | PARTIAL_PRECEDENT | V10-05 toco la capa de decision por shift de prevalencia, y R1 |
| stacking o meta-modelo general | EXACT_DUPLICATE | V4-X01 y las familias meta36, y P64-05 con el arbol |
| selector keep/flip | EXACT_DUPLICATE | V18 y V50, y P64-19 lo excluyo explicitamente |
| control de score barajado | GENUINELY_NEW | ninguno. Existia el control de BLOQUE aleatorio en la etapa de |
| unica informacion mas alla de V03 en el score del candidato | GENUINELY_NEW | P63-07 y P64-03 midieron el AUC de la evidencia frente al ERRO |

**4 genuinamente nuevas, 3 duplicadas exactas, 1 con precedente parcial.**

## La diferencia central, que es lo que hace nueva la fase

Un **blend** que falla dice que la MEZCLA estaba mal. Un **residuo** que falla dice que el
candidato es **REDUNDANTE**. Son conclusiones distintas, y solo la segunda habla de
informacion.

Mezclar un segundo score dentro de V03 pregunta si el segundo score es util. Residualizar
pregunta si el segundo score contiene algo que el score de V03 **no predice**, y
corrige solo esa parte. La segunda pregunta es la que P64 y P65 dejaron abierta.

Lo que ya se cerro y no se reabre: el blend lineal con un alpha, medido por V6-05 con
+0,0014800075 y 2 de 5, y por P59-09 otra vez. Se ejecutara aqui como **control**, no
como novedad. Y quedan excluidos por el encargo el stacking general y el selector
keep/flip, que ademas ya se habian cerrado como V18, V50 y P64-19.

## Las reglas de la fase, fijadas antes

- solo el score de V03 y el del candidato, **nunca** las matrices TF-IDF del candidato
- todo lo aprendido va por nested: la outer **solo evalua**
- umbrales 0,44 / 0,51 / 0,46 congelados
- P66 **no abre ninguna familia de representaciones nueva**
- control de score barajado obligatorio en toda fusion promising

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p66_00_audit_gap.py`. Sin push.*