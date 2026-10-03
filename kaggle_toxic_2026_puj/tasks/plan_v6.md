# Plan de la fase v6: mecanismos que el proyecto aun no ha evaluado

Estado: **en curso**. Rama `proyecto-kaggle-toxic-2026-v6`, creada desde el cierre de v5.1.

## 1. Que busca esta fase

No mas modelos. **Mecanismos**: representacion, aprendizaje, ponderacion, interaccion,
estructura de etiquetas y modelado local que el proyecto no haya probado de verdad. V03
queda congelada como referencia: `corrected_cv` 0.777211, LB 0.78306.

## 2. Lo que el gap analysis cerro (P0)

| veredicto | lineas |
|---|---|
| **Ejecutables** (11) | A, A2, B, C1, D, E, F, G, H, I, J |
| **Cerradas por duplicacion** (3) | C2, C3, C4 |
| **Cerradas por techo teorico** (2) | D y E (ver seccion 3) |

La identificacion no se hizo de memoria: `v6_00_gap_analysis.py` busca cada mecanismo en el
codigo real con 15 patrones que exigen implementacion, y excluye los scripts de la propia
fase para que no se detecten a si mismos. La evidencia cruda esta en
`reports/v6_verificaciones_gap_v6.json`.

## 3. Lo que el diagnostico cerro (P0-bis)

- **Interseccion de errores: 9 a 12 filas de 1.483** frente a 1.133 donde nadie acierta.
- **Techo teorico de "anadir otro modelo": +0.0027**, por debajo del gate de +0.003.
- Combinando dos clasificadores se **elige** entre sus aciertos, no se suman.
- **D y E se cierran por presupuesto**, no por resultado negativo.
- **TR=1 implica HS=1 de forma determinista** (0 casos de TR=1 con HS=0): invariante que un
  clasificador independiente no expresa. **Eso si sostiene el track B.**
- P(AG|HS=0) = 0.026 con 22 ejemplos: obliga a la forma regularizada B2r.

## 4. Criterio de adopcion (puerta de dos niveles)

- **Gate 1**: `delta >= +0.003` **y** `IC95 que excluye el 0` **y** `>= 4/5 pliegues`.
- **Gate 2**: segunda particion agrupada con otra semilla, solo para finalistas.
- **Adopcion**: supera ambos. El holdout no se usa (agotado 5/5).

## 5. Orden

1. **B** factorizacion condicional: B1, B2, B2r, B3.
2. **A** ponderacion por tamano de grupo: A1, A2, A3 y la variante con tope.
3. **C** solo **C1** (skip-grams con hueco acotado).
4. **I** ponderacion word/char: cuatro corridas, la mas barata que queda.
5. **H** bagging por grupos.
6. **F** ruido por par instancia-etiqueta.
7. **J** aumento, al final y con las restricciones de seguridad de etiqueta.

Un resultado negativo **cierra** su pregunta. No se abre una rejilla nueva sin hipotesis.

## 6. Presupuesto

24 experimentos principales y 6 de reserva. No es obligatorio gastarlos.

## 7. Prohibiciones (se maintains)

Sin transformers, LLM ni embeddings preentrenados de terceros. Sin etiquetas externas, sin
pseudo-labeling, sin usar `toxic_test.csv` para entrenar, sin labels de test, sin usar el
leaderboard como validacion, sin usar el holdout. Los embeddings de FastText de v5/v5.1
**si** son reutilizables, porque los entrenamos nosotros desde cero. Sin submissions
automaticas.
