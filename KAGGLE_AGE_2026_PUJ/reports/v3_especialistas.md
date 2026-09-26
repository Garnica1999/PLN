# V13 · P3a — Especialistas binarios de frontera sobre las variables de V12

## 1. Qué se entrena

Para cada par de frontera, un modelo **nuevo** entrenado sobre las filas del train cuyo valor verdadero es una de las dos clases, con las mismas 279 variables de V12 y su propio fold. En inferencia solo redecide **entre las dos clases del par**, y solo cuando V12 ya ha dicho que la fila es de una de ellas: es la forma `gate(especialista, V12)` de §37, y por construcción no puede tocar las filas en las que V12 no está en el par.

Esto **no** es la línea que D-037 cerró: allí se aplicaron 33 reglas a posteriori sobre las probabilidades de V12; aquí se entrena un modelo con sus propios datos de entrenamiento.

## 2. Resultado por frontera y por modelo

| Frontera | Modelo | Accuracy V12 en el par | Accuracy del especialista | Mejora | Rescate | Daño | Neto |
|---|---|---|---|---|---|---|---|
| 18-24|25-34 | logreg | 0.4963 | **0.4944** | -0.0019 | 1,023 | 1,132 | -109 |
| 18-24|25-34 | mlp | 0.4963 | **0.4946** | -0.0017 | 1,482 | 1,578 | -96 |
| 25-34|35-49 | logreg | 0.4974 | **0.4942** | -0.0032 | 695 | 875 | -180 |
| 25-34|35-49 | mlp | 0.4974 | **0.4945** | -0.0029 | 1,262 | 1,425 | -163 |

## 3. Efecto global de la puerta

- macro-F1 de V12: **0.6005**
- macro-F1 con la puerta (mejor modelo por frontera): **0.5993** (-0.0012)
- Filas cuya prediccion cambia: 6,944

## 4. Puerta

| Criterio | Valor | Cumple |
|---|---|---|
| mejora >= 0.01 en 18-24|25-34 | -0.0017 | **no** |
| mejora >= 0.01 en 25-34|35-49 | -0.0029 | **no** |
| rescate neto total >= 1000 | -259 | **no** |
| las dos fronteras con neto positivo | {'18-24|25-34': -96, '25-34|35-49': -163} | **no** |

## 5. Decisión

**La puerta no se supera**: la pista P3 se cierra sin pagar la evaluación completa. Un especialista que solo repita lo que ya hace V12 no aporta, y un especialista que redecida peor destruye filas que V12 tenía bien.
Con esto queda agotado también el segundo tipo de cambio posible. P15 había dicho que los miembros no tienen señal de confianza y P16d que el bloque de P1 tampoco; ahora se añade que **ni siquiera un modelo entrenado específicamente para la frontera mejora la decisión dentro de ella**. Las tres vías que quedaban (más variables, más modelos, más estructura de decisión) fallan con evidencia medida, no con suposiciones.

