# V13 · P16 — ¿La información de los bloques nuevos ya estaba dentro de V12?

## 1. Por qué esta pregunta

Cuatro pistas consecutive con el mismo patrón: el bloque tiene señal aislada y al combinarlo con V12 desaparece (P1: 0,3997 aislado y -0,0002 al fusionar; P17: 0,1918 y neto -64.075; P20: 0,1270; lexicones: 0,2033 y -0,0018). Hay dos explicaciones incompatibles y el resto del plan depende de cuál sea:

1. **La información ya estaba en V12.** Los miembros de V12 son clasificadores de TF-IDF, que es una forma de estimar `P(clase|token)`. Si es así, ningún bloque del mismo tipo puede aportar y el espacio de representaciones está agotado.
2. **La información es nueva pero el combinador no la usa.** Si es así, el problema es de arquitectura y la pista de especialistas de frontera (P3) tiene sentido.

Este experimento las separa sin entrenar el meta-modelo: reconstruye cada bloque como combinación lineal de las 279 variables de V12, con ridge ajustada **dentro de cada fold**, y vuelve a medir el control *block-only* sobre la reconstrucción. Si el control reconstruido reproduce el original, la explicación es la primera.

## 2. Cuánta información de cada bloque contiene V12

| Bloque | R2 mediano | R2 máximo | Variables con R2 > 0,5 | Variables con R2 > 0,8 |
|---|---|---|---|---|
| P1 tokens | 0.095 | 0.424 | 0 de 36 | 0 de 36 |
| P17 pronombres | 0.021 | 0.079 | 0 de 36 | 0 de 36 |
| P20 modalidad | 0.009 | 0.042 | 0 de 36 | 0 de 36 |

## 3. Control del bloque reconstruido

El control que importa: si el bloque se reemplaza por su reconstrucción a partir de V12 y el rendimiento no cae, la información ya estaba dentro.

| Bloque | Modelo | Control original | Control reconstruido desde V12 | Suelo de azar |
|---|---|---|---|---|
| P1 tokens | logreg | 0.3915 | **0.0485** | 0.1667 |
| P1 tokens | linearsvc | 0.3841 | **0.0485** | 0.1667 |
| P17 pronombres | logreg | 0.1918 | **0.0485** | 0.1667 |
| P17 pronombres | linearsvc | 0.1881 | **0.0485** | 0.1667 |
| P20 modalidad | logreg | 0.1270 | **0.0485** | 0.1667 |
| P20 modalidad | linearsvc | 0.1259 | **0.0485** | 0.1667 |

## 4. Lectura

- La explicación 1 (V12 ya tenía la información) se sostiene para: **P1 tokens**: el control reconstruido conserva el -51 % de la mejora sobre el azar; **P17 pronombres**: el control reconstruido conserva el -470 % de la mejora sobre el azar; **P20 modalidad**: el control reconstruido conserva el -7849999900 % de la mejora sobre el azar.
- Consecuencia para el plan: si la explicación 1 domina, **no tiene sentido abrir más pistas de variables nuevas del mismo tipo** (P2 de dominios de URL, P6 de centroides, P18, P21, P24) porque caerán en el mismo pozo, y el esfuerzo debe ir a donde la decisión difiere: la frontera (P3), que es un problema de arquitectura y no de representación.
- Este control no sustituye al control de permutado: no prueba que la fusión fuera ineficaz por redundancia, prueba que la información no era accesible para el modelo lineal. Ambas cosas pueden ser ciertas a la vez.

