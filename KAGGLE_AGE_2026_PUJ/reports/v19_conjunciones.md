# V13 · P19 — Árboles de conjunciones por función

## 1. Qué se construye y por qué

El material del curso recomienda árboles sobre palabras funcionales. La vía sintáctica que **no** exige etiquetado POS (cerrado desde D-026) es clasificar conjunciones por función, y eso son listas cerradas escritas en el proyecto, que es lo que permite R-1.

La hipótesis pre-registrada es que la función gramatical de las conjunciones varía con la edad. El modelo lineal ya se ha medido tres veces (P1 0,3997, P17 0,1918, P20 0,1270), así que **el árbol es la parte nueva de la prueba**: si hay algo que un lineal no ve, es una interacción entre recuentos por función.

## 2. Cobertura por familia

| Familia | Documentos | Apariciones | Formas sobre el mínimo |
|---|---|---|---|
| causal | 3,823 | 3,978 | 4/5 |
| concesiva | 1,305 | 1,334 | 2/4 |
| condicional | 11,421 | 12,333 | 4/5 |
| temporal | 6,656 | 7,004 | 5/6 |
| comparativa | 9,438 | 10,098 | 5/6 |
| consecutiva | 3,942 | 4,053 | 3/5 |
| adversativa | 8,037 | 8,300 | 3/5 |
| sustitutiva | 41,720 | 53,241 | 3/4 |
| disyuntiva | 7,293 | 7,646 | 2/4 |

## 3. Características

36 columnas: conteo, ratio por cien tokens y presencia para cada familia, más nueve derivadas, entre ellas la **proporción de cada función** (adversativa, concesiva, causal, temporal), el **índice de subordinación** y el **rango del perfil de conjunciones** del documento.

## 4. Cribado de redundancia contra las 279 variables de V12

- Variables redundantes: **0** de 36
- Correlación máxima: **0.5593** · mediana 0.1614

## 5. Control block-only

| Modelo | macro-F1 | std | Suelo de azar | Supera el azar |
|---|---|---|---|---|
| arbol_profundidad_4 | 0.1489 | 0.0008 | 0.1667 | no |
| arbol_profundidad_6 | 0.1512 | 0.0039 | 0.1667 | no |
| arbol_profundidad_8 | 0.1539 | 0.0017 | 0.1667 | no |
| logreg | 0.1539 | 0.0008 | 0.1667 | no |
| linearsvc | 0.1522 | 0.0016 | 0.1667 | no |

## 6. Puerta de rescate

- Rescate: **0** · daño: **0** · **neto +0**
- Umbral: 1,014 filas

- 18-24|25-34: rescate 0 · daño 0 · neto +0
- 25-34|35-49: rescate 0 · daño 0 · neto +0

## 7. Decisión

**Ningún modelo llega al suelo de azar.** El árbol no encuentra nada donde los lineales tampoco encuentran, que es lo coherente con P20: en un corpus de 140 caracteres, las conjunciones no llegan a la frecuencia necesaria para discriminar. La pista se cierra sin gastar la evaluación del meta-modelo.

