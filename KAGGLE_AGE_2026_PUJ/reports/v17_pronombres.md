# V13 · P17a — Pronombres de persona

## 1. Que se construye y por que

El artículo de referencia (Moreno-Sandoval, Pomares-Quimbaya y Alvarado-Valencia, 2021) identifica los pronombres de persona como su variable más predictiva, con la tercera persona singular a la cabeza. El gap audit de P0 verificó que de sus 18 variables, cinco no existían en el corpus: V14-V18, los pronombres. Esta es la pista que las cubre.

El vocabulario es **cerrado y propio** (regla R-1): palabras funcionales del español escritas en este archivo, no un lexicón externo. Como el corpus es ASCII sin tildes (D-009), `tu` y `tú` son la misma cadena y las formas que solo se distinguen por la tilde quedan fuera.

**Decisión de diseño**: `el` y `los` son artículo o pronombre según el contexto, y `el` es la palabra más frecuente del corpus. Incluirlas en un contador de tercera persona mediría longitud de texto, no persona, así que se declaran aparte como constante de normalización y las cinco categorías usan solo formas no ambiguas.

## 2. Cobertura por categoría

| Categoría | Documentos con la categoría | Apariciones | Formas sobre el mínimo |
|---|---|---|---|
| primera_singular | 26,399 | 33,594 | 7/10 |
| primera_plural | 2,320 | 2,452 | 4/4 |
| segunda | 15,406 | 19,615 | 9/13 |
| tercera_singular | 40,158 | 50,419 | 4/9 |
| tercera_plural | 743 | 762 | 1/2 |
| impersonal | 29,344 | 36,290 | 4/4 |
| ambigua | 74,269 | 114,140 | 10/11 |

Umbral de permanencia de una forma: 200 documentos. El detalle de cobertura forma por forma está en `reports/v17_pronombres_cobertura.csv`.

## 3. Características

26 columnas: conteo, ratio por cien tokens y presencia para cada una de las cinco categorías de persona y para el impersonal, más la proporción de artículos ambiguos y cinco derivadas:

- **distancia_social**: tercera persona menos (primera más segunda). Es la lectura psicológica del artículo: cuanto más lejos del yo, más probable es que el tema sea la tercera persona.
- **proporcion_tercera** y **proporcion_primera** de los pronombres del documento.
- **indice_segunda_primera**: segunda sobre primera, con suavizado para que los documentos sin pronombres no explode.
- **n_pronombres** y su ratio por token.

## 4. Cribado de redundancia contra las 279 variables de V12

- Variables redundantes (|rho| >= 0,80): **0** de 26
- Correlacion maxima: **0.4757** (`v12_090`)
- Mediana: 0.3218

## 5. Control block-only

| Modelo | macro-F1 | std | Suelo de azar | Supera el azar |
|---|---|---|---|---|
| logreg | 0.1918 | 0.0013 | 0.1667 | si |
| linearsvc | 0.1881 | 0.0021 | 0.1667 | si |

## 6. Decision

El bloque **supera el suelo de azar en los dos modelos**: procede P17b (fusion con V12, control de permutado, gate §24 y reporte por cabeza, que es donde se mide la predicción pre-registrada de que la señal entra por la rama lineal y se diluye en la MLP).

Comparación de referencia: el bloque de evidencia por token (P1a) da 0,3997 y el mejor bloque léxico de la fase cerrada 0,2033, con el mismo suelo de azar.

