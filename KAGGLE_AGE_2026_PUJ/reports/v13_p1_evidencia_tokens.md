# V13 · P1a — Evidencia de edad por token, dentro del fold

## 1. Que se construye y por que va ahora

Las 72 variables supervisadas que ya tiene V12 son `P(clase | entidad)` para menciones, hashtags y fuentes. **No existe `P(clase | token)` para tokens simples.** Es la primera pista de V13 y va primera porque P0b showed que las familias de textometria estan saturadas (VIF mediano 12,6 a 42,8) y las de evidencia supervisada (3,1) y NB-SVM (0,77) son las unicas con margen.

El vocabulario no se declara: se **deriva del propio fold** por frecuencia de documento (minimo 20 documentos). No es un lexico externo ni cerrado, es una estadistica derivada del train, que es lo que permite R-1.

## 2. Protocolo antileakage (§38)

1. Tokenizado unico con `src.lexicons.tokeniza`, el del vectorizador de V12: no se anade ningun vectorizador nuevo. El mapa token -> identificador es una codificacion, no transporta frecuencias ni etiquetas.
2. Para cada fold exterior, el train exterior se reparte en 5 folds interiores (agrupados por duplicados) y las caracteristicas de cada fila de train salen de un `P(clase | token)` calculado sobre las otras cuatro partes: ninguna fila ve su propia etiqueta.
3. Las filas de validacion reciben el perfil calculado sobre todo el train exterior.
4. Vocabulario y frecuencias de documento se calculan siempre sobre el train interior.
5. El test no se toca en ninguna parte.

## 3. Cobertura y vocabulario por fold

| Fold | Filas train | Filas validacion | Columnas con varianza | Tokens banda especifica | Tokens banda frecuente |
|---|---|---|---|---|---|
| 0 | 135,468 | 33,868 | 36 de 36 | 5,508 | 1,513 |
| 1 | 135,468 | 33,868 | 36 de 36 | 5,457 | 1,526 |
| 2 | 135,469 | 33,867 | 36 de 36 | 5,458 | 1,526 |
| 3 | 135,470 | 33,866 | 36 de 36 | 5,509 | 1,524 |
| 4 | 135,469 | 33,867 | 36 de 36 | 5,483 | 1,523 |

## 4. Cribado de redundancia contra las 279 variables de V12

- Variables redundantes (|rho| >= 0,80): **1** de 36
- Correlacion maxima: **0.8883** (`v12_079`)
- Mediana: 0.4708

## 5. Control block-only

Mismo protocolo que la fase de lexicones: LogReg y LinearSVC sobre el bloque, corrected CV agrupado. El suelo de azar es la clase mayoritaria (1/6).

| Modelo | macro-F1 | std | Suelo de azar | Supera el azar |
|---|---|---|---|---|
| logreg | 0.3997 | 0.0042 | 0.1667 | si |
| linearsvc | 0.3917 | 0.0032 | 0.1667 | si |

## 6. Decision

El bloque **supera el suelo de azar en los dos modelos**: hay senal en `P(clase | token)`. Procede P1b (fusion con V12, control de permutado y gate §24).

Caracteristicas: 36 (2 bandas de frecuencia de documento: especifica df en [20, 100), frecuente df en [100, inf)).

Los tokens mas asociados a cada clase estan en `reports/v13_p1_tokens_destacados.csv`. Son **descriptivos**: la prueba de significancia con permutado (regla §27) se hace en P1b, sobre el bloque completo y no sobre tokens sueltos.

