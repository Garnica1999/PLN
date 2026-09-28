# V26 · Gap analysis del espacio experimental restante

**Este script no entrena.** Su objeto es decidir que hipotesis merece un experimento, buscando **equivalencia funcional** y no solo coincidencia de nombre.

## 0. La distincion que gobierna el analisis

**V20 ya es un meta-modelo.** De las 279 features, **84 son predicciones de modelos base** (78 de los 14 miembros del ensamble de V12 y 6 de NB-SVM), y el meta-modelo de V20 es **LightGBM**: cinco binarios para la cadena de umbrales y uno mas para el arbol del blend. La pregunta de V26 no es si construir un meta-modelo, sino si su **representacion de la evidencia** es explotable de otra forma.

### Dos afirmaciones que hubo que medir en vez de suponer

1. **Una transformacion monotona no cambia un arbol.** Es cierto en teoria y **falso en la practica de esta LightGBM**: con `colsample_bytree=0,5` y `subsample=0,8`, un z-score afin cambia la prediccion **bit a bit**, con diferencia maxima **0,219** sobre las **135.468** filas del fold. Un argumento de invariancia **no** es un argumento para no experimentar.
2. **Un percentil no es una transformacion monotona estricta.** Colapsa valores a empates, luego pierde orden fino, y el arbol tambien cambia: diferencia maxima **0,166**.

Ambas mediciones se hicieron con el percentil y la media **aprendidos solo del training fold**, que es la unica forma legal.

## 1. Baseline

- V20 reproduce **0.6097782** | coincide con 0,6097782: **True**.
- Hash de la submission del champion: `2c964af78c90bf08` | LB **0,62689**.
- Folds: [33868, 33868, 33867, 33866, 33867], los mismos de V20.

## 2. La tabla de hipotesis

| ID | Familia | Estado | Nuevo | Prioridad |
|---|---|---|---|---|
| H02 | probability normalization (afin) | **GENUINELY_OPEN** | si | 1 |
| H03 | rank transformation / percentil | **GENUINELY_OPEN** | si | 2 |
| H12 | OOF reliability modeling | **GENUINELY_OPEN** | si | 2 |
| H14 | rank transformation DENTRO del blend | **GENUINELY_OPEN** | si | 2 |
| H11 | monotone constraints en LightGBM | **GENUINELY_OPEN** | si | 3 |
| H01 | probability calibration | **CLOSED** | no | 0 |
| H04 | model weighting | **CLOSED** | no | 0 |
| H05 | diversity weighting | **CLOSED** | no | 0 |
| H06 | stacking de segundo nivel | **CLOSED** | no | 0 |
| H07 | interaction terms | **CLOSED** | no | 0 |
| H08 | nonlinear meta-model | **CLOSED** | no | 0 |
| H09 | class-specific meta models | **CLOSED** | no | 0 |
| H10 | ordinal constraints | **CLOSED** | no | 0 |
| H13 | residual correction | **CLOSED** | no | 0 |
| H15 | supervised feature selection | **CLOSED** | no | 0 |

### El detalle de cada una

#### H02 · probability normalization (afin)

- **Hipotesis**: centrar y escalar las probabilidades de los miembros antes del meta-modelo
- **Busqueda de equivalencia funcional**: grep de `StandardScaler` devuelve 67 scripts, pero ninguno los aplica a las 84 columnas de predicciones de miembros antes del meta-modelo de V20; V16 los aplica dentro del combinador logistico, que es un meta-modelo distinto (lineal, no arbol)
- **Conclusion**: No se midio nunca sobre el meta-modelo real de V20, y la invariancia que lo haria trivial es FALSA en esta implementacion de LightGBM
- **Razon de la prioridad 1**: track prioritario del brief, y la medicion de invariancia lo hace no trivial

#### H03 · rank transformation / percentil

- **Hipotesis**: sustituir la magnitud absoluta de la confianza por su posicion relativa en la distribucion
- **Busqueda de equivalencia funcional**: los 8 scripts que contienen `percentile` o `quantile` los usan sobre LONGITUD de palabra, nunca sobre probabilidades de modelos
- **Conclusion**: Genuinamente nueva, y no trivial: el percentil colapsa empates y si cambia el arbol
- **Razon de la prioridad 2**: mismo mecanismo que H02 pero con colapso de empates, luego informacion adicional

#### H12 · OOF reliability modeling

- **Hipotesis**: estimar fiabilidad por miembro y por clase, y usarla como evidencia
- **Busqueda de equivalencia funcional**: `reliability` y `calibracion_error` no existen como concepto en el proyecto; los 4 matches de `reliability` son la palabra española `familia`
- **Conclusion**: Genuinamente nueva, y no es solo calibracion porque es fiabilidad POR MIEMBRO Y POR CLASE
- **Razon de la prioridad 2**: el brief la situa como M4, solo si M2/M3 muestran senal

#### H14 · rank transformation DENTRO del blend

- **Hipotesis**: aplicar la normalizacion a los cuatro miembros del blend, no solo a las columnas de entrada
- **Busqueda de equivalencia funcional**: ningun script normaliza las salidas de logreg, mlp y arbol antes de combinarlas; el blend es una media aritmetica de cuatro escalas distintas
- **Conclusion**: Genuinamente nueva y distinta de H02: H02 acts antes del meta-modelo, esto despues
- **Razon de la prioridad 2**: el blend suma cuatro probabilidades con escalas y calibraciones distintas

#### H11 · monotone constraints en LightGBM

- **Hipotesis**: imponer la monotonia DENTRO del meta-modelo con el parametro nativo
- **Busqueda de equivalencia funcional**: `monotone_constraints` devuelve **cero** resultados en todo el proyecto. V20 impone la monotonia DESPUES, con maximo acumulado sobre cinco modelos independientes
- **Conclusion**: Genuinamente nueva: nadie ha usado la herramienta que LightGBM ofrece para esto
- **Razon de la prioridad 3**: mecanismo distinto al de V20, pero aplica a las fronteras y no a la evidencia de los miembros

#### H01 · probability calibration

- **Hipotesis**: isotonica, Platt o temperatura sobre las salidas del ensemble
- **Busqueda de equivalencia funcional**: grep de `isotonic|calibracion|temperature` devuelve 21 scripts; phase15_calibracion_scores y submission_v01_calibracion son medidas sobre la salida
- **Conclusion**: Medido y cerrado: la recalibracion de las salidas no aporta sobre el macro-F1
- **Razon de la prioridad 0**: familia agotada, con numero, en fases anteriores

#### H04 · model weighting

- **Hipotesis**: reponder los miembros del ensemble segun su contribucion
- **Busqueda de equivalencia funcional**: v16_arquitectura.py midio `peso_aprendido` (delta -0,0003) y `combinador_logistico` (+0,0006); V18 midio re-peso de clases; V22 intento ponderar las fronteras
- **Conclusion**: Medido tres veces con tres familias distintas, ninguna mejora el global
- **Razon de la prioridad 0**: familia agotada

#### H05 · diversity weighting

- **Hipotesis**: ponderar por complementariedad en lugar de por fiabilidad
- **Busqueda de equivalencia funcional**: 29 scripts contienen `diversidad` o `complementariedad`; V17 completo midio interseccion de errores por miembro y la funcion de decision local (D-047)
- **Conclusion**: La complementariedad es real y grande pero no identificable, segun D-047
- **Razon de la prioridad 0**: familia agotada, con la razon ya identificada

#### H06 · stacking de segundo nivel

- **Hipotesis**: apilar las salidas del ensemble en un modelo de segundo nivel
- **Busqueda de equivalencia funcional**: 72 scripts contienen `stacking` o `meta`; el propio `advanced_02_features.py` apila 14 miembros sobre 279 columnas para producir las 279 que consume V20
- **Conclusion**: V20 ya ES el segundo nivel: las 84 columnas de miembros son la salida de un apilado previo
- **Razon de la prioridad 0**: V20 ya es el resultado de este stacking, luego repetirlo seria redundante

#### H07 · interaction terms

- **Hipotesis**: productos explicitos entre las probabilidades de dos miembros
- **Busqueda de equivalencia funcional**: 12 scripts contienen `interaccion` o `interaction`; advanced_05c combino 273, 324 y 329 columnas con interacciones y no supero la referencia
- **Conclusion**: Medido: ampliar la matriz con interacciones no mejora
- **Razon de la prioridad 0**: ademas el meta-modelo de arbol ya aprende interacciones por splits

#### H08 · nonlinear meta-model

- **Hipotesis**: un meta-modelo no lineal distinto del LightGBM de V20
- **Busqueda de equivalencia funcional**: 9 scripts contienen meta-modelos no lineales; V21 uso HistGradientBoosting, RBF-SVM, MLP y uniones lineales sobre las 279, y todos quedaron por debajo de LightGBM (D-056)
- **Conclusion**: Ninguna clase de funcion no lineal supero al boosting; ExtraTrees tambien quedo 0,0236 debajo (D-053)
- **Razon de la prioridad 0**: la pregunta ya tiene respuesta y es negativa

#### H09 · class-specific meta models

- **Hipotesis**: un meta-modelo distinto por clase
- **Busqueda de equivalencia funcional**: 66 scripts; D-037 cerro nueve hipotesis de capa de decision ordinal y V17 cerro tres meta-combinadores por frontera (P1b, P5, P9)
- **Conclusion**: Cerrado con numero en D-037 y D-047
- **Razon de la prioridad 0**: familia agotada

#### H10 · ordinal constraints

- **Hipotesis**: imponer el orden entre las salidas de las cinco fronteras
- **Busqueda de equivalencia funcional**: 100 scripts; V20 las impone con maximo acumulado, D-054 cerro la reconstruccion, D-055 los pesos, D-056 los clasificadores de B1, D-057 el modelo compartido y D-058 el CORAL
- **Conclusion**: La familia ordinal esta cerrada en sus cinco subfamilias
- **Razon de la prioridad 0**: no abrir una sexta sin evidencia nueva

#### H13 · residual correction

- **Hipotesis**: corregir con un modelo de residuo lo que el ensemble no captura
- **Busqueda de equivalencia funcional**: 20 scripts contienen `residual` o `residuo`; V18 midio destilacion y V15 la poda
- **Conclusion**: La destilacion de V18 es la forma canonica de este mecanismo y cerro con 0 adopciones
- **Razon de la prioridad 0**: familia agotada

#### H15 · supervised feature selection

- **Hipotesis**: podar o seleccionar features con supervision
- **Busqueda de equivalencia funcional**: V25 midio mutual information top 50 por ciento dentro de cada fold (D-060); V15 midio poda no supervisada (D-045)
- **Conclusion**: Cerrado: la seleccion no elimina ruido, elimina senal, y las probabilidades de los miembros son lo que el MI ordena alto
- **Razon de la prioridad 0**: no reabrir, salvo defecto metodologico demostrado

## 3. Las cuatro familias genuinamente abiertas

- **H02 probability normalization (afin)**: No se midio nunca sobre el meta-modelo real de V20, y la invariancia que lo haria trivial es FALSA en esta implementacion de LightGBM
- **H03 rank transformation / percentil**: Genuinamente nueva, y no trivial: el percentil colapsa empates y si cambia el arbol
- **H12 OOF reliability modeling**: Genuinamente nueva, y no es solo calibracion porque es fiabilidad POR MIEMBRO Y POR CLASE
- **H14 rank transformation DENTRO del blend**: Genuinamente nueva y distinta de H02: H02 acts antes del meta-modelo, esto despues
- **H11 monotone constraints en LightGBM**: Genuinamente nueva: nadie ha usado la herramienta que LightGBM ofrece para esto

## 4. Duplicaciones que deben bloquearse

- otra K u otro porcentaje en feature selection (V25 cerrado, D-060)
- otra variante de feature selection equivalente: RFE, chi2, ANOVA, variantes de MI
- otra formulacion ordinal: PAVA, coste ordinal, CORAL, umbrales anidados, modelo compartido
- otra combinacion lineal del ensemble: V16 midio combinador logistico y peso aprendido
- otra especializacion de la primera frontera ordinal: B1 (V21 cerrado, D-056)
- varias veces seguidas el mismo eje de parametros del champion (V19.x cerrado, D-052)

## 5. Hipotesis seleccionada

**`V26_SELECTED_TRACK = H02`**

- **Mecanismo**: centrado y escalado de las 84 columnas de predicciones de modelos base, con la media y la desviacion estimadas SOLO dentro de cada training fold, antes de alimentar los cinco LightGBM de V20
- **Por que no es trivial**: un arbol es invariante a una transformacion monotona en teoria, pero la medicion muestra que en esta LightGBM con colsample_bytree 0,5 y subsample 0,8 un z-score afin cambia la prediccion bit a bit, con diferencia maxima 0,219 sobre las 135.468 filas del fold
- **Coste**: bajo: una transformacion por fold, sin entrenamiento nuevo ni features nuevas
- **Riesgo**: puede empeorar: la escala de las probabilidades de los miembros esta calibrada por construccion y recentrarla puede desalinear el meta-modelo

### Alternativas descartadas por ahora

- **H03**: mismo mecanismo con colapso de empates; se reserva
- **H11**: nueva pero aplica a las fronteras y no a la evidencia de los miembros
- **H12**: el brief la condiciona a que M2 o M3 muestren senal
- **H14**: nueva y distinta, se reserva como segundo experimento si H02 no muestra senal

**Presupuesto**: {'principal': 8, 'reserva': 4, 'absoluto': 12, 'primer_experimento': 1}

El primer experimento tendra exactamente **V20 como control y una unica transformacion nueva**. No se cambia simultaneamente la representacion, el meta-modelo, la regularizacion ni los thresholds.
