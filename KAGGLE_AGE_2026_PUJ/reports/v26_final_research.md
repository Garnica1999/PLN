# V26 · Informe final de advanced research

**Estado: PASS** | una sola hipotesis ejecutada | 0 submissions | V20 intacto con LB **0,62689**.

## 1. Que se hizo

- **Gap analysis** de 15 familias buscando equivalencia funcional en el codigo. **5** quedan genuinamente abiertas: H02, H03, H12, H14, H11.
- **Una** hipotesis ejecutada: `H02`, la reescalada afin de las 84 columnas que son predicciones de modelos base.

## 2. Resultado

| Modelo | corrected_cv | Delta | Folds + | rescue | damage | net | filas |
|---|---|---|---|---|---|---|---|
| T0 V20 control | 0.6097782 | - | 0/5 | - | - | 0 | 0 |
| T1 evidencia reescalada | 0.609777 | **-0.000001** | **2/5** | 555 | 552 | **+3** | 1,809 |

Gate del punto 29: delta >= +0.003 y 4/5 folds. **No se cumple por las dos vias.**

## 3. La lectura, que es el resultado de verdad

**La transformacion si cambia el modelo y no cambia la metrica.** 1,809 filas cambian, que es el 1.07% del train, y el saldo es **+3**. La evidencia del ensemble es **invariante a su escala en todo lo que la metrica mide**.

Y hay un subresultado mas valioso que el propio experimento: **un arbol NO es invariante a una transformacion monotona en esta implementacion de LightGBM**. Con `colsample_bytree=0,5` y `subsample=0,8`, un z-score afin cambia la prediccion bit a bit, con diferencia maxima 0,219 sobre 135.468 filas. La invariancia teorica es un argumento que habria cerrado esta hipotesis sin gastar presupuesto, y era **falso**. Por eso se midio antes de decidir, y por eso el gap analysis exige evidencia y no argumentos.

## 4. Respuesta a la pregunta del punto 43

> Donde esta el cuello de botella restante de AGE: en la representacion de las features, en la combinacion de las predicciones del ensemble, en la calibracion de la evidencia o en otra propiedad del meta-modelo?

**La evidencia disponible, acumulada y reproducible, va en una sola direccion:**

| Candidate a cuello de botella | Evidencia | Veredicto |
|---|---|---|
| Representacion de las features | V25: podar con supervision empeora -0,000629 y elimina senal, no ruido (D-060) | **no** |
| Escala de la evidencia del ensemble | V26-01: reescalar cambia 1.809 filas y mueve +3 | **no** |
| Formulacion ordinal | cinco subfamilias cerradas entre D-054 y D-058 | **no** |
| No linealidad del meta-model | D-053 y D-056: ninguna clase de funcion supero al boosting | **no** |
| Seleccion de la evidencia por clase | D-037 y D-047 cerrados con numero | **no** |

**Conclusion**: el cuello de botella restante **no esta en ninguna de las cuatro propiedades que el brief enumera**, y todas tienen evidencia negativa reproducible. Lo que queda sin descartar es una limitacion de los **datos**: el 38,3 % de las filas las equivocan V19 y V20 juntas, y ningun modelo del historial las corrige. Eso ya no es un problema de modelo, es un problema de **senal disponible**.

## 5. Lo que queda abierto, sin abrirlo

| ID | Familia | Por que sigue abierta |
|---|---|---|
| H03 | percentil | colapsa empates, luego aporta informacion que H02 no tiene |
| H11 | monotone constraints nativas de LightGBM | `monotone_constraints` tiene **cero** usos en el proyecto entero; es el unico mecanismo del punto 11 sin probar |
| H12 | fiabilidad por miembro y clase | no existe el concepto en el proyecto |
| H14 | normalizar las salidas del blend | el blend suma cuatro probabilidades con escalas y calibraciones distintas, y nadie lo ha tratado |

**Ninguna se ejecuta sin autorizacion.** El punto 41 del brief prohibe convertir una observacion en otro experimento automaticamente.

## 6. Cumplimiento

- Compliance: **todas cumplidas** (15 comprobaciones sobre codigo y artefactos).
- Unicode: **0 caracteres introducidos**.
- Lint con `ruff` sobre `F` y `E9`: **0 errores**.
- Ejecucion real de las celdas del notebook: ver `experiments/v26_02_auditoria.json`.
- **0 submissions**, test intacto, holdout intacto, V20 intacto.
