# V20 · F3c — Candidato de umbrales anidados: submission preparada y NO enviada

**Este script no envia nada.** El fichero queda listo, validado con 8 de 8 checks y con su SHA256. Enviarla es una decision de Carlos.

## 1. Que es el candidato

```
(1/4) x LogReg + (1/4) x MLP(64) + (1/4) x arbol de V19 + (1/4) x cadena de umbrales anidados
```

V19 conserva sus tres miembros y sus pesos originales **intactos**: solo se le anade un miembro con peso propio.

## 2. Lo que dice el OOF y lo que puede decir el leaderboard

| Magnitud | Valor |
|---|---|
| CV del candidato | **0.6097782** |
| CV de V19 | 0.6079388 |
| delta | **+0.0018** |
| folds positivos | 5 de 5 |
| IC95 contra V19 | [+0,0012, +0,0025] |
| neto de filas en el train | +294 |
| filas de la submission que cambian | **3,546** de 112,892 (3.14%) |
| LB de V19 | **0.62578** |
| LB previsto por la calibracion | **0.62923** |

## 3. El riesgo, medido y no supuesto

La mejora esperada es de **+0,0019** en LB. Pero el residuo de esa misma calibracion sobre V19 fue de **-0,0016**, con oscilacion de **0,0034** entre modelos vecinos. **La mejora esperada es del mismo orden que la incertidumbre del aparato.**

Lo que diferencia este candidato de los cuatro de V19.x que se descartaron es que aquellos no cambiaban de mecanismo: una **cadena de cinco discriminadores binarios, sin un solo modelo multiclase, alcanza 0.6046893 por si sola, por encima del 0.6005017 de V12**, y en tres de los cinco umbrales sus discriminadores superan a V12 medido en la misma fila.

## 4. Distribucion de clases

| Clase | Candidato | V19 |
|---|---|---|
| 13-17 | 18,384 | 18,521 |
| 18-24 | 20,440 | 20,263 |
| 25-34 | 17,372 | 17,430 |
| 35-49 | 16,323 | 16,412 |
| 50-64 | 20,790 | 20,599 |
| 65-xx | 19,583 | 19,667 |

## 5. Registro

- Fichero: `age_submission_v20_candidato_umbrales.csv`
- SHA256: `2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d`
- Timestamp de construccion: 2026-09-27T11:06:20+00:00
- Configuracion del cuarto miembro: cinco LightGBM con {"objective": "multiclass", "num_class": 6, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 8, "max_depth": 3, "min_child_samples": 200, "colsample_bytree": 0.5, "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 1.0, "n_jobs": 32, "random_state": 42, "verbose": -1}
- Hipotesis que prueba: que un clasificador de umbral anidado, con el 100 % de los datos, cobra decisiones de frontera que el clasificador multiclase y el par adyacente dejan sueltas, y que por tanto aporta informacion distinta como cuarto miembro.
- **Estado: NO ENVIADO.**
