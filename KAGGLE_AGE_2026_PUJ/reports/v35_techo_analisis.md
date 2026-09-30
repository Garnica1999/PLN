# V35 - Acta D-087: analisis del techo alcanzable

**Fecha**: 2026-09-30. Analisis **descriptivo del residuo**: no entrena, no selecciona, no
integra y no decide nada. Su unico proposito es medir un techo. Particion canonica en modo
**solo lectura**; test, holdout y leaderboard intocados.

## Ancla trivial (verificada aqui, no copiada)

- Clase mayoritaria: **13-17**, fraccion **0.170448**
- macro-F1 de predecir siempre esa clase: **0.04854212180687087**
- Coincide con `2p/(6(1+p))` = 0.04854212180687087: **True**

## Base y residuo

- macro-F1 del argmax de `v30_prob`: **0.6134471** (con plug-in: 0.6159732; CV declarado del campeon: 0.6159732)
- Errores: **65403** (38.62 %)

| Tipo de confusion | Filas | % de los errores |
|---|---|---|
| adyacentes | 39202 | 59.94 % |
| lejanos_2 | 15519 | 23.73 % |
| lejanos_3 | 6779 | 10.36 % |
| todos_los_lejanos | 26201 | 40.06 % |

## Oraculo por un solo tipo de error (correccion perfecta, sin reentrenar)

| Tipo corregido al 100 % | Ganancia de macro-F1 |
|---|---|
| adyacentes | +0.2314841 |
| lejanos_2 | +0.0934158 |
| lejanos_3 | +0.0405468 |
| todos_los_lejanos | +0.1570939 |

## La curva decisiva: oraculo restringido a la informacion del campeon

Corrige una fila **solo** si `p(y_true) >= tau`, sin ninguna informacion nueva.

| tau | Filas corregibles | Ganancia | Supera el objetivo |
|---|---|---|---|
| 0.30 | 9683 | +0.0567955 | NO |
| 0.35 | 4516 | +0.0264312 | NO |
| 0.40 | 1383 | +0.0080912 | NO |
| 0.45 | 126 | +0.0007503 | NO |
| 0.50 | 0 | +0.0000000 | NO |

**Techo de la curva: +0.0567955** frente al objetivo **+0.0626**.
Veredicto: NO alcanzable con la informacion actual: un modelo perfecto con la informacion del campeon rinde +0.0567955, por debajo del +0.0626 que exige el objetivo

El objetivo exige ademas ~10.562 correcciones netas = **16.15 % de los 65403 errores**, mientras el mejor oraculo
disponible alcanza 9683 filas.

## Limites estructurales del dato (no del modelo)

- Textos truncados: mediana **93**, maximo **140**, y **2.5 %** en el maximo: perdida de informacion **irrecuperable**.
- `len_le_20`: 5194 filas | error 0.4804 | macro-F1 0.5249
- `len_le_3`: 213 filas | error 0.6056 | macro-F1 0.1363
- `solo_url`: 3891 filas | error 0.6831 | macro-F1 0.1970
- `texto_identico_etiqueta_distinta`: 713 grupos / 1958 filas | error 0.3080 | techo del oraculo de mayoria +0.0018081
- Filas no ASCII: **51** de 169336, luego **no hay via multilingue**.

## La calibracion no puede decidirlo

- Se sostiene en **n = 5** puntos; dominio CV `[0.5344, 0.616]`; `sigma` del residuo 0.001701.
- El objetivo pide CV 0.6786: **36.8 sigma** y **77 % fuera** del rango ajustado.
- el ajuste se sostiene en 5 puntos y su dominio es CV [0.5344, 0.6160]; el objetivo pide +0.0626, que es mas de 2 sigma y 77 % fuera del rango, luego extrapolar la calibracion hasta alli NO es un instrumento de decision.
