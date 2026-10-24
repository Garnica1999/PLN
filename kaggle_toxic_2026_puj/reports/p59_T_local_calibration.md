# P59-T - Calibracion del score local: CLOSED_BY_DUPLICATION

Generado por `experiments/p59_08_calibracion_local.py`. Fecha 2026-10-20T02:48:46+00:00. Sin entrenar.

## Que proponia T

Tomar el score local crudo de B1, `max_j p(ventana_j)`, y calibrarlo con Platt ajustado solo con el training fold, y compararlo con V03 y con B1 crudo a los umbrales congelados.

## T0 control

```text
V03 0.7772114525165721  reproduce exacto: True
B1 (insumo de T) 0.7765868373082134
```

## T1 gap check: EXISTE EQUIVALENCIA FUNCIONAL

### Pierna 1, algebrica

Platt es una transformacion afin en el logit, `p_cal = sigmoid(a*logit(p_raw) + b)`. Con un umbral fijo `t`:

```text
p_cal >= t  <=>  a*logit(p_raw) + b >= logit(t)
           <=>  p_raw >= sigmoid((logit(t) - b)/a)
```

Aplicar Platt y mantener `t` **es exactamente** reseleccionar el umbral por etiqueta sobre el score crudo. No aporta informacion, solo mueve el corte. Y el encargo excluye el threshold sweep al definir T.

### Pierna 2, precedente, y es un cierre de raiz

| id | que midio | resultado |
|---|---|---|
| V8-09 | F1 frente a la tasa de positivos predicha, sobre quince sondas independientes | es **unimodal y V03 esta exactamente en el maximo**; cierra de raiz calibraciones, umbrales y pesos por etiqueta |
| V10-04 | Platt, isotonica recortada y beta de tres parametros sobre los scores de V03, calibrador anidado, cinco semillas | la calibracion funciona: ECE 0,058293 -> 0,024801. Pero el delta medio anidado de Platt es **+0,000148** con IC95 [-0,005221, +0,005932]. Cerrada con clausula pre-registrada: no se reintenta con mas variantes |
| E2-14-MEJOR-M2 | calibracion de M2 | +0,0018 con 3 de 5 pliegues, CERRADA |
| E2-14-MEJOR-META | calibracion de META | -0,0017, CERRADA |
| V4-H01 | Platt sobre los scores del meta de 36 rasgos | **-0,0171**, empeora |
| V9-02 | Platt mas EM de Saerens | sonda no adoptada ni enviada |

La calibracion no esta sin explorar aqui: esta explorada cuatro veces y cerrada de raiz por una medicion que no depende de ajustar nada.

## Por que U no esta cerrada

U fusiona un segundo score dentro del primero, y eso cambia el **ranking**, que ninguna reparametrizacion monotona puede hacer. Esa es la linea algebraica exacta que separa U de T, y por eso la fusion continua no cae en el cierre de V8-09 ni en la clausula de V10-04.

## Decision

```text
decision  CLOSED_BY_DUPLICATION. 0 experimentos consumidos. No es que la calibracion del score local haya fallado: es que no es un experimento, porque Platt a umbral fijo ES un cambio de umbral, y esa familia esta cerrada de raiz por V8-09 y de forma especifica por V10-04. Sin sonda, sin entrenamiento, sin submissions. V03 remains the best verified leaderboard result.
```

> V03 remains the best verified leaderboard result.

*Seccion generada por `experiments/p59_08_calibracion_local.py`. Sin push.*