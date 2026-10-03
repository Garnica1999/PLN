# P62 - Pista L, tokens condicionados por POS

Generado por `experiments/p62_03_pista_l.py`. Fecha 2026-09-29T05:18:44+00:00.

## Que decia la EDA

```text
palabra sola              +0.3773
categoria sola            +0.0095
palabra y categoria       +0.4525
synergy                   +0.0657  (17.41 por ciento de la senal de la palabra)
```

La synergy es la parte que no se puede escribir como funcion de la palabra y de la
categoria por separado. V03 sabe la palabra y tira la categoria; el canal de E2-03
sabe la categoria y tira la palabra. Esta pista le da el par.

## El hueco, comprobado contra el codigo

`p62_00_auditoria_y_gap.py` verifica contra el fuente que ningun token `palabra__POS`
se emite en el proyecto, y ABORTA si aparece. No es una busqueda de nombres.

## El confound de escala, que es real

Cada bloque TF-IDF se normaliza por separado, luego V03 tiene norma de fila raiz de 2
y anadir un tercer bloque la sube. Eso es un reescalado uniforme de los
coeficientes, que equivale a cambiar C, y el parrafo 36 lo prohibe. La matriz
candidata se reescala para que su norma media de fila sea la del control. En el
control el factor es exactamente uno, luego el control es V03 intacto.

## Los tres brazos

| brazo | corrected_cv | delta vs V03 |
|---|---:|---:|
| control, V03 sin canal | 0.7772114525 | +0.0000000000 |
| pista L, + palabra__POS | 0.7676396879 | -0.0095717646 |
| control aleatorio, igual dimension | 0.7656259836 | -0.0115854689 |

El control reproduce V03 exactamente: True.

## El control de dimensionalidad, que es el que decide

Un candidato que anade rasgos puede ganar por una razon que no tiene nada que ver con
linguistica: mas dimensiones, mas dispersidad, otra regularizacion efectiva. Por eso
el tercer brazo anade un bloque de tokens ALEATORIOS del mismo numero y con las
mismas categorias.

```text
aporte de la pista L                    -0.0095717646
aporte del bloque aleatorio             -0.0115854689
diferencia                              +0.0020137043
```

### La lectura en tres partes, que conviene no reducir a un signo

```text
(1) coste de anadir un bloque de rasgos   +0.0115854689
(2) valor de la estructura linguistica    +0.0020137043
(3) fraccion repagada                     17.4 por ciento
    saldo                                 -0.0095717646
```

lectura en tres partes. (1) Anadir un bloque de 1196 rasgos CUESTA +0.0115854689 de macro-F1, y ese coste lo paga incluso un bloque sin significado linguistico. (2) La estructura linguistica REAL vale +0.0020137043 respecto a ese bloque aleatorio, luego la interaccion palabra-POS existe y es medible. (3) Pero repaga solo el 17.4 por ciento del coste, de modo que el saldo es -0.0095717646. La senal existe y es pequena frente al precio de pagarla

## Control de vocabulario

| brazo | rasgos por pliegue | densidad de no nulos |
|---|---|---|
| control | [20441, 21044, 20802, 21038, 20982] | [0.008744, 0.008712, 0.008771, 0.008685, 0.008775] |
| pista_L | [21637, 22257, 22012, 22263, 22185] | [0.008598, 0.008572, 0.008627, 0.008541, 0.008635] |
| control_aleatorio | [21809, 22406, 22160, 22402, 22343] | [0.008678, 0.008662, 0.008715, 0.008634, 0.008719] |

El canal usa min_df=2 y se induce DENTRO de cada pliegue, luego ningun rasgo de su
vocabulario ve las filas validadas.

## Probabilidad, umbral y error de alta confianza

| et | errores V03 | errores candidato | alta conf. corregidos | introducidos | neto |
|---|---:|---:|---:|---:|---:|
| HS | 316 | 316 | 7 | 8 | -1 |
| TR | 164 | 174 | 0 | 4 | -4 |
| AG | 277 | 300 | 2 | 6 | -4 |

Una correccion que solo ocurra en errores de BAJA confianza seria un efecto de
decision y no evidencia de nueva representacion (parrafo 57). Por eso la tabla
separa las dos clases.

## Gate

```text
delta -0.009572 >= 0.003: False
pliegues que mejoran 1/5 >= 4: False
superado: False
bootstrap: NO EJECUTADO
```

## Decision

```text
CLOSED. La interaccion palabra-POS no aporta sobre V03
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p62_03_pista_l.py`. Sin push.*