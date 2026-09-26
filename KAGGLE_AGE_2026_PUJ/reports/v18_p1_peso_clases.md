# V18 · P1 — Peso de clases consciente de la frontera

## 1. Diseno

El control es un **V12-equivalente**: mismas 279 variables, mismos folds agrupados, mismos dos
modelos y mismo promedio 0,5 + 0,5, cambiando **solo** el vector de pesos de clase. Los tres
pesos estan **pre-registrados** y no se eligen mirando ningun resultado, asi que no hace falta
particion interior para elegirlos:

| Nivel | Peso de 18-24, 25-34 y 35-49 | Resto |
|---|---|---|
| nivel_0_base | 1.0 | 1,0 |
| nivel_1_moderado | 1.5 | 1,0 |
| nivel_2_fuerte | 2.5 | 1,0 |

**El handicap, declarado antes de ejecutar**: las clases estan balanceadas (16,04 % a 17,04 %, razon 1,062), asi que no hay desbalance que corregir. El unico efecto posible del peso de clases es desplazar el equilibrio entre clases, que es justo lo que D-037 intento 33 veces como capa de decision a posteriori.

## 2. Comprobacion de fidelidad del control

El nivel base se reentrena con una reimplementacion local que acepta `sample_weight`, porque la clase de produccion `EncodedMLP` no lo admite. Para que la comparacion sea atribuible a los pesos y no al cambio de codigo, ese nivel tiene que reproducir la OOF guardada de V12. **Diferencia: 0.000000** (tolerancia 0.0005) -> **cumple**.

## 3. Resultado

| Nivel | macro-F1 | Delta | Folds positivos | IC95 | 18-24\|25-34 | 25-34\|35-49 | Adopta |
|---|---|---|---|---|---|---|---|
| nivel_0_base | 0.6005 | +0.0000 | 0/5 | [+0.0000, +0.0000] | 0.4963 | 0.4974 | **no** |
| nivel_1_moderado | 0.6001 | -0.0004 | 3/5 | [-0.0016, +0.0009] | 0.5441 | 0.5341 | **no** |
| nivel_2_fuerte | 0.5841 | -0.0164 | 0/5 | [-0.0182, -0.0147] | 0.5786 | 0.5699 | **no** |
| V12 (referencia) | 0.6005 | - | - | - | 0.4963 | 0.4974 | - |

## 4. F1 por clase, que es donde se ve el intercambio

| Nivel | 13-17 | 18-24 | 25-34 | 35-49 | 50-64 | 65-xx |
|---|---|---|---|---|---|---|
| nivel_0_base | 0.671 | 0.5059 | 0.4858 | 0.5593 | 0.6549 | 0.7261 |
| nivel_1_moderado | 0.6544 | 0.5229 | 0.4844 | 0.5568 | 0.6562 | 0.7261 |
| nivel_2_fuerte | 0.6051 | 0.5208 | 0.4807 | 0.5408 | 0.637 | 0.7199 |
| V12 (referencia) | 0.671 | 0.5059 | 0.4858 | 0.5593 | 0.6549 | 0.7261 |

## 5. Lectura

**El peso de clases no aporta.** Ningun nivel alcanza el gate exploratorio de +0,0015. Y la tabla de F1 por clase muestra el motivo: subir el peso de las tres clases centrales sube su F1 y **baja el de las otras tres en la misma medida**, porque la macro-F1 es una media y el juego es de suma cero. Un objetivo de pesos no puede ganar aqui por construccion: solo puede redistribuir.

**Que significa para la fase**: el handicap que la auditoria senalo era real. La via P1 se cierra con resultado estable negativo, y la siguiente palanca no es reordenar el equilibrio entre clases, sino **cambiar la funcion de perdida de verdad**, que es lo que miden P2 y P3.

