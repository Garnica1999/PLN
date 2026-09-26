# V17 · P1b — Curva de compromiso de la compuerta

Inerte: no se ha entrenado nada. La compuerta se construye con el margen del especialista que ya estaba calculado, y V12 decide en el resto de las filas.

## Advertencia que va antes que los numeros

El umbral de **0.05** es el mejor de esta tabla y por eso **esta sesgado hacia arriba**: se eligio mirando las mismas filas que despues se miden. No es un resultado y no puede citarse como tal. Lo que si son resultados son la existencia de la curva, su forma y su techo. P3 elegira el umbral dentro del train exterior, y ese es el momento en que el numero valdra.

## 1. Curva medida (delta de macro-F1 global, en milesimas)

| Frontera | umbral 0.02 | umbral 0.05 | umbral 0.10 | umbral 0.20 | umbral 0.35 | umbral 0.60 |
|---|---|---|---|---|---|---|
| 18-24\|25-34 | +0.0 | +0.2 | +0.1 | -0.0 | -0.1 | -1.0 |
| 25-34\|35-49 | +0.0 | +0.1 | +0.0 | +0.0 | -0.2 | -0.8 |

## 2. Rescate y dano por umbral

| Frontera | Umbral | Activadas | Rescate | Dano | Neto |
|---|---|---|---|---|---|
| 18-24\|25-34 | 0.02 | 608 | 98 | 93 | +5 |
| 18-24\|25-34 | 0.05 | 1,545 | 242 | 213 | +29 |
| 18-24\|25-34 | 0.10 | 2,891 | 432 | 410 | +22 |
| 18-24\|25-34 | 0.20 | 5,478 | 750 | 751 | -1 |
| 18-24\|25-34 | 0.35 | 8,929 | 1,134 | 1,156 | -22 |
| 18-24\|25-34 | 0.60 | 13,487 | 1,498 | 1,674 | -176 |
| 25-34\|35-49 | 0.02 | 419 | 60 | 55 | +5 |
| 25-34\|35-49 | 0.05 | 966 | 138 | 117 | +21 |
| 25-34\|35-49 | 0.10 | 1,913 | 261 | 264 | -3 |
| 25-34\|35-49 | 0.20 | 3,626 | 475 | 483 | -8 |
| 25-34\|35-49 | 0.35 | 5,794 | 715 | 764 | -49 |
| 25-34\|35-49 | 0.60 | 8,487 | 945 | 1,098 | -153 |

## 3. Lectura

**La curva es monotona y sube deprisa al principio y despacio despues**, que es la forma tipica de un intercambio rescate-contra-dano: al principio se activan filas en las que el especialista acierta y V12 tambien, que es ruido puro, y a partir de cierto umbral solo quedan filas en las que V12 dudaba de verdad.

| Frontera | Mejor delta de la curva | Gate exploratorio (+1,5) | Gate de adopcion (+3) |
|---|---|---|---|
| 18-24\|25-34 | +0.2 | no supera | no supera |
| 25-34\|35-49 | +0.1 | no supera | no supera |

**La curva no llega ni al gate exploratorio en ninguna frontera.** El techo de P1 era real pero la confianza de V12 no alcanza para cobrarlo. Eso apuntaria a que el problema no es de enrutamiento sino de informacion, y P3 deberia empezar por medir si algun otro indicador separa mejor.


Figura: `reports/figures/v17_p1b_curva_compuerta.png`.

