# P60-H - Transporte optimo entre distribuciones sobre las unidades del texto

Generado por `experiments/p60_07_transporte.py`. Fecha 2026-09-28T16:44:59+00:00.

## La condicion de novedad, y por que hay que verificarla midiendo

El gap check de esta pista es raro. El transporte optimo entre un objeto de prueba y un
prototipo de clase parece pertenecer a la familia de vecinos, prototipos, centroides y
distancia al centroide, y esa familia esta cerrada en TOXIC. Pero ese cierre aplica
cuando el objeto de prueba es un PUNTO. Con un delta de Dirac en x,

```text
W1(delta_x, nu) = media de d(x, v) sobre el soporte de nu = distancia al centroide
```

Luego OT **colapsa de forma demostrable** a la familia cerrada en ese caso, y solo puede
aportar algo si el objeto de prueba es el mismo una **distribucion**. Eso da un test que se
puede ejecutar ANTES de entrenar: puntuar la misma clase con un objeto de Dirac y con uno
distribucional, y ver si coinciden. Si coinciden, la pista cierra sin experimento. Si no,
el transporte esta haciendo algo que un centroide no puede hacer.

## La representacion, que es lo que hace distribuido al objeto de prueba

```text
tweet  ->  sus tokens, cada uno proyectado al espacio SVD que induce el TF-IDF de V03
clase  ->  la medida agrupada de los tweets de TRAINING de esa clase
```

Un tweet son unas quince unidades y una clase unos miles, luego se comparan dos medidas
empiricas y no dos vectores. El coste de suelo es euclideo en el espacio SVD, los pesos
son `1 + log1p(tf sublineal del token)`, y no hay ningun recurso externo.

## Por que Wasserstein seccionado y no Sinkhorn

Con soportes de quince y varios miles de puntos, Sinkhorn no es barato ni numericamente
seguro con estas proporciones de clase. En una dimension el transporte optimo es la
distancia L1 entre las funciones de cuantil, que es **exacta** y no necesita iterar. El
estimador seccionado usa 64 direcciones fijadas de antemano.

## H1 GAP CHECK, medido antes de entrenar

El gap check es **autocalibrado**: se mide primero el sesgo numerico de la propia
metrica, comparando la version Dirac con el coste exacto calculado a mano, que es la
media de distancias a las muestras de la clase. Sin esa referencia, una diferencia de
dos por ciento podria ser senal o podria ser la rejilla de cuantiles.

| etiqueta | sesgo de la rejilla | diferencia Dirac/distribucion | razon | n |
|---|---:|---:|---:|---:|
| HS | 0.013 por ciento | 44.833 por ciento | 3547.21 veces | 120 |
| TR | 0.017 por ciento | 41.557 por ciento | 2427.42 veces | 120 |
| AG | 0.017 por ciento | 44.288 por ciento | 2552.22 veces | 120 |

Sesgo medio **0.016 por ciento**, diferencia media **43.559 por ciento**, razon **2842.28 veces**.

**GAP CHECK SUPERADO:** la version distribucional es una divergencia genuina y no ruido de discretizacion.

## H2 Resultados

```text
brazo 0 (centroide, familia cerrada)  corrected_cv 0.5409847232
brazo 1 (transporte, distribucion)    corrected_cv 0.4686780062
delta DIVERGENCIA PURA                -0.0723067170
delta vs V03                          -0.3085334463
pliegues que mejoran                  0/5
rescate 391 | dano 2655 | neto -2264 | cambian 3046
V03 referencia                        0.7772114525
```

| etiqueta | F1 transporte | F1 centroide |
|---|---:|---:|
| HS | 0.5591 | 0.6082 |
| TR | 0.3562 | 0.4616 |
| AG | 0.4907 | 0.5532 |

## Gate

| criterio | valor | superado |
|---|---|---|
| efecto de la divergencia >= +0,003 | -0.072307 | **False** |
| al menos 4 de 5 pliegues | 0/5 | **False** |
| gate completo | | **False** |

Bootstrap NO ejecutado: el gate no se supero.

## Decision

```text
decision  CLOSED_BY_DUPLICACION. El gap check mostro que la version Dirac del transporte es la familia cerrada, y la version distributiva no mejora sobre ella
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p60_07_transporte.py`. Sin push.*