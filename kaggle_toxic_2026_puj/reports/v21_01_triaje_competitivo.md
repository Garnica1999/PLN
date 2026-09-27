# V21-01: triaje competitivo de las configuraciones ya entrenadas

Generado por `experiments/v21_01_triaje_competitivo.py`. **Ningun modelo se entrena.**

## 1. La tabla que decide

| candidata | DeltaF1 | rescates | danos | **neto** | celdas | filas | DeltaF1/celda | **neto/celda** | novedad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **FUSION-WS-3** | +0.005747 | 74 | 63 | **+11** | 137 | 119 | 4.20e-05 | **8.03 %** | ya entrenada |
| FUSION-VS-100 | +0.005046 | 77 | 67 | **+10** | 144 | 123 | 3.50e-05 | 6.94 % | ya entrenada |
| L1 C=4.0 | +0.006431 | 208 | 205 | **+3** | 413 | 296 | 1.56e-05 | 0.73 % | nueva, P1-03 |

Referencia: V03 = 0.7772114525165721, LB 0.78306.

**El orden por metrica y el orden por perfil de cambios no coinciden.** L1 gana en
DeltaF1 y pierde en todo lo demas: mueve **tres veces mas celdas** que WS-3 para un saldo
neto **tres veces menor**, y su eficiencia por celda es 2,7 veces menor.

## 2. El detalle por etiqueta, que es donde esta la advertencia

| candidata | etiqueta | cambios | rescates | danos | neto | tasa positiva |
|---|---|---:|---:|---:|---:|---|
| FUSION-WS-3 | HS | 64 | 33 | 31 | +2 | 0.4572 -> 0.4693 (+0.0121) |
| FUSION-WS-3 | TR | 26 | 17 | 9 | **+8** | 0.3001 -> 0.3041 (+0.0040) |
| FUSION-WS-3 | AG | 47 | 24 | 23 | +1 | 0.3776 -> 0.3823 (+0.0047) |
| FUSION-VS-100 | HS | 58 | 30 | 28 | +2 | 0.4572 -> 0.4653 (+0.0081) |
| FUSION-VS-100 | TR | 29 | 16 | 13 | +3 | 0.3001 -> 0.3075 (+0.0074) |
| FUSION-VS-100 | AG | 57 | 31 | 26 | +5 | 0.3776 -> 0.3823 (+0.0047) |
| L1 C=4.0 | HS | 164 | 84 | 80 | +4 | 0.4572 -> 0.4626 (+0.0054) |
| L1 C=4.0 | TR | 103 | 50 | 53 | **-3** | 0.3001 -> 0.3358 (**+0.0357**) |
| L1 C=4.0 | AG | 146 | 74 | 72 | +2 | 0.3776 -> 0.3965 (+0.0189) |

**El riesgo de L1 tiene nombre: es el patron de V20.** L1 desplaza la tasa de TR en
+0.0357, que es ocho veces mas que el desplazamiento de WS-3 en cualquier etiqueta, y es
la unica etiqueta cuyo saldo es **negativo**. V20 cambio 51 celdas, todas de HS y todas de
0 a 1, con un desplazamiento de tasa de +0.0442 y un saldo de -6: el leaderboard bajo
0.00643. La mecanica es la misma: un cambio grande y en una sola direccion sobre una
etiqueta hunde su F1 mas de lo que sube el resto.

WS-3 y VS-100 mueven las tasas entre +0.0040 y +0.0121, tienen saldo positivo en las tres
etiquetas, y concentran 26 cambios en TR, que es donde mas gaina (+8 de 26).

## 3. Solapamiento: WS-3 y VS-100 no son dos oportunidades

| par | celdas comunes | solo la primera | solo la segunda | proporcion de solape |
|---|---:|---:|---:|---:|
| WS-3 vs VS-100 | 110 | 27 | 34 | **80.3 %** |
| WS-3 vs L1 C=4.0 | 67 | 70 | 346 | 48.9 % |
| VS-100 vs L1 C=4.0 | 73 | 71 | 340 | 50.7 % |

WS-3 y VS-100 hacen casi la misma perturbacion. Tratarlas como dos candidatas
independientes seria contarlas dos veces. En cambio, L1 **si** es una perturbacion
independiente: solo solapa la mitad con cada una de las otras.

Eso abre una posibilidad que no estaba sobre la mesa: **combinar WS-3 con L1**. Pero el
informe de P1-03 ya midio que combinar dos mecanismos que reducen capacidad efectiva
empeora (-0.0085 para 1+2), y no se ha medido esta combinacion concreta. Queda como
pregunta abierta, no como candidata.

## 4. Los controles de cordura, que es lo que hace creible el resto

| control | calculado | publicado | desviacion | dentro de tolerancia |
|---|---:|---:|---:|---|
| FUSION-B0-200 | 0.781766 | 0.7818 | 0.000034 | si |
| FUSION-WS-3 | 0.782959 | 0.7830 | 0.000041 | si |
| FUSION-VS-100 | 0.782257 | 0.7823 | 0.000043 | si |

Los tres reproducen con desviacion inferior a 0.00005, con el protocolo de pliegues
canonico y los umbrales congelados de V03.

### 4.1 Tres fallos reales que el control de cordura detecto antes de medir nada

El control no solo verifico el numero: **impidio que se publicara una tabla falsa**. En el
proceso aparecieron tres diferencias entre una reimplementacion razonable y el pipeline de
V5, y las tres cambiaban el resultado:

1. **La union del bloque denso.** V5 une con `sparse.hstack` sobre `csr_matrix`. Una
   reimplementacion con `np.hstack(...).toarray()` entrega a liblinear una matriz densa de
   27.431 columnas, lo que cambia el camino de optimizacion y el resultado.
2. **El conjunto de tipos consultados.** V5.1 construye `tipos` como
   `sorted(vocabulario_toxic(read_train()))`, es decir **6.059 tipos del train**, y consulta
   al modelo solo esos. `vectores_documento` promedia unicamente los tokens presentes.
3. **El `.vec` no sirve para eso.** V5.1 usa `vectores_de_tipos`, que invoca al binario de
   fastText en `print-word-vectors` sobre el `.bin`. El `.vec` solo exporta los 100.098 tipos
   del vocabulario del modelo, y de los 6.059 tipos consultados **solo 3.777 aparecen en
   el `.vec`**: los 2.282 restantes existen porque el modelo compone vector con
   subpalabras, que es justo lo que aporta FastText. Con el `.vec` el pooling cambia y
   FUSION-B0-200 daba **0.7728** en lugar de 0.7818.

La leccion es la misma de V20: **las metricas de esta linea son sensibles a detalles de
construccion que no estan en el numero publicado**. Cualquier comparacion futura de una
configuracion de embedding debe declarar de donde salen los vectores de tipo.

## 5. Conclusion del triaje

1. **WS-3 tiene el mejor perfil competitivo de las tres candidatas**: el mayor
   DeltaF1 por celda (2,7 veces el de L1), el mejor saldo neto en proporcion (8,03 % frente
   a 0,73 %), saldos positivos en las tres etiquetas y desplazamientos de tasa pequenos.
2. **L1 C=4.0 no es la mejor candidata pese a tener el mayor DeltaF1**: su saldo de +3 en
   413 celdas es microscopico, y su desplazamiento de tasa en TR con saldo negativo repite
   el mecanismo que hundio a V20 en el leaderboard.
3. **VS-100 no es una segunda oportunidad**: solapa el 80,3 % con WS-3. En el mejor de los
   casos,WS-3 la domina, y en el peor son la misma perturbacion.
4. El coste de medir WS-3 es **cero**: ya esta entrenado, ya esta reevaluado aqui, y su
   artefacto tiene sha256 registrado.

**Pendiente de decision**: la eleccion entre WS-3 y L1 no se cierra con DeltaF1, sino con
el criterio de transferencia. Corresponde al especialista competitivo.
