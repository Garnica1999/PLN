# V17 · P1c — Regla conjunta: las dos señales en sentido contrario

Inerte. Construye el sistema con compuerta completo y mide su macro-F1 global de verdad.

> **Aviso.** El optimo de cada rejilla esta sesgado hacia arriba, porque se elige mirando las mismas filas que despues se miden. **Ninguno de los numeros de la rejilla es resultado.** Los que valen son el techo, la comparacion entre univariada y conjunta, y el numero publicable saldra de P9 con inner OOF.

## 1. El hallazgo que cambia la regla

Las dos señales de P8 apuntan en sentidos **contrarios**, y eso es lo que faltaba:

- el **margen del especialista** es mayor en las filas que arregla, asi que manda escucharle cuando el especialista esta seguro;
- el **acuerdo entre miembros** es menor en las filas que arregla, asi que manda escucharle cuando el ensemble esta repartido.

La compuerta de una sola señal que propone el brief era, por tanto, incompleta: le faltaba la mitad de la condicion.

## 2. Resultados por frontera (delta global, milesimas)

| Regla | 18-24|25-34 | 25-34|35-49 |
|---|---|---|
| techo (oracle, no alcanzable) | +10.7 | +7.7 |
| mejor univariada (sesgada) | +0.0 | +0.0 |
| mejor conjunta (sesgada) | +0.0 | +0.0 |

**Mejor regla conjunta en cualquier frontera: +0.0** frente a **+0.0** de la mejor univariada. Gate exploratorio +1.5, gate de adopcion +3.0.

## 3. Todas las reglas, con su intercambio

| Frontera | Regla | Activadas | Fraccion | Rescate | Dano | Neto | Delta global |
|---|---|---|---|---|---|---|---|
| 18-24|25-34 | techo (oracle) | 1,830 | 1.1% | 1,830 | 0 | +1,830 | +10.7 |
| 18-24|25-34 | univariada: margen > 0.80 | 819 | 0.5% | 1 | 0 | +1 | +0.0 |
| 18-24|25-34 | conjunta: acuerdo < 0.20 y margen > 0.80 | 1 | 0.0% | 1 | 0 | +1 | +0.0 |
| 18-24|25-34 | conjunta: acuerdo < 0.35 y margen > 0.80 | 1 | 0.0% | 1 | 0 | +1 | +0.0 |
| 18-24|25-34 | conjunta: acuerdo < 0.50 y margen > 0.80 | 1 | 0.0% | 1 | 0 | +1 | +0.0 |
| 18-24|25-34 | conjunta: acuerdo < 0.65 y margen > 0.80 | 6 | 0.0% | 1 | 0 | +1 | +0.0 |
| 18-24|25-34 | conjunta: acuerdo < 0.20 y margen > 0.35 | 329 | 0.2% | 83 | 122 | -39 | -0.2 |
| 18-24|25-34 | conjunta: acuerdo < 0.35 y margen > 0.35 | 463 | 0.3% | 103 | 164 | -61 | -0.4 |
| 18-24|25-34 | univariada: margen > 0.35 | 6,569 | 3.9% | 103 | 166 | -63 | -0.4 |
| 18-24|25-34 | conjunta: acuerdo < 0.50 y margen > 0.35 | 514 | 0.3% | 103 | 166 | -63 | -0.4 |
| 18-24|25-34 | conjunta: acuerdo < 0.65 y margen > 0.35 | 835 | 0.5% | 103 | 166 | -63 | -0.4 |
| 18-24|25-34 | conjunta: acuerdo < 0.20 y margen > 0.15 | 1,832 | 1.1% | 458 | 608 | -150 | -0.9 |
| 18-24|25-34 | univariada: acuerdo < 0.20 | 3,754 | 2.2% | 852 | 1,077 | -225 | -1.3 |
| 18-24|25-34 | conjunta: acuerdo < 0.20 y margen > 0.00 | 3,754 | 2.2% | 852 | 1,077 | -225 | -1.3 |
| 18-24|25-34 | conjunta: acuerdo < 0.35 y margen > 0.15 | 2,841 | 1.7% | 637 | 871 | -234 | -1.4 |
| 18-24|25-34 | conjunta: acuerdo < 0.50 y margen > 0.15 | 3,306 | 2.0% | 671 | 907 | -236 | -1.4 |
| 18-24|25-34 | conjunta: acuerdo < 0.65 y margen > 0.15 | 4,587 | 2.7% | 678 | 923 | -245 | -1.5 |
| 18-24|25-34 | univariada: margen > 0.15 | 15,635 | 9.2% | 681 | 930 | -249 | -1.5 |
| 18-24|25-34 | univariada: acuerdo < 0.35 | 6,788 | 4.0% | 1,388 | 1,805 | -417 | -2.5 |
| 18-24|25-34 | conjunta: acuerdo < 0.35 y margen > 0.00 | 6,788 | 4.0% | 1,388 | 1,805 | -417 | -2.5 |
| 18-24|25-34 | univariada: acuerdo < 0.50 | 8,569 | 5.1% | 1,600 | 2,099 | -499 | -2.9 |
| 18-24|25-34 | conjunta: acuerdo < 0.50 y margen > 0.00 | 8,569 | 5.1% | 1,600 | 2,099 | -499 | -2.9 |
| 18-24|25-34 | univariada: acuerdo < 0.65 | 11,531 | 6.8% | 1,737 | 2,336 | -599 | -3.5 |
| 18-24|25-34 | conjunta: acuerdo < 0.65 y margen > 0.00 | 11,531 | 6.8% | 1,737 | 2,336 | -599 | -3.5 |
| 18-24|25-34 | univariada: margen > 0.00 | 25,477 | 15.0% | 1,830 | 2,485 | -655 | -3.9 |
| 25-34|35-49 | techo (oracle) | 1,239 | 0.7% | 1,239 | 0 | +1,239 | +7.7 |
| 25-34|35-49 | univariada: margen > 0.80 | 3,528 | 2.1% | 0 | 0 | +0 | +0.0 |
| 25-34|35-49 | conjunta: acuerdo < 0.20 y margen > 0.80 | 0 | 0.0% | 0 | 0 | +0 | +0.0 |
| 25-34|35-49 | conjunta: acuerdo < 0.35 y margen > 0.80 | 0 | 0.0% | 0 | 0 | +0 | +0.0 |
| 25-34|35-49 | conjunta: acuerdo < 0.50 y margen > 0.80 | 0 | 0.0% | 0 | 0 | +0 | +0.0 |
| 25-34|35-49 | conjunta: acuerdo < 0.65 y margen > 0.80 | 5 | 0.0% | 0 | 0 | +0 | +0.0 |
| 25-34|35-49 | univariada: margen > 0.35 | 10,110 | 6.0% | 79 | 101 | -22 | -0.1 |
| 25-34|35-49 | conjunta: acuerdo < 0.50 y margen > 0.35 | 389 | 0.2% | 79 | 101 | -22 | -0.1 |
| 25-34|35-49 | conjunta: acuerdo < 0.65 y margen > 0.35 | 619 | 0.4% | 79 | 101 | -22 | -0.1 |
| 25-34|35-49 | conjunta: acuerdo < 0.35 y margen > 0.35 | 335 | 0.2% | 76 | 99 | -23 | -0.1 |
| 25-34|35-49 | conjunta: acuerdo < 0.20 y margen > 0.35 | 247 | 0.1% | 61 | 86 | -25 | -0.1 |
| 25-34|35-49 | conjunta: acuerdo < 0.20 y margen > 0.15 | 1,350 | 0.8% | 297 | 441 | -144 | -0.8 |
| 25-34|35-49 | conjunta: acuerdo < 0.35 y margen > 0.15 | 2,121 | 1.3% | 429 | 625 | -196 | -1.1 |
| 25-34|35-49 | conjunta: acuerdo < 0.50 y margen > 0.15 | 2,521 | 1.5% | 464 | 675 | -211 | -1.2 |
| 25-34|35-49 | conjunta: acuerdo < 0.65 y margen > 0.15 | 3,415 | 2.0% | 475 | 709 | -234 | -1.4 |
| 25-34|35-49 | univariada: margen > 0.15 | 16,861 | 10.0% | 475 | 718 | -243 | -1.4 |
| 25-34|35-49 | univariada: acuerdo < 0.20 | 2,663 | 1.6% | 539 | 796 | -257 | -1.5 |
| 25-34|35-49 | conjunta: acuerdo < 0.20 y margen > 0.00 | 2,663 | 1.6% | 539 | 796 | -257 | -1.5 |
| 25-34|35-49 | univariada: acuerdo < 0.35 | 4,804 | 2.8% | 906 | 1,286 | -380 | -2.2 |
| 25-34|35-49 | conjunta: acuerdo < 0.35 y margen > 0.00 | 4,804 | 2.8% | 906 | 1,286 | -380 | -2.2 |
| 25-34|35-49 | univariada: acuerdo < 0.50 | 6,186 | 3.7% | 1,070 | 1,549 | -479 | -2.8 |
| 25-34|35-49 | conjunta: acuerdo < 0.50 y margen > 0.00 | 6,186 | 3.7% | 1,070 | 1,549 | -479 | -2.8 |
| 25-34|35-49 | univariada: acuerdo < 0.65 | 8,338 | 4.9% | 1,191 | 1,789 | -598 | -3.5 |
| 25-34|35-49 | conjunta: acuerdo < 0.65 y margen > 0.00 | 8,338 | 4.9% | 1,191 | 1,789 | -598 | -3.5 |
| 25-34|35-49 | univariada: margen > 0.00 | 24,001 | 14.2% | 1,239 | 1,947 | -708 | -4.2 |

## 4. Tercera frontera, solo diagnostico

No se ha entrenado nada: se mide el margen que E-028a ya produce para `35-49|50-64`. V12 acierta 0.6124 de esas filas y el especialista 0.7735, con un techo de 0.8165. El area bajo la curva del acuerdo (leida al reves) es 0.3298 y la del margen 0.7523.

Alguno de los dos valores no llega al area util, asi que la tercera frontera **no se abre** por ahora, que es lo que el brief pide: no gastar por simetria. La decision se difiere a que las centrales funcionen.

## 5. Decision

**La regla no llega ni al gate exploratorio con el optimo sesgado.** Con esta formulacion, P9 no esta indicado: la separabilidad de P8 no se traduce en macro-F1 global. Antes de pagar el protocolo anidado habria que buscar una señal mas, no afinar la que hay.

Figura: `reports/figures/v17_p1c_reglas.png`.
