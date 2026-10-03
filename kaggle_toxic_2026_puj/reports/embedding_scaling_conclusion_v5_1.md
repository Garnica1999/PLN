# v5.1: conclusion sobre el escalado del corpus (EBD-23)

La seccion 12 de la spec solo permite considerar una expansion del corpus si se dan
cinco condiciones a la vez. Se evaluan una por una, con la evidencia de v5 y de v5.1.

## 1. La curva medida

| corpus | tokens | cobertura por tipo | embedding solo | fusion | delta |
|---|---|---|---|---|---|
| wikipedia 5M | 5M | 0.6302 | 0.6670 | 0.7757 | -0.0015 |
| wikipedia 10M | 10M | 0.6839 | 0.6530 | 0.7818 | +0.0046 |
| wikipedia 50M | 50M | 0.7703 | 0.6687 | 0.7803 | +0.0031 |
| web mC4-es 10M | 10M | 0.8256 | 0.6638 | 0.7812 | +0.0040 |
| mixto 20M | 20M | 0.8413 | 0.6658 | 0.7776 | +0.0004 |

## 2. Las cinco condiciones

1. **Una arquitectura nueva mejora claramente respecto del baseline**: **NO**. Ninguna
   de las nueve configuraciones nuevas es adoptable; la mejor de ellas toca el umbral
   con un intervalo de confianza que incluye el 0.
2. **Razon explicita para pensar que mas escala podria potenciarla**: **NO**. La
   deficiencia de v5 no es de cobertura (el corpus web llega a 0.8413) sino de
   aportacion, y mas cobertura no ha movido la aportacion en ninguna de las cinco
   configuraciones medidas.
3. **El beneficio esperado esta respaldado por la curva previa**: **NO**. De 5M a 50M
   (diez veces mas tokens) el delta va de -0.0015 a +0.0031: no hay tendencia.
4. **Queda presupuesto**: **SI**, queda la unidad de reserva (0 de 1 usada).
5. **La decision se registra antes de entrenar**: se cumpliria, pero no hay nada que
   decidir porque las condiciones 1, 2 y 3 fallan.

## 3. Conclusion

**No se escala el corpus.** Tres de las cinco condiciones fallan y las que fallan son
las que tienen que ver con la evidencia, no con el coste. El techo autorizado de 500M
queda sin usar, y la decision no escalate se sostiene en evidencia y no en
presupuesto. El rango de deltas medido (-0.0015 a +0.0046) es de
un orden de magnitud pequeno y con el intervalo de confianza tocando el 0 en todos
los casos.

La unica variable que sigue abierta, y que no es de esta fase, es el **tamano de los
datos etiquetados**: 1.483 ejemplos. Con mas etiquetados, tanto el bloque disperso de
V03 como el denso podrian aportar mas de lo que aportan hoy.
