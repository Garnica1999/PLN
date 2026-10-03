# V10-06: tres parametrizaciones de la capa de decision (P1, tercera linea de la via A)

Generado por `experiments/v10_06_umbrales_estables.py` (V10-06 (P1, umbrales estables)).

## 1. La pregunta

Los umbrales de V03, 0.44, 0.51 y 0.46, son tres numeros libres elegidos sobre un unico
OOF. La fase v6 atribuyo a eso un inflado de +0.0036, pero el informe de la fase v7
demostro que la cifra confunde el coste del protocolo con el efecto del umbral: el
control de tres niveles con umbrales congelados da -0.0081 solo por ajustar sobre cinco
sextos de las filas, y al comparar contra un brazo que comparte conjunto de ajuste el
efecto aislado del umbral queda en -0.0019. La pregunta que queda es si los tres numeros
son un punto real de la superficie o un sorteo afortunado. Si son reales, una
parametrizacion **mas restringida** de la misma capa deberia rendir parecido con menos
variacion entre particiones.

| parametrizacion | grados de libertad |
|---|---|
| tres_libres | 3, uno por etiqueta |
| desplazamiento_y_escala | 2, un desplazamiento y una escala compartidos |
| encogimiento_a_la_mediana | 1, el factor de encogimiento hacia la mediana |

## 2. Ganancia y dispersion

| parametrizacion | media | peor | recorrido | dispersion | delta medio | mejoran | celdas |
|---|---|---|---|---|---|---|---|
| tres_libres | 0.776041 | 0.766534 | 0.013072 | 0.005408 | +0.000131 | 3/5 | 124 |
| desplazamiento_y_escala | 0.775946 | 0.766353 | 0.012964 | 0.005420 | +0.000037 | 3/5 | 126 |
| encogimiento_a_la_mediana | 0.775788 | 0.766534 | 0.012177 | 0.005226 | -0.000122 | 3/5 | 124 |

La referencia de dispersion es
0.005408, la de tres umbrales libres.

## 3. Un segundo resultado que sale del mismo experimento

La fila de tres umbrales libres es una **version castigada de V03**: los mismos tres
grados de libertad, pero re-estimados dentro de cada pliegue de entrenamiento en vez de
elegidos una vez sobre todo el OOF. La diferencia entre V03 y esa fila es por tanto una
**medicion directa** de cuanto aporto optimizar los umbrales sobre el conjunto que
despues se reporta, y no una hipotesis de la auditoria.

El delta medio de esa fila es +0.000131 y sus celdas
distintas frente a V03 son 124 de
media.

**Correccion de una cita previa, que importa.** La fase v6 atribuyo a la reoptimizacion
de umbrales un inflado de +0.0036. Este experimento mide +0.000131, y el informe de la
fase v7 ya habia mostrado que la cifra de v6 confundia el coste del protocolo con el
efecto del umbral, con un efecto aislado de **-0.0019**. Las tres cifras son distintas
porque miden cosas distintas: +0.0036 es un error de comparacion entre brazos con
distinto conjunto de ajuste, -0.0019 es el efecto del umbral en el protocolo estricto de
tres niveles, y +0.000131 es el efecto de reestimar los umbrales por pliegue en el
protocolo de dos niveles. **Las tres coinciden en lo substancial: el efecto del umbral es
pequeno.** Lo que no cabe es citar +0.0036 como si fuera el efecto del umbral, y esa
correccion se aplica tambien a la documentacion de V10-04.

## 4. Gates pre-registrados

- **Gate 1**, igual o mejor media con menor dispersion: **False**
  - candidatas estables: ninguna
- **Gate 2**, los tres criterios del addendum: **False**
  - parametrizacion evaluada: tres_libres
  - delta medio de al menos 0.002: False
  - al menos 4 de 5 particiones: False
  - intervalo que no incluye el cero: False

## 5. Veredicto

**GATE 1 NO SUPERADO.** Ninguna parametrizacion mas restringida mantiene la ganancia
de la de tres umbrales libres y ademas es mas estable. La holgura de la capa de
decision es **irreducible con estos datos**, que es el criterio de cierre escrito
antes de ejecutar. La consecuencia practica es que **no hay una version de la capa de
decision que sea a la vez igual de buena y mas robusta**, de modo que los umbrales
congelados se quedan como estan y no se sustituyen por una version mas elegante.

El gate 2 no se supera, asi que **ninguna de las tres se envia**: reparametrizar no
supera a V03 en la metrica, que es lo que un envio necesita.

Fecha: 2026-09-26T07:58:18+00:00. Duracion: 11.3 s.
