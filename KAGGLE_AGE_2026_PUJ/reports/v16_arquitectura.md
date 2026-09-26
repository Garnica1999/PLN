# V16 — Arquitectura del meta-modelo

## 1. Qué se cambia

V12 combina las dos cabezas con un promedio fijo 0,5 + 0,5. Es una decisión de diseño de la primera versión y **nunca se optimizó**. V13, V14 y V15 apuntan a que el cuello de botella es arquitectónico, así que aquí se mide la palanca: mismo protocolo, mismas 279 variables, tres formas de combinar.

- **A0**: promedio fijo 0,5 + 0,5 (producción, y su OOF ya estaba calculado).
- **A1, peso aprendido**: sigue siendo un promedio, pero el peso se elige por fold sobre una partición **interior** del train exterior (20 %, agrupada por duplicados), de modo que no se mira la validación.
- **A2, combinador logístico**: las doce probabilidades de las dos cabezas alimentan una LogReg, es decir apilamiento en dos niveles. Es la que puede dar peso distinto a cada clase y a cada cabeza, y la que de verdad cambia la arquitectura.

Ninguna introduce información nueva. Si alguna mejora, el hallazgo es que el modelo llevaba desde el principio una decisión de combinación no medida, que es distinto de "no hay más señal".

## 2. Resultado en la partición de 5 folds

| Arquitectura | macro-F1 | Delta | Folds positivos |
|---|---|---|---|
| A0 promedio fijo (producción) | 0.6005 | - | - |
| peso_aprendido | 0.6002 | -0.0003 | 0/5 |
| combinador_logistico | 0.6011 | +0.0006 | 3/5 |

Pesos aprendidos por fold en A1: fold 0 = 0.1, fold 1 = 0.5, fold 2 = 0.5, fold 3 = 0.4, fold 4 = 0.5

## 3. Segunda partición (4 folds, arquitectura elegida en la de 5)

La arquitectura que se confirma es **combinador_logistico**.

| Fold | A0 | Elegida | Diferencia |
|---|---|---|---|
| 1 | 0.6013 | 0.6009 | -0.0003 |
| 2 | 0.5998 | 0.6015 | +0.0017 |
| 3 | 0.6039 | 0.6043 | +0.0005 |
| 4 | 0.6048 | 0.6056 | +0.0008 |

**Positivos: 3/4**

## 4. Doble métrica (§36) y complementariedad

| Frontera | A0 | Elegida |
|---|---|---|
| 18-24\|25-34 | 0.4963 | 0.5065 |
| 25-34\|35-49 | 0.4974 | 0.5131 |

Complementariedad global: rescate 2,692 · daño 2,987 · **neto -295**

## 5. Gate D-042

| Criterio | Valor | Cumple |
|---|---|---|
| efecto minimo | delta +0.0006 >= 0.001 | **no** |
| folds positivos | 3/5 >= 3 | si |
| significacion | IC95 [-0.0003, +0.0015] | **no** |
| segunda particion | 3/4 >= 3 | si |
| fronteras | 2/2 mejoran | si |

**Decision: NO ADOPTAR**

## 6. Lectura

**A1, el peso aprendido, no aporta nada, y eso es un resultado en si mismo.** Los pesos que elige
por fold son 0,1 · 0,5 · 0,5 · 0,4 · 0,5: el 0,5 de produccion se elige solo en tres de los cinco
folds, y la macro-F1 global sale **-0,0003** con 0 de 5 folds positivos. El promedio fijo 0,5 + 0,5
no era una decision desafortunada: era, sencillamente, la decision correcta. Eso descarta de golpe
la version mas barata de la hipotesis arquitectonica.

**A2, el combinador logistico, es lo mas cerca que ha estado este proyecto de mejorar a V12, y aun
asi no se adopta.** El signo es el correcto por primera vez en mucho tiempo: **+0,0006** con
**3 de 5 folds** positivos, y **3 de 4** en la segunda particion. Pero el IC95 del bootstrap pareado
es **[-0,0003; +0,0015]**, que **contiene el cero**, y el efecto queda por debajo del minimo de
+0,001. Falla dos de los cinco criterios y el veredicto es NO ADOPTAR.

**Lo que si queda, y es lo importante de esta fase: las fronteras centrales mejoran de forma
notable.** La `18-24|25-34` pasa de 0,4963 a **0,5065** y la `25-34|35-49` de 0,4974 a **0,5131**.
Son ganancias de **+1,0 y +1,6 puntos**, del orden de tres a cinco veces el efecto global, y en el
mismo sentido. La macro-F1 global apenas se mueve porque la media ponderada por clases la dominan
las clases FACILES, y ahi el combinador no tiene nada que ganar.

Esto **no es una contradiccion**, es la misma conclusion de siempre vista desde el otro lado. El
problema de este conjunto de datos esta en las fronteras de edad, y el meta-modelo tiene algo que aportar
justo ahi. Lo que no tiene es magnitud suficiente para que la metrica global la registre por encima
del ruido. En otras palabras: **la arquitectura no estaba desperdiciando una oportunidad grande de senal, sino algo marginal**. El meta-modelo funciona; lo que no hace es ganar metrica global.

- El gate D-042 **funciona** y ha vuelto a hacer su trabajo: un candidato con el signo correcto en
  cuatro medidas independientes y con mejora visible en las fronteras duras sigue siendo
  rechazado, porque el intervalo de confianza no excluye el cero. Es exactamente el comportamiento
  que se le pidio, y ahora esta respaldado por un caso en el que la **intuicion** (que las fronteras
  mejoran, luego mejora el modelo) y el **gate** discrepan.
- La combinacion de las dos familias de evidencia (global y frontera) sugiere que un trabajo
  futuro deberia medir **sobre las fronteras** desde el principio, no como metrica secundaria. Con
  el desglose actual, un +0,0157 en la frontera 25-34|35-49 es invisible en la global.
- V16 **no cierra por completo** el problema arquitectonico: cierra el combinador de dos niveles
  sobre las dos cabezas. Quedan sin medir la perdida de la propia combinacion (un objetivo distinto,
  no un promedio), y arquitecturas de mas de dos niveles.

> Nota de edicion (2026-09-26): esta seccion 6 se anadio despues de generar el informe, porque el
> generador no la incluia. **No se modifico ninguna cifra** y `scripts/v16_arquitectura.py` no se ha
> editado ni vuelto a ejecutar. Las cuatro filas de la tabla de la seccion 4 se repararon con
> `scripts/v15_v16_tablas.py`, que escapa el guion vertical de las etiquetas de frontera sin tocar
> ningun numero.

