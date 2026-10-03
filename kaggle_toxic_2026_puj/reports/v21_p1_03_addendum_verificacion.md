# Addendum de verificacion al informe P1-03

Fecha: 2026-09-27. Autor: orquestador (V21). Motivo: verificacion independiente del claim
principal del informe, con el control de cordura del champion.

**Este addendum corrige un numero del informe y confirma otro. Prevalece sobre
`reports/v21_p1_03_nbsvm_y_hibridos.md` en lo que toca.**

---

## 1. Lo que se confirma

**El control reproduce el champion de forma exacta.** Con el protocolo canonico de
splits (`src/splits.py`, `StratifiedGroupKFold(5, shuffle=True, random_state=42)` sobre
grupos de near-duplicado, desarrollo de 1.483 filas):

```
corrected_cv L2 C=2.0 = 0.7772114525165721
esperado                = 0.7772114525165721   REPRODUCE EXACTO
f1 por pliegue          = [0.796010, 0.794034, 0.778860, 0.738598, 0.779056]
OOF congelado           = [0.796010, 0.794034, 0.778860, 0.738598, 0.779056]   identico
```

Ademas se verifico que la asignacion de pliegues del OOF congelado coincide fila a fila
con la de los pliegues reconstruidos, de modo que la comparacion por pliegue es legitima.

**La brecha train-OOF del champion es real y vale 0.198.** Medida como la media de las
tres F1 de la clase positiva sobre el conjunto de ajuste (umbral 0.5) contra el OOF con
umbrales congelados: `f1_train = 0.9752`, brecha `= +0.1980`. Confirma la magnitud que
el informe citaba como 0.1818 y sostiene su mecanismo.

**La superficie en C se reproduce** (el informe la daba como monotonica, y lo es):

| C | corrected_cv | delta | f1_train | brecha train-OOF |
|---|---:|---:|---:|---:|
| 0.5 | 0.726651 | -0.050560 | 0.6798 | +0.2670 |
| 1.0 | 0.755465 | -0.021746 | 0.7633 | +0.1297 |
| 2.0 | 0.781985 | +0.004774 | 0.8485 | +0.0665 |
| 4.0 | 0.783642 | +0.006431 | 0.9458 | +0.1622 |

El mecanismo queda confirmado con numero: **L1 recorta la brecha train-OOF de 0.1980 a
0.0665** en C=2.0. No es "mas regularizacion" en general: L1 en C bajo es
catastrofico, y la mejora aparece solo en la region de C alta.

---

## 2. Lo que se corrige: el 5/5 de pliegues no se reproduce

El informe declara para L1 con C=4.0 una ventaja de **5 de 5 pliegues**, y llama a eso
"la senal mas consistente medida". La verificacion independiente da **3 de 5**.

Comparacion por pliegue, L1 C=4.0 contra el OOF congelado del champion:

| pliegue | champion | L1 C=4.0 | diferencia |
|---|---:|---:|---:|
| 1 | 0.796010 | 0.7929 | **-0.0032** |
| 2 | 0.794034 | 0.7781 | **-0.0159** |
| 3 | 0.778860 | 0.7887 | +0.0098 |
| 4 | 0.738598 | 0.7532 | +0.0146 |
| 5 | 0.779056 | 0.8056 | +0.0266 |

Se ganan 3 y se pierden 2, y la perdida del pliegue 2 es de 0.0159, que es la mayor
perdida individual de la tabla. El valor global tambien difiere, aunque poco: el informe
reporta 0.783423 y la reproduccion da **0.783642** (+0.006431 en vez de +0.006211). La
diferencia de 0.00022 es atribuible a la version de scikit-learn, que ya emite
`FutureWarning` de deprecacion de `penalty`; la diferencia de conteo de pliegues, no.

Como control, **L1 C=2.0 si reproduce exactamente** lo que el informe reportaba para
ese punto: 0.781985 y 3/5. La discrepancia esta aislada en C=4.0.

**Consecuencia para la lectura de la candidata**: sigue siendo la senal local mas fuerte
de la linea clasica y con superficie suave, pero pierde el atributo de unanimidad que
la hacia destacada. Pasa de "la senal mas consistente medida" a "la senal con la mejor
superficie en C", que es un merito real pero menor.

---

## 3. Lo que el informe no midio y es lo que decide si hay submission

El informe reporta la metrica y el IC95, pero **no el recuento de celdas**. Ese recuento
es la cifra que, en este proyecto, ha predicho la transferencia mejor que el delta de
metrica. Se mide aqui, sobre las mismas 1.483 filas de desarrollo, en el mismo orden que
el OOF congelado, con los umbrales congelados de V03:

| candidata | celdas cambiadas | rescates | danos | **neto** | delta F1 |
|---|---:|---:|---:|---:|---:|
| L1 C=2.0 | 468 | 225 | 243 | **-18** | +0.004774 |
| L1 C=4.0 | 413 | 208 | 205 | **+3** | +0.006431 |

Desglose por etiqueta de L1 C=4.0:

| etiqueta | rescates | danos | neto | tasa positiva |
|---|---:|---:|---:|---|
| HS | 84 | 80 | +4 | 0.4572 -> 0.4626 |
| TR | 50 | 53 | -3 | 0.3001 -> 0.3358 |
| AG | 74 | 72 | +2 | 0.3776 -> 0.3965 |

**Esta es la cifra que cambia la lectura de la candidata.** Un neto de **+3 sobre 413
celdas** es un saldo casi nulo. Con la escala medida del proyecto (unos 107 cambios netos
de celda equivalen a 0.003 de leaderboard), un neto de +3 implica un efecto de cerca de
0.0001. Es decir: **la mejora de +0.0064 no viene de acertar mas celdas, sino de
redistribuir la ganancia entre las que cambia**. Coincide con lo medido en V19, donde la
F1 resulto no lineal y aparecieron 12 valores distintos de ganancia por celda.

Consecuencia directa: L1 C=4.0 tiene el mejor `deltaOOF` de la historia reciente de la
linea clasica y a la vez el cambio mas equilibrado en numero de celdas, lo que la hace la
candidata mas enviable en apariencia. Pero su saldo de celdas es microscopico, que es
exactamente el perfil que en V16 dio cinco de cinco a la baja y en V20 dio -0.00643 con
un neto de -6.

**Un `deltaOOF` de +0.0064 no debe leerse como una mejora solida solo porque sea el mayor
que se ha medido. El saldo de celdas es la cifra que decide.**

---

## 4. Nota sobre advertencias del entorno

scikit-learn en este venv emite `FutureWarning` de deprecacion de `penalty` y
`UserWarning` de `l1_ratio` incoherente al pedir `penalty="l1"` con el defecto
`l1_ratio=0.0`. La ejecucion es correcta (los resultados son coherentes y el control
reproduce), pero cualquier reimplementacion debe usar `l1_ratio=1` en lugar de
`penalty="l1"` para evitar la deprecacion, y registrar cual de las dos formas uso.

## 5. Lo que queda abierto

- El **elastic net** no se midio. El informe razona que no anade nada, y la medicion de
  L1 y L2 no lo contradice, pero es una conjetura y no una medida.
- **No se sabe si C por encima de 4.0 mejora o rompe.** La curva es creciente con
  incrementos decrecientes, asi que el optimo podria estar mas alla. Explorar mas alla
  de 4.0 sin criterio pre-registrado seria buscar a posteriori, y no se ha hecho.
- El **IC95 pareado** de la tabla del informe sigue conteniendo el cero en las tres
  candidatas, asi que segun el gate de V5 ninguna es adoptable. Segun el criterio de V21
  (leaderboard) la decision se toma por LB, y no se ha enviado nada.
