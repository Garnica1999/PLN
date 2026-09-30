# Decision de P73

**P73 = COMPLEMENTARY**

delta A - B +0.0316259472 | bootstrap de clusters [+0.010066, +0.053484] | gana A en 4/5 pliegues | veredicto interno A_SUPERA_A_B_CLARO

Hay evidencia de que la geometria aprendida aporta informacion que una representacion aleatoria no ofrece al lector secuencial. Antes de abrir fine-tuning: bootstrap, complementariedad, segunda particion agrupada y revisiones de compliance y de DL.

```text
V03 = FROZEN
AGE = sin cambios
Word2Vec Twitter 100d = congelado
P72 = CLOSED
P73 = COMPLEMENTARY
test = 0
holdout = 0
leaderboard = 0
submission = 0
```

---

## ADENDA 2: la complementariedad es un diagnostico en celdas, no una mejora

Los recuentos de complementariedad (rescue 2096, damage 602, net 1494) estan en
unidades **tweet-etiqueta** sobre 4449 celdas y son un diagnostico. **No** son
comparables con `corrected_cv`, **no** son una ganancia de macro-F1 y **no**
constituyen una mejora sobre el campeon. La TextCNN queda 0.168 por debajo de V03
y mientras eso siga asi no se evalua ninguna fusion.

## ADENDA: la robustez de particion es PARCIAL

Anadida despues de ejecutar P73-03. **No modifica el texto anterior ni la regla pre-registrada**,
que se aplico correctamente a la particion congelada. Anade la evidencia posterior y califica.

```text
particion                    delta         pliegues A      IC 95 por clusters            excluye 0
la congelada                +0.0316259472      4 de 5      [+0.010066, +0.053484]          SI
alternativa 20261103        +0.0344649141      4 de 5      [+0.014094, +0.055680]          SI
alternativa 20261104        +0.0112854167      4 de 5      [-0.009276, +0.031672]          NO
```

Estable en las tres: el signo y la direccion (A gana en 4 de 5 pliegues en cada una).
Inestable: la magnitud, que va de +0.011 a +0.034, y la significacion, que se pierde en una de
las tres.

**Afirmacion defendible**: la direccion del efecto es estable en las tres particiones agrupadas
probadas, pero su magnitud no lo es. La evidencia es direccionalmente consistente y
cuantitativamente inestable.

`P73 = COMPLEMENTARY` se mantiene CON esa calificacion explicita, no como efecto establecido.

**Amenaza principal sin medir**: el control aleatorio es UNA sola tirada, y el mandato de la fase
prohibe otra semilla.

**CORRECCION (revision de cierre del Kaggle Specialist).** El texto original decia 0.5731 a
0.6183 con un rango de 0.045. **Era un error**: mezclaba el minimo de una corrida con un maximo
que no es de ninguna. Las cifras correctas:

```text
P72-03, sin estandarizar    de 0.577587 a 0.617904   rango 0.040318
P72-04, pipeline de P1      de 0.573062 a 0.604959   rango 0.031897
```

La lectura es mas debil de lo que decia el original: el rango es del ORDEN del delta medido
aqui, no mayor que el, y es el rango de OTRO lector. Es un argumento de orden de magnitud, no
una prueba.

**Y una precision que acota la robustez de particion**: las tres particiones reutilizan EL MISMO
par de matrices, luego NO replican el sorteo aleatorio y NO cubren la amenaza dominante. Esa
amenaza es comun a las tres mediciones.

Lo que si sostiene la conclusion: la varianza del sorteo bajo el lector de P73 nunca se midio,
y eso basta para que la amenaza siga abierta. No se puede distinguir "la geometria aporta" de
"esa matriz aleatoria concreta salio floja". Es lo primero que habria que medir antes de
invertir mas en esta linea.
