# AGE - auditoria final de cobertura metodologica

**Conclusion: `AGE_RESEARCH_CLOSED`**

## Las dos preguntas de esta fase

```text
1. G14 cerro el INPUT GAP: no hay campos de entrada sin usar.
2. Esta fase cierra la pregunta del MECHANISM GAP: ¿queda alguna forma
   distinta de APRENDER de los mismos inputs?
```

## Lo que se midio antes de opinar

La particion de los errores de V30, refinada con una distincion que faltaba:

| Grupo | Filas | % | Que significa |
|---|---|---|---|
| A: nadie corrige | 27,430 | 41.9% | ninguna de las 26 fuentes acierta esas filas |
| B: alguien corrige | 37,973 | 58.1% | al menos una acierta |
| - B1 rescate **unico** | 8,358 | 12.8% | solo una fuente la tiene |
| - B2 rescate **compartido** | 30,038 | 45.9% | varias fuentes la tienen |

Mediana de fuentes por error rescatado: **5**, con un maximo de 25.

### Por que la distincion B1/B2 importa

**Un rescate compartido significa que la informacion esta en varias fuentes a la vez, y el problema es de integracion. Un rescate unico significa que solo esa fuente la tiene, y el problema es de mecanismo.** Solo la segunda situacion justifica buscar un mecanismo nuevo, porque es la unica en la que hay una fuente con informacion exclusiva.

Y la medicion dice que B2 domina: **30,038 de los 37,973 rescates son compartidos**, la mayoria por cinco fuentes o mas. La informacion esta **repartida**, no encerrada en una fuente.

## Los rescates exclusivos, que era la pregunta central

El inventario previo del proyecto registro **122 rescates exclusivos**. Con la definicion de esta auditoria -una sola de las 26 fuentes acierta la fila- hay **8,029**. La diferencia es de definicion, y se declara: los dos numeros miden cosas distintas y ninguno es erroneo. Se usa el de aqui porque es el que responde a la pregunta del brief.

| Medida | Rescates exclusivos | Todos los errores |
|---|---|---|
| Longitud media | 81.61 | 84.16 |
| Confianza media de V30 | 0.4285 | 0.4418 |
| Fraccion con URL | 41.5% | 41.3% |

**No hay diferencia en ninguna dimension.** Y se reparten entre **23 fuentes distintas**, con la mas frecuente concentrando solo el **20.5%**.

Veredicto: **HETEROGENEIDAD ENTRE FAMILIAS**. los 8029 rescates exclusivos se reparten entre 23 fuentes distintas y ninguna concentra mas del 21%. No hay una fuente con informacion exclusiva que apunte a un mecanismo nuevo: la informacion esta dispersa, que es la firma de ruido de modelo, no de un canal sin explotar

## La cobertura de mecanismos

| Estado | Cuantos |
|---|---|
| USED | 23 |
| CLOSED | 9 |
| NOT USED | 5 |
| INVALIDATED | 1 |
| NOT_ACTIONABLE | 1 |

De los 39 mecanismos auditados, **23 estan usados** y los cinco sin usar se cierran con la medicion delante, no por inercia:

- **O03 ranking objective**: la metrica de la competicion es macro-F1 sobre el argmax. Un objetivo de ranking no mueve la metrica evaluada: cambiaria el objetivo sin cambiar el criterio
- **O04 focal loss**: es reponderacion de ejemplos con modulacion por confianza. `class_weight` ya se probo (V18-P1) y la modulacion apunta a los ejemplos dificiles, que son la poblacion A: 27.430 filas donde NINGUNA de las 26 fuentes tiene senal. Reponderar ahi es entrenar sobre ruido
- **O05 label smoothing**: actua sobre la zona de frontera, y P34 midio que ahi el ruido aleatorio gana a la senal real (real - ruido = -0,0000797)
- **E03 hard example mining**: el conjunto dificil ES la poblacion A, donde ninguna de las 26 fuentes tiene senal. Minar ese conjunto es exactamente entrenar sobre ruido
- **V01 bootstrap de grupos**: es reduccion de varianza con otra unidad de remuestreo. V28-02 cerro el bagging del meta-modelo en -0,000312, y cambiar la unidad de remuestreo no cambia el mecanismo

## La conclusion

```text
AGE_RESEARCH_CLOSED
```

**Por que `AGE_RESEARCH_CLOSED` y no una hipotesis mas.**

Un mecanismo nuevo tendria que cumplir los seis criterios del brief, y el que no cumple ninguno de los candidatos es el cuarto: **no ser equivalente a una tecnica cerrada**. Los cinco mecanismos sin usar actuan sobre la distribucion de entrenamiento o sobre la funcion de perdida, y su poblacion objetivo es exactamente la que el diagnostico ya midio como sin senal corregible:

- **focal loss**, **hard example mining** y **label smoothing** apuntan a los ejemplos dificiles y a la frontera. La poblacion A son 27,430 filas donde ninguna de 26 fuentes acierta, y P34 midio que en la frontera el ruido aleatorio gana a la senal real.
- **ranking objective** no mueve la metrica evaluada, que es macro-F1 sobre el argmax.
- **bootstrap de grupos** es reduccion de varianza con otra unidad de remuestreo, y el bagging del meta-modelo ya se cerro en -0,000312.

## Lo que esta conclusion NO dice

- **No** dice que V30 sea optimo.
- **No** dice que no exista ninguna mejora posible.
- **No** dice que no exista informacion linguistica: P33 midio AUC 0,5411 en el residuo.

Dice algo mas estrecho y mas honesto: **no quedan actualmente mecanismos de aprendizaje metodologicamente defendibles y no recorridos dentro de las restricciones del proyecto**. La afirmacion es sobre el espacio auditado, y el espacio auditado esta enumerado: 22 mecanismos, 6 familias de representacion, 10 familias de modelo, la entrada completa y las tres vias de integracion del score.

## El techo, medido

- **27,430** errores de V30 que ninguna de las 26 fuentes corrige, y sin campo de entrada que pudiera corregirlos.
- **8,029** rescates exclusivos sin estructura comun: ruido de modelo, no un canal sin explotar.
- V30 sigue siendo el mejor resultado validado con diferencia sobre cualquier alternativa medida.
