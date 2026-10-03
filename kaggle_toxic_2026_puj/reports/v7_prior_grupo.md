# Track P8: prior de etiqueta del grupo (V7-02)

Generado por `experiments/v7_02_prior_grupo.py` (V7-02 (P8)).

## 1. Que se prueba y por que no es el track A de la fase v6

La distincion es mecanica y no nominal. La fase v6 **repondero** filas por tamano de
grupo, que resta capacidad a un ejemplo repetido, y empeoro de forma monotona. Aqui el
grupo se convierte en una **caracteristica** que aporta la distribucion de etiquetas
de las otras observaciones del mismo grupo: informacion que el modelo lineal no puede
obtener del texto porque requiere etiquetas de vecinos.

## 2. El techo, medido antes de entrenar

| magnitud | valor |
|---|---|
| filas del desarrollo | 1483 |
| grupos distintos | 1415 |
| tamaño maximo de grupo | 7 |
| **filas con grupo repetido** | **105 (7.08 %)** |
| evidencia de etiqueta uniforme | 80 |
| evidencia de etiqueta contradictoria | 31 |

**Este es el dato que condiciona el track**: solo el 7.08 % de las filas tiene "otra observacion del
mismo grupo". Para el resto, el prior no esta definido. Es el mismo techo que
condiciono el track A, y por eso se mide antes de gastar una unidad de entrenamiento.

Y hay dos datos mas que acotan la linea antes de mirar ninguna metrica.

**Uno: la evidencia es casi siempre unanime.** De los pares de grupo y etiqueta,
80 tienen la misma etiqueta en todas sus filas y solo
31 son contradictorios. Un prior casi
deterministico **no aporta informacion que el texto no tenga ya**: dos filas del mismo
grupo son casi el mismo texto, y por tanto el modelo lineal ya les da la misma
representacion y ya las predice igual. El prior solo confirma lo que el texto ya decia.

**Dos: la evidencia de grupo no se concentra donde V03 falla.** V03 acierta el
22.9 % de las filas que **si** tienen evidencia
de grupo y el 22.9 % de las que no la tienen, es decir
**lo mismo**. Una caracteristica que informa sobre un subconjunto en el que el modelo
no falla mas que en el resto no tiene donde trabajar. Esta version del informe daba
por hecho lo contrario y los datos lo refutan.

## 3. Diseno de la caracteristica

Para evitar valores ausentes y para que la variable sea exactamente la informacion
extra, la caracteristica es la **desviacion de la prior del grupo respecto a la prior
global** del pliegue:

```text
desviacion[etiqueta] = prior_del_grupo_sin_la_propia_fila - prior_global
soporte = log(1 + numero_de_otras_filas_del_grupo)
```

- Si el grupo tiene una sola fila, la desviacion es **exactamente cero**: neutral, sin
  imputacion y sin columna muerta.
- Para las filas de entrenamiento el prior se calcula **dejando fuera la propia fila**,
  que es lo que impide que una fila se use a si misma como evidencia.
- Para las filas de validacion se usan las filas de su grupo que esten en el
  entrenamiento, que por construccion no incluyen su etiqueta.

## 4. Resultados

Control: **0.777211** frente a los 0.777211 de V03, hueco de
+0.000000 con tolerancia declarada de 0.000500.

| variante | columnas | corrected_cv | delta | IC95 | HS | TR | AG | rescate | dano |
|---|---|---|---|---|---|---|---|---|---|
| P8-0-control | 0 | 0.777211 | +0.0000 | [+0.0000, +0.0000] | 0.7617 | 0.8157 | 0.7542 | 0 | 0 |
| P8-1-desviacion | 3 | 0.776145 | -0.0011 | [-0.0045, +0.0021] | 0.7592 | 0.8194 | 0.7498 | 1 | 2 |
| P8-2-desviacion-soporte | 4 | 0.775530 | -0.0017 | [-0.0051, +0.0016] | 0.7592 | 0.8176 | 0.7498 | 1 | 2 |

## 5. Gate de adopcion

| variante | delta al menos +0.003 | IC95 excluye 0 | pliegues positivos | gate |
|---|---|---|---|---|
| P8-1-desviacion | no (-0.0011) | no | 1/5 | no supera |
| P8-2-desviacion-soporte | no (-0.0017) | no | 1/5 | no supera |

## 6. Interseccion de errores

| magnitud | valor |
|---|---|
| filas_con_evidencia | 105 |
| v03_acierta_en_evidencia | 24 |
| candidato | P8-1-desviacion |
| candidato_acierta_en_evidencia | 24 |
| v03_falla_y_candidato_acierta_en_evidencia | 0 |
| salvage_total | 1 |
| damage_total | 2 |

## 7. Conclusion

El techo medido antes de entrenar anticipaba el resultado, y esa es la leccion mas
util de la fase: **la estructura del corpus limita la linea antes de que la metrica
la limite**. Tres hechos se encadenan:

1. Solo el 7.08 % de las filas tiene otra
   observacion del mismo grupo, luego la caracteristica no esta definida en el resto.
2. Donde si esta definida, la etiqueta del grupo es **unanime en 80 de los casos**, de modo que el prior es casi
   determinista y **redundante**: el texto casi identico ya produce una
   representacion casi identica.
3. Y el error de V03 **no se concentra** en esas filas, porque su acierto es el mismo
   con y sin evidencia de grupo.

El dato que remata el diagnostico es el de la interseccion: **la caracteristica no
arregla ni una sola fila de las que tiene evidencia**. No es que estorbe, es que es
inerte exactamente donde se aplica.

Eso no dice que la senal de repeticion sea inutil. Dice que **no es informacion
nueva para el modelo**, y esa es una distincion que la fase v6 no podia hacer: alli se
concluyo que la repeticion es senal, y es cierto, pero **ser senal y ser senal nueva
son dos cosas distintas**. Aqui se separan: la repeticion es senal que V03 ya captura,
y por tanto no queda nada que extraer de ella.

Fecha: 2026-09-26T02:13:11+00:00. Duracion: 10.2 s.
