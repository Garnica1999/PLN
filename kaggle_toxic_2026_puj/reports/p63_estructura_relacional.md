# P63 - La estructura de repeticion del registro, como rasgo y no como particion

Generado por `experiments/p63_04_estructura_relacional.py`. Fecha 2026-09-29T06:21:35+00:00.

## Que es realmente nuevo aqui

El proyecto tiene 1415 grupos canonicos de casi duplicados y los ha usado mucho, pero
siempre para **repartir** los datos. `tamano_grupo` aparece doce veces en el codigo y
las doce estan donde se construye la particion. Nunca ha sido entrada de un modelo.

Usar el grafo como predictor tampoco es nuevo: P59-10 lo uso como regularizador y dio
+0,0000988776, unas treinta veces menor que la dispersion entre pliegues. La pregunta
de aqui es otra y conviene enunciarla con precision:

> puede la estructura de repeticion de un registro modificar la **confiabilidad** de
> una prediccion, incluso en las celdas donde los vecinos no son buenos
> clasificadores?

No es propagacion de etiquetas y no es V24. Nada aqui predice una etiqueta a partir de
la etiqueta de un vecino, y nada se ajusta dentro del pliegue, que es justamente lo que
hace que la seguridad contra fugas sea toda la dificultad del experimento.

## El problema de fuga, enunciado antes de escribir los rasgos

Este es el punto de la fase donde una implementacion descuidada produce un numero
excelente que no significa nada, asi que el protocolo se fija antes:

```text
alcance ENTRENAMIENTO  el conteo usa SOLO las filas del pliegue de entrenamiento.
                        Es el unico alcance que puede entrar como feature
alcance COMPLETO        el conteo usa todas las filas, validacion incluida. Es FUGA
                        y se reporta como diagnostico, nunca se concatena a V03
```

Contar el grupo sobre todo el dataset es fuga por dos motivos a la vez: una fila de
validacion aprenderia de filas que el modelo no puede ver, y el propio conteo cambia
segun quien este en el pliegue de validacion. Una feature que no es funcion del
conjunto de entrenamiento no es una feature, es una mirada.

Reportar los dos no es un truco para conseguir un numero mayor. Es la manera de dejar
a la vista el tamano de la fuga, porque si la version con todo el dataset fuera mucho
mejor, esa es exactamente la informacion que un revisor tendria que Wrestling para
excluir, y debe estar en la pagina y no descubrirse despues.

## El bloque

Seis rasgos, sin usar ninguna etiqueta: tamano_de_grupo, log_tamano, es_unico, es_par, es_grupo_grande, n_en_universo.

**Un rasgo que se midio y se descarto:** la contradiccion de etiqueta dentro del
grupo. Un grupo con HS contradictoria no puede acertarla todas, luego es un techo y no
una feature, y por eso no entra.

| etiqueta | grupos con la etiqueta contradictoria dentro |
|---|---:|
| HS | 11 |
| TR | 8 |
| AG | 12 |

## Resultados

| brazo | corrected_cv | delta vs V03 | rescate | dano | neto | alcance | permisible |
|---|---:|---:|---:|---:|---:|---|---|
| V03_rel | 0.6727442905 | -0.1044671620 | 285 | 1124 | -839 | ENTRENAMIENTO | si |
| V03_random | 0.7558082301 | -0.0214032224 | 98 | 185 | -87 | ENTRENAMIENTO | si |
| rel_solo | 0.5205452541 | -0.2566661984 | 371 | 2165 | -1794 | ENTRENAMIENTO | si |
| rel_whole | 0.3999237487 | -0.3772877038 | 373 | 1818 | -1445 | COMPLETO, ES FUGA | NO |
| random_solo | 0.4721428240 | -0.3050686285 | 364 | 1996 | -1632 | ENTRENAMIENTO | si |

## Las tres preguntas que importan

```text
1. el bloque relacional MEJORA a V03: -0.1044671620, 0/5 pliegues, gate False
2. frente al control de capacidad de igual ancho: brecha -0.0830639396
3. cuanto compra la FUGA, midiendo el grupo con validacion incluida: -0.1206215055
```

Si 2 es pequena, cualquier ganancia de 1 era capacidad y no estructura. Si 3 es
grande, el numero sin alcance es exactamente la trampa que este experimento existe
para dejar a la vista.

## El diagnostico que explica las cifras

**El bloque esta casi vacio.** 1378 de 1483 filas,
el 92.92 por ciento, estan en un grupo singleton, y
el bloque entero tiene 7 perfiles distintos
sobre 6 columnas. Solo 105 filas tienen un perfil
relacional que no sea el de singleton.

Eso explica las tres cifras de la tabla, y es un resultado sobre el CORPUS y no sobre
el modelo: la estructura de casi duplicados del agrupador canonico es practicamente
inexistente en la particion de desarrollo. El tamano maximo de grupo es
7.

Y explica tambien por que anadir el bloque da PEOR que anadir ruido: un bloque casi
constante no aporta variacion, pero si desplaza la norma de las filas, que es el
confound de escala que P62 ya habia medido en otro contexto.

## Veredicto

```text
El bloque relacional vale -0.1044671620 frente a V03 con 0/5 pliegues, y el control de capacidad de igual ancho vale -0.0214032224, luego la brecha especifica es -0.0830639396. La causa NO es que la estructura relacional sea inutil: es que esta CASI VACIA. 1378 de 1483 filas, el 92.92 por ciento, estan en un grupo singleton, y el bloque entero tiene 7 perfiles distintos sobre 6 columnas. En solitario el bloque SI supera al aleatorio, +0.0484024302, o sea hay una senal debil en las 105 filas no singleton, pero son pocas para mover el agregado. Medir el mismo rasgo con validacion incluida, que es fuga, daria -0.1206215055
```

```text
CLOSED. La estructura de repeticion no aporta sobre V03
```

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p63_04_estructura_relacional.py`. Sin push.*