# P61 - Reconciliacion de los problemas de instrumentacion de P60

Generado por `experiments/p61_00_reconciliacion.py`. Fecha 2026-09-28T19:01:18+00:00. Commit e5de2ce.

## Por que esto es el primer paso de P61

P60 termino con ocho errores reales en su propia instrumentacion, y casi todos se
detectaron y corrigieron dentro de la misma fase. El riesgo que queda es concreto: un
**artefacto escrito por el codigo con bug y nunca reejecutado por el codigo corregido**.
Una cifra cuyo codigo productor no se ha vuelto a ejecutar no es evidencia. La pregunta
que responde este informe no es si el bug se corrigio, sino si la cifra que sobrevive
procede de una ejecucion del codigo tal como esta ahora.

## El criterio de obsolescencia

Cada script de P60 escribe su propio artefacto JSON al final de su ejecucion. Por tanto:

```text
si el SCRIPT se modifico DESPUES de que se escribiera el ARTEFACT,
el artefacto esta obsoleto y sus numeros proceden de codigo anterior
```

Las marcas de tiempo dan una respuesta directa y falsable. Ademas, para cada bug se
comprueba que el GUARDIA que se anadio despues sigue presente en el codigo actual, lo
que distingue una correccion viva de una correccion perdida en un merge.

## Clasificacion de los nueve problemas

| id | pista | clasificacion | artefacto obsoleto | guardia presente | resultado valido |
|---|---|---|---|---|---|
| P60-B01 | A (ASL) | **CORRECTED_AND_VALID** | False | True | corrected_cv 0.624338862, delta del mecanismo -0.1529258699 |
| P60-B02 | G (contraste) | **CORRECTED_AND_VALID** | False | True | corrected_cv 0.4497522457, delta del mecanismo -0.294907274 |
| P60-B03 | L (DRO) | **CORRECTED_AND_VALID** | False | True | corrected_cv 0.777600986, delta del mecanismo 0.0003362541 |
| P60-B04 | C (deteccion) | **CORRECTED_AND_VALID** | False | True | AUC incremental de la senal estructural en AG 0.662712, ganancia 0.112898 |
| P60-B05 | H (transporte) | **CORRECTED_AND_VALID** | False | True | corrected_cv 0.4686780062, delta del mecanismo -0.072306717 |
| P60-B06 | C3 (intervencion) | **VALID** | False | True | corrected_cv 0.7772647319, delta del mecanismo 0.0 |
| P60-B07 | I (meta-reweighting) | **CORRECTED_AND_VALID** | False | True | corrected_cv 0.4761602463, delta del mecanismo -0.3011044856 |
| P60-B08 | J (PolyLoss) | **CORRECTED_AND_VALID** | False | True | corrected_cv 0.7755546945, delta del mecanismo -0.0017100374 |
| P60-B09 | cierre | **CORRECTED_AND_VALID** | False | True | delta de C 0.0 tras la correccion |

## Detalle de cada problema

### P60-B01 - A (ASL)

* **Script**: `experiments/p60_02_asl.py`
* **Artefacto**: `reports/p60_A_asl.json`
* **Bug**: dos errores en la misma perdida. Primero, en gamma mas igual a cero la derivada del termino positivo es menos g, es decir menos uno menos p, y el codigo usaba menos p, con lo que el error relativo del gradiente llegaba a 4,79e+00. Segundo, el VALOR de la perdida no multiplicaba por la constante de regularizacion mientras el GRADIENTE si, de modo que la derivada numerica era exactamente la mitad de la analitica
* **Correccion**: se derivo d/dF de menos g para el termino positivo y se aplico el factor c al valor; se verifico en dos regimenes, con todos los negativos por encima del corte y con todos por debajo, porque una diferencia central que cruza la muesca de p = m compara dos ramas distintas y da un error espurio
* **Reejecutado**: True | el STOP condition del propio script fired con el gradiente erroneo, luego no llego a escribir artefacto; las tres versiones posteriores si llegaron a escribirlo
* **Verificacion tras la correccion**: 3,22e-07 con el corte inactivo y 5,79e-06 con el activo
* **Clasificacion**: **CORRECTED_AND_VALID** - el artefacto es POSTERIOR al script actual, luego lo produjo el codigo corregido, y el guardia que se anadio esta presente
* **Resultado final valido**: corrected_cv 0.624338862, delta del mecanismo -0.1529258699
* **Decision**: CLOSED

### P60-B02 - G (contraste)

* **Script**: `experiments/p60_01_contrastive.py`
* **Artefacto**: `reports/p60_G_contrastive.json`
* **Bug**: la perdida contrastiva devolvia NaN. El log-softmax se calculaba con la diagonal enmascarada a menos infinito, y al multiplicar por una mascara que vale cero en la diagonal aparecia menos infinito por cero
* **Correccion**: la diagonal se anula DESPUES de la resta del log-particion, con torch.where, en vez de antes; verificado con un test unitario que comprueba valor finito, gradiente finito y el caso sin positivos
* **Reejecutado**: True | la primera ejecucion con NaN no llego al bloque de metricas util
* **Verificacion tras la correccion**: test unitario: valor finito, gradiente finito, sin positivos da 0
* **Clasificacion**: **CORRECTED_AND_VALID** - el artefacto es POSTERIOR al script actual, luego lo produjo el codigo corregido, y el guardia que se anadio esta presente
* **Resultado final valido**: corrected_cv 0.4497522457, delta del mecanismo -0.294907274
* **Decision**: CLOSED

### P60-B03 - L (DRO)

* **Script**: `experiments/p60_03_dro.py`
* **Artefacto**: `reports/p60_L_dro.json`
* **Bug**: el objetivo de riesgo por grupos repartia la perdida en escala O(1), porque cada riesgo de grupo es una MEDIA, frente a la escala O(n) del objetivo de V03, que suma sobre ejemplos. El brazo no media la agregacion del riesgo sino una regularizacion efectiva unas 1186 veces mayor, y daba -0,163083
* **Correccion**: se multiplico el objetivo por n, que es una CONSTANTE, de modo que la masa total coincida con la del control y el gradiente analitico siga siendo exacto sin terminos de derivada del factor; un primer intento de normalizar por la masa DEPENDIA de los parametros y rompia el gradiente, y se detecto porque la verificacion fallo con 6,85e-02
* **Reejecutado**: True | el numero -0,163083 es un resultado CORREGIDO que debe conservarse como evidencia del fallo, no como medida de DRO
* **Verificacion tras la correccion**: verificacion del gradiente 6,73e-07 con un grupo y 1,42e-06 con cuatro; razon de masas entre brazos 0,9996
* **Clasificacion**: **CORRECTED_AND_VALID** - el artefacto es POSTERIOR al script actual, luego lo produjo el codigo corregido, y el guardia que se anadio esta presente
* **Resultado final valido**: corrected_cv 0.777600986, delta del mecanismo 0.0003362541
* **Decision**: MARGINAL

### P60-B04 - C (deteccion)

* **Script**: `experiments/p60_05_ruido_por_etiqueta.py`
* **Artefacto**: `reports/p60_C_ruido_por_etiqueta.json`
* **Bug**: dos fallos en la misma auditoria. Primero, el score COMPUESTO se ajustaba regresion logistica sobre el indicador de error y se evaluaba sobre las MISMAS filas, luego su AUC era sobreajuste: 0,5407 donde la version out-of-fold da 0,5070. Segundo, la senal ESTRUCTURAL se guardaba en una columna compartida, de modo que la rama de AG sobrescribia la de HS dentro del mismo bloque de validacion y HS acababa puntuando contra un indicador que no es suyo
* **Correccion**: el compuesto se ajusta en los cuatro pliegues externos y se aplica al quinto, y las senales se guardan POR ETIQUETA. Ademas la referencia del umbral paso de ser 0,5 a ser el incremental del PROPIO CONTROL de esa etiqueta, porque una funcion monotona del control comparte su residuo, y las senales constantes se excluyen del veredicto
* **Reejecutado**: True | la conclusion de la pista se apoya en la senal ESTRUCTURAL de AG, cuyo incremental de 0,6627 se midio con columnas por etiqueta y no depende del compuesto
* **Verificacion tras la correccion**: compuesto out-of-fold 0,5070 frente a 0,5406 en las mismas filas; el incremental del compuesto paso de +0,1966 a -0,0457
* **Clasificacion**: **CORRECTED_AND_VALID** - el artefacto es POSTERIOR al script actual, luego lo produjo el codigo corregido, y el guardia que se anadio esta presente
* **Resultado final valido**: AUC incremental de la senal estructural en AG 0.662712, ganancia 0.112898
* **Decision**: ABIERTA_PARA_C3

### P60-B05 - H (transporte)

* **Script**: `experiments/p60_07_transporte.py`
* **Artefacto**: `reports/p60_H_transporte.json`
* **Bug**: tres fallos encadenados en la metrica. Primero, el coste en una dimension se interpolaba en 128 puntos de cuantil, con un sesgo del 44 por ciento, mayor que la diferencia de 43,6 que se pretendia medir. Segundo, la referencia exacta a mano usaba la media NO PONDERADA de distancias en vez de la ponderada por la medida de la clase. Tercero, la regla de puntuacion exp menos d sobre la mediana colapsaba los dos brazos a la clase mayoritaria, con delta exactamente cero
* **Correccion**: el coste 1D se integro de forma EXACTA como la integral de la diferencia de las dos funciones de distribucion, la referencia paso a ponderar con los pesos de la clase, y la puntuacion se sustituyo por un sigmoidal de la diferencia de distancia a los dos prototipos, que conserva el ranking
* **Reejecutado**: True | el gap check autocalibrado fue lo que detecto que no se podia afirmar nada
* **Verificacion tras la correccion**: sesgo autocalibrado 0,016 por ciento frente a una diferencia Dirac contra distribucion de 43,559 por ciento, o sea 2842 veces el ruido
* **Clasificacion**: **CORRECTED_AND_VALID** - el artefacto es POSTERIOR al script actual, luego lo produjo el codigo corregido, y el guardia que se anadio esta presente
* **Resultado final valido**: corrected_cv 0.4686780062, delta del mecanismo -0.072306717
* **Decision**: CLOSED_BY_DUPLICACION

### P60-B06 - C3 (intervencion)

* **Script**: `experiments/p60_06_intervencion_ag.py`
* **Artefacto**: `reports/p60_C3_intervencion_ag.json`
* **Bug**: ninguno de instrumentacion. El factor de fiabilidad se derivo de las tasas de error IN-SAMPLE del modelo ajustado a las filas de training, que son 0,016 a 0,019 tanto dentro como fuera de la region, luego el ratio salio entre 0,993 y 1,009 y la intervencion fue un no-op
* **Correccion**: no se corrigio el estimador, porque hacerlo habria sido elegir el factor DESPUES de ver el resultado, que es exactamente lo que esta fase prohibe. Se declaro el sesgo del estimador y se registro que el ratio out-of-fold habria sido 0,9614, y se declaro que el eje queda acotado entre dos extremos medidos
* **Reejecutado**: True | el resultado de 0,0000000000 es VALIDO como medicion, y lo que es insuficiente es el estimador del factor, que se declaro como tal
* **Verificacion tras la correccion**: no aplica: no hubo correccion de codigo
* **Clasificacion**: **VALID** - no hubo correccion de codigo: el factor mal estimado es un defecto del estimador, declarado como tal, y la medicion es valida
* **Resultado final valido**: corrected_cv 0.7772647319, delta del mecanismo 0.0
* **Decision**: CLOSED

### P60-B07 - I (meta-reweighting)

* **Script**: `experiments/p60_10_metareweight.py`
* **Artefacto**: `reports/p60_I_metareweight.json`
* **Bug**: al peso de Ren et al. le faltaba el factor c en el NUMERADOR. La forma es eps menos uno sobre g por g, con g el gradiente del objetivo meta, y como G escala con c el numerador debia llevar un factor c para que la razon fuera la del producto interno explicito. El peso salia exactamente la mitad
* **Correccion**: se aplico el factor c al numerador, que hace el peso invariante en c porque el numerador escala como c al cuadrado y el denominador tambien; verificado contra el producto interno explicito en cuatro ejemplos
* **Reejecutado**: True | lo detecto un test unitario, no la ejecucion del experimento
* **Verificacion tras la correccion**: coincide con el producto interno explicito en 1e-12
* **Clasificacion**: **CORRECTED_AND_VALID** - el artefacto es POSTERIOR al script actual, luego lo produjo el codigo corregido, y el guardia que se anadio esta presente
* **Resultado final valido**: corrected_cv 0.4761602463, delta del mecanismo -0.3011044856
* **Decision**: CLOSED

### P60-B08 - J (PolyLoss)

* **Script**: `experiments/p60_09_polyloss.py`
* **Artefacto**: `reports/p60_J_polyloss.json`
* **Bug**: el optimizador compartido de la pista L acepta un parametro `funcion` con objetivo propio y lo IGNORA en silencio, de modo que devuelve la solucion del control. El delta salia con exactamente cero en los cinco pliegues y las tres etiquetas, que es un NO-OP y no un resultado del metodo
* **Correccion**: el optimizador se reescribio con dos ramas explicitas y un protocolo de seis argumentos, y la pista J recibio un GUARDIA que detiene la ejecucion con SystemExit si las dos soluciones no difieren
* **Reejecutado**: True | el numero de 0,0000000000 quedo en el historial como evidencia del no-op y el resultado valido es -0,0017100374; ambos se conservan, segun la regla de conservacion de evidencia
* **Verificacion tras la correccion**: diferencia maxima entre las dos soluciones de 1,31 a 1,65, luego el objetivo propio si se uso
* **Clasificacion**: **CORRECTED_AND_VALID** - el artefacto es POSTERIOR al script actual, luego lo produjo el codigo corregido, y el guardia que se anadio esta presente
* **Resultado final valido**: corrected_cv 0.7755546945, delta del mecanismo -0.0017100374
* **Decision**: CLOSED

### P60-B09 - cierre

* **Script**: `experiments/p60_11_cerrar.py`
* **Artefacto**: `reports/p60_candidate_comparison.csv`
* **Bug**: la cadena que resuelve el delta de cada pista era 'a or b or c', que en Python se detiene en el primer valor VERDADERO. El delta de la pista C es exactamente cero, cero es falsy, luego se pasaba por alto y la tabla lo mostraba como NaN, ocultando el dato mas llamativo de la fase
* **Correccion**: se sustituyo por un helper que toma el primer valor NO NULO, y se reejecutaron el cierre, el informe acumulado y la auditoria
* **Reejecutado**: True | commit e5de2ce
* **Verificacion tras la correccion**: la fila de C pasa a +0,000000
* **Clasificacion**: **CORRECTED_AND_VALID** - el artefacto es POSTERIOR al script actual, luego lo produjo el codigo corregido, y el guardia que se anadio esta presente
* **Resultado final valido**: delta de C 0.0 tras la correccion
* **Decision**: corregido y reejecutado en el commit e5de2ce

## Cifras de P60 que NO pueden usarse como evidencia

**Ninguna.** Las nueve cifras de P60 proceden de una ejecucion del codigo que esta
actualmente en disco, y en los siete casos con correccion de codigo el guardia
correspondiente sigue presente en el script. No hay ninguna cifra de P60 que
haya que retirar.

Esto es una afirmacion fuerte y conviene decir por que es creible: no se deduce de
que alguien afirme que reejecuto, sino de la marca de tiempo del artefacto, que es
posterior a la del script, mas la presencia literal del guardia en el codigo.

## Cifras que quedan como evidencia del fallo, no del metodo

Por la regla de conservacion de evidencia, se conservan los dos valores de los dos
casos en que una cifra erronea fue muy plausible:

```text
P60-B03  DRO con la masa sin normalizar   -0,163083   ERA una regularizacion
                                                   efectiva 1186 veces mayor
P60-B08  PolyLoss con el objetivo ignorado   0,000000   ERA un no-op del
                                                   optimizador compartido
```

Ambas se conservan como registro de instrumentacion, y ninguna se usa como medida del
metodo. Sus valores corregidos son +0,000336 y -0,001710 respectivamente.

## Verificacion del baseline al abrir P61

```text
sha256 de la submission de V03  7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e  coincide: True
corrected_cv del OOF canonico   0.7772114525165721  exacto: True
grupos canonicos                1415 con 0 partidos entre pliegues
```

No se reentrena ninguna pista de P60. La reconciliacion se resuelve de forma no
destructiva, con marcas de tiempo y presencia de los guardias, y el unico calculo que
se repite es la verificacion del baseline, que no entrena nada.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p61_00_reconciliacion.py`. Sin push.*