# Validacion del notebook consolidado por ejecucion (EBD-26)

Generado por `experiments/embedding_20_validacion_notebook_v5_1.py` (EBD-26 (P10)).

## 1. Que se valida

El **notebook entero**, no solo la seccion nueva. El notebook comparte estado entre
celdas (la de configuracion define `RUTAS` y los ayudantes de presentacion) y las 44
celdas anteriores se ejecutan antes, de modo que ejecutar la seccion 18 por separado
daria una verificacion parcial que podria fallar por una dependencia de las previas.

## 2. Resultado de la ejecucion

| magnitud | valor |
|---|---|
| notebook | `notebooks/10_final_pipeline.ipynb` |
| celdas totales | 63 |
| celdas de codigo | 32 |
| celdas ejecutadas | 32 |
| celdas sin ejecutar | 0 |
| errores | 0 |
| duracion | 4.6 s |
| codigo de salida | 0 |
| limite por celda | 1800 s |

## 3. Comprobaciones exigidas por la seccion 18

| comprobacion | resultado |
|---|---|
| la seccion 18 se ejecuta | OK |
| las aserciones de presupuesto se cumplen | OK |
| el contador de presupuesto se imprime | OK |
| V03 verificado en 0.777211 | OK |

## 4. Errores

Ninguno.

### Ultimas lineas de stderr

```
[NbConvertApp] Converting notebook /home/carlos/Git/PLN/kaggle_toxic_2026_puj/notebooks/10_final_pipeline.ipynb to notebook
[NbConvertApp] Writing 2921089 bytes to /tmp/validacion_notebook_ddz30bln/10_final_pipeline.ipynb

```

## 5. Cobertura de lo que pide la seccion 23 de la spec

| requisito | donde esta |
|---|---|
| Motivacion | 18.0.1 |
| Restricciones y autorizacion | 18.0.2 |
| Corpus: construccion, limpieza, tokenizacion | 18.1 |
| Las tres variantes de corpus con escalas | 18.1 y 18.5 |
| Coverage y OOV | 18.2 y 18.3 |
| FastText frente a Word2Vec-control | 18.3 |
| Pooling y downstream | 18.4 |
| Downstream y fusion | 18.4 y 18.5 |
| Curva de aprendizaje con linea de V03 | 18.5 |
| Complementariedad y slices | 18.6 |
| Vector size, window, min count, subwords, negative | 18.7 |
| Tabla maestra A y B con doble contabilidad | 18.8 |
| Contador automatico de presupuesto con asercion | 18.8 |
| Conclusiones separadas (representacion y tarea) | 18.9 |

## 6. Veredicto

**GO**

GO significa que el notebook completo se ejecuta sin errores y que las
comprobaciones de la seccion 18 se cumplen: la seccion se ejecuta, las aserciones de
presupuesto pasan y V03 se verifica en 0.777211 desde la celda de configuracion.

Fecha: 2026-09-26T00:42:56+00:00. Duracion total: 5.4 s.
