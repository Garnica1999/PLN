# Tareas de la fase v10

Pre-registro fechado el 2026-09-26, junto con `plan_v10.md`. Los gates estan ahi; aqui solo
se lleva el estado.

**Estado: fase cerrada.** Once experimentos, V10-00 a V10-10, todos ejecutados y registrados.
Ninguna candidata es enviable. V03 se mantiene. Sin commit, por indicacion del usuario.

## Correspondencia de identificadores

El prompt original numeraba V10-01 a V10-06 para las lineas de las vias. La numeracion real
desplazo un puesto porque P0b necesita su propio identificador, se anadio un arnes de evaluacion
que V10-02 demostro necesario, y la peticion de correlacion abrio dos lineas mas:

| guion del prompt | identificador real | script |
|---|---|---|
| P0 verificacion | V10-00 | verificacion en linea, evidencia en `reports/v10_00_verificacion.json` |
| P3 holdout2 | V10-01 | `experiments/v10_01_holdout2.py` |
| (nuevo) diagnostico del ruido | V10-02 | `experiments/v10_02_ruido_particion.py` |
| (nuevo) arnes de evaluacion | V10-03 | `experiments/v10_03_arnes_evaluacion.py` |
| P1 calibracion fuerte | V10-04 | `experiments/v10_04_calibracion_fuerte.py` |
| P1 regla de decision F1 optima | V10-05 | `experiments/v10_05_regla_f1_optima.py` |
| P1 umbrales estables | V10-06 | `experiments/v10_06_umbrales_estables.py` |
| P2 shift de features | V10-07 | `experiments/v10_07_shift_features.py` |
| P3 confirmacion en dev2 | V10-08 | `experiments/v10_08_confirmacion_dev2.py` |
| P4 BiGRU con afinamiento | V10-11 | `experiments/v10_11_bigru_finetuning.py` |
| (nuevo) correlacion | V10-09 | `experiments/v10_09_correlacion_eliminacion.py` |
| (nuevo) regla de conservacion | V10-10 | `experiments/v10_10_regla_conservacion.py` |

## Hecho

- [x] **P0** worktree `/home/carlos/Git/PLN-v10` en la rama `proyecto-kaggle-toxic-2026-v10` desde
      `23aac93`, con `data/` enlazado y `features/` y `models/` copiados (22 y 9 artefactos).
- [x] **P0** baseline verificado: `corrected_cv` reconstruido 0.7772114525165721 con coincidencia
      exacta, `sha256` de la submission v03 identico, 207 filas de registro, 15 sondas.
- [x] **P0** pre-registro escrito con los gates de las cinco sub-vias, la advertencia de que
      holdout2 no es virginal, y la regla de adopcion de cinco condiciones.
- [x] **P0b** holdout2 creado y congelado, 247 filas en 235 grupos y dev2 con 1236 en 1180, cero
      grupos compartidos, particion reproducible, cero solape con el holdout v2. **GATE NO
      SUPERADO**: la referencia de dev2 quedo en 0.763436, desviacion -0.013776.
- [x] **V10-02** causa de la desviacion identificada: el control de equivalencia da diferencia
      exactamente 0.0, y la banda de subconjuntos de 1236 filas, de recorrido 0.020248, contiene el
      0.763436. Es ruido de tamano y de particion.
- [x] **Addendum de protocolo** escrito **antes** de la via A: delta medio sobre 5 particiones,
      intervalo pareado y recuento de particiones que mejoran, exigiendose las tres a la vez.
- [x] **V10-03** arnes validado en los dos sentidos: el nulo da 0 de 5 y el oraculo da 5 de 5 con
      +0.011781. Reproduce las cifras de V10-02.
- [x] **P1-04** calibracion fuerte: **NO ACCIONABLE**. ECE de la isotonica un 29.94 por ciento mejor
      que Platt, y F1 en +0.000148 con 3 de 5 particiones.
- [x] **P1-05** regla de decision F1 optima: **CERRADA**. -0.003921 con prior simulado, 1 de 5.
- [x] **P1-06** umbrales estables: **CERRADA**. Gate 1 no superado, la holgura es irreducible.
- [x] **P2** shift de features: **VIA CERRADA**, AUC de 0.487717, por debajo del azar.
- [x] **P3** integridad confirmada, 0 de 3 evaluaciones de holdout2 gastadas, nada enviable.
- [x] **Correlacion**: estructura medida por primera vez, correlacion con la etiqueta medida, y
      eliminacion probada en tres umbrales. Ademas, la regla de conservacion probada con un control
      que la v8 no tenia.
- [x] **P5** informe de fase, `D-068` a `D-076`, `RUNLOG` y `PROJECT_STATE` al dia.

## Pendiente

Nada. Las dos lineas que quedaban estan cerradas:

- [x] **P4, V10-11** BiGRU con afinamiento: **ejecutado y H1 cerrada**. Congelado 0.693994, que
      reproduce el 0.6943 de V9-04 como control interno; afinado **0.711639**, con **+0.017645**;
      y el control de tasa unica en 0.699086, solo +0.005092. De los +0.017645 del afinamiento,
      **+0.012553 los aporta la tasa de aprendizaje diferencial** y no el hecho de mover el
      embebido. Sigue a **-0.065572** de V03, asi que **H1 queda cerrada con dos variantes** y la
      via de representacion secuencial se cierra.
- [x] **Notebook, seccion 20** anadida con sintesis de la fase y celda de verificacion con
      aserciones, y **re-ejecutado completo**: 67 celdas, 34 de codigo, **0 errores**, 4.1 s.
      Registro en `reports/notebook_ejecucion_v10.json`.

## Resultados de la via A

| linea | gate 1 | gate 2 | gate 3 | veredicto |
|---|---|---|---|---|
| V10-04 calibracion fuerte | superado, ECE de isotonica 29.94 por ciento mejor que Platt | superado | **no superado** | **NO ACCIONABLE** |
| V10-05 regla F1 optima | no | no | no | cerrada |
| V10-06 umbrales estables | no | no | no | cerrada |

El patron es el mismo de D-067: el mecanismo funciona en su propio lenguaje y no se traslada a la
metrica de la competencia. La calibracion mejora el ECE de 0.0583 a 0.0174 y a la vez mueve la F1
en +0.000148, que es unas treinta veces mas pequena que la banda de ruido de 0.0053.

## Decisiones de conduite acordadas

- Holdout2 se creo **antes** de la via A, no despues.
- Holdout2 **no es virginal** y eso quedo escrito antes de ver resultados: sirve para detectar
  sobreajuste al gate, no para probar generalizacion.
- El filtro de detectabilidad de V8-02 se aplico a cada linea de la via A.
- **Sin commit hasta que el usuario lo pida expresamente.**

## Hallazgo de gobierno que condiciona el resto

V10-02 midio que la metrica corregida tiene una banda de ruido de **mas de 0.017 entre particiones
equivalentes**, por el mismo orden que el ruido de 0.01524 que V8-02 medico en el externo.
Consecuencia: el gate original de +0.002 medido en un solo calculo esta por debajo del suelo de
medicion del instrumento, y por eso el addendum de protocolo exige particiones repetidas y recuento
de particiones que mejoran. Esto no abre ninguna familia cerrada ni relaja ninguna restriccion:
solo deja de pedirle al instrumento mas precision de la que tiene.

## Sobre la peticion de correlacion

La peticion fue: medir la correlacion entre variables de entrada, eliminar una de cada par
correlacionado, medir cuales aportan mas segun su correlacion con la etiqueta, y re-ejecutar para
ver si algo supera a V03. Se hizo todo, y el resultado es negativo en las cuatro partes.

Conviene dejar escrito por que **no es una repeticion** de la linea de la fase v8. Aquella se cerro
diciendo que reducir 20.870 rasgos a 251 por magnitud y redundancia dejaba la metrica en 0,594 y
que la linea quedaba descartada de raiz. Eso colapso el 99 por ciento de las caracteristicas y uso
un criterio de magnitud, no de redundancia, de modo que no podia distinguir quitar una de cada par
de quitar casi toda la senal. Aqui se elimination **una** de cada par, en tres umbrales, y ademas se
midio la estructura de correlacion y la correlacion con la etiqueta, que antes no existian.

Y se vio una cosa que cambia el cierre de la v8, no su veredicto: la conclusion de la v8 era
correcta, pero su prueba no la sostenia. Con un control que la v8 no tenia, la eliminacion por
correlacion queda descartada de forma solida, y la conclusion es **mas fuerte** que la de la v8,
no mas debil.

## Observaciones de calidad pendientes

Ninguna de estas se corrige, y el motivo es el mismo en las cuatro: **el guion ya se ejecuto y su
informe y su fila en `experiments.csv` salieron de el**. Editarlo para que `ruff` quede limpio
romperia la correspondencia entre el codigo y la evidencia que produjo, y un diff que no puede
cambiar ni un numero no vale esa ruptura. Se dejan como observacion.

- `experiments/v10_02_ruido_particion.py`: una variable local llamada `Decision_ficticio` con
  mayuscula inicial, que incumple la convencion de nombres.
- `experiments/v10_09_correlacion_eliminacion.py`: dos variables desempaquetadas y no usadas, `filas`
  y `columnas`, en el bloque A, y una concatenacion implicita de cadenas sin parentesis en el
  generador del informe. Ninguna de las tres afecta al resultado.
- La correccion de la cita de +0.0036 se aplico a V10-06 pero **V10-04 ya estaba registrado** en
  `experiments.csv` cuando se detecto. No se re-ejecuto ni se edito, para no romper la
  correspondencia entre el codigo, el informe y la fila. La correccion queda en D-072 y en el
  informe de fase.
- Ninguna. Las cuatro observaciones anteriores son las que quedan.
