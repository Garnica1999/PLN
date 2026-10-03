# V25-05: dictamen de gate final

Proyecto `kaggle_toxic_2026_puj`, rama `v25`, 2026-09-27. Referencia **V03 = 0.78306 LB**
congelada, 3 submissions sin gastar. Gastos de esta fase: **0 submissions**.

## 1. El hallazgo de liblinear no abre ninguna via: CONFIRMADO

Tu lectura es correcta, y la razon es mas fuerte que "su OOF cae":

**El knob esta acotado por dos puntos ya medidos.** `tol=1e-4` da 0.777211 (V03, dOOF 0) y
`tol` completo da -0.0016, con la superficie plana entre ambos (objetivo 11512.154 frente a
11512.308 sobre quince ajustes). Cualquier `tol` intermedio tiene como mejor caso posible
"reproducir V03", o sea dOOF 0, que ya queda bajo el umbral +0.03756 por construccion.
**El eje queda cerrado por sus dos extremos, sin medir nada.**

- Con el optimo: dOOF -0.0016, `dLB` implicito -0.00959, o sea 0,79 SE a la baja.
- Precedente directo: enviar por "mejor local" perdio 2 de 2. V05 (Optuna) tuvo el mejor local
  de las 22 (0.8039) y perdio -0.00704; V12_p7 fue el mejor local de la fase v8 (+0.005066,
  4 de 5 pliegues) y perdio -0.01184, la mayor caida de esa fase.
- La metrica de gate es F1 umbralada, no el objetivo de la loss: "menor objetivo" no es
  "mejor modelo" aqui. Y V23 ya demonstro que la mejor evidencia OOF del proyecto no
  transfiere: el gap local-LB paso de +0.005849 a -0.011030.

El valor del hallazgo es de registro, no de candidata: el `corrected_cv` de V03 es en parte
la parada temprana de liblinear, y toda comparacion futura debe declarar si reproduce esa
parada. Ya quedo escrito en `reports/v25_04_cierre.md`. No abre fase.

## 2. Reservas: ninguna de las dos pasa el filtro

**CNN de caracteres: no es una reserva, ya esta medida.** DL-02 (`11_dl_desde_cero.py`), OOF
**0.7410 +/- 0.0097** = **-0.0362** frente a V03, y marcada `cerrado` en
`docs/v25_mechanism_audit.md` linea 24. La familia DL entera esta medida y en contra: CharCNN
0.7410, BiLSTM 0.7364, BiGRU 0.693994 y afinado 0.711639 (**-0.0656**). Re-ejecutarla es
duplicar DL-02. La reserva 0/1 de v5.1 tampoco es una salida: es de arquitectura de
embeddings, y D-041 ya concluyo que la arquitectura mejora la representacion pero no aporta
a V03.

**SCL: no pasa, por precondicion medido y no por falta de evidencia.** SupCon exige un encoder
entrenable, y todos los entrenables legales de este repo, en solitario, quedan 0.036 a 0.117
bajo V03: embedding solo 0.6603 y 0.6687, CharCNN 0.7410, BiLSTM 0.7364, BiGRU 0.7116. Y el
eje de objetivo por pares, el mas proximo a SupCon, ya se ejecuto en
`v12_03_entrenamiento_por_pares.py`: monotono decreciente desde `lambda=0` (+0.000053) hasta
`lambda=2` (-0.010258), con maximo en cero.

- **Hipotesis mecanica:** acercar ejemplos de la misma etiqueta en un espacio denso, y fregar
  la decision de V03 con esa prediccion densa.
- **Refutacion en menos de una hora:** barrido del peso de fusion de 0.00 a 0.30 en pasos de
  0.05, con perfil de celdas y McNemar. El techo de fusion medido del repo es **+0.0064**
  (FastText, 14 configuraciones, 0 adoptables), o sea el 17 % del umbral; un optimo en 0.00
  la refuta.
- **Riesgo:** es el perfil de V23, el unico `dLB` medido de un modelo denso no lineal: 70,9 %
  de vectores de soporte y -0.017 en el gap local-LB.

## 3. El techo de +0.0376 es estructural, y sobre todo es un artefacto del ajuste

**Razon 1, aritmetica.** +0.0376 de dOOF sobre 4.449 celdas de desarrollo, con 757 errores de
V03, es eliminar cerca de un tercio de esos errores. Con 1.483 etiquetas, 4,6 mecanismos
independientes (SVD+RBF, kNN, FM, FastText, ABTT) caen todos entre -0.028 y +0.0132.

**Razon 2, la que decide.** El +0.03756 no es un objetivo de modelado: es el punto donde la
recta ajustada cruza `dLB=0`, y esa recta tiene intercepto -0.00920 porque la muestra
contiene 19 perdedores y un unico ganador, de modo que **debe pasar por debajo del maximo
observado**. Ademas la pendiente 0.2448 esta ajustada sobre n=19 con 4 puntos contaminados
(V16), y el propio informe declara que 19 observaciones no distinguen nada. El +0.03756 es en
su mayor parte una reformulacion de "V03 fue afortunado", y por eso es inalcanzable para
cualquier modelo que no sea una replica de V03.

**Delta OOF realista de lo que queda: +0.005 a +0.010.** Es el rango de las tres mejoras
realizables ya medidas: fusion de FastText +0.0064, FM +0.0055, kNN +0.0050. Con la
pendiente ajustada eso es un `dLB` esperado de -0.008 a -0.007, entre 0,6 y 0,7 SE a la
baja: indetectable por el lado bueno y negativo en el lado malo.

**Cambio en la meta de cierre.** La meta no es "batir a V03" sino "confirmar que V03 es la
meseta". Y hay que corregir un criterio del registro: el filtro de 200 celdas de V24 queda
insuficiente. La FM cambio 278 celdas, **pasa** el filtro, y aun asi tiene **neto negativo**
(-40) con dOOF positivo (+0.0055). El recuento neto y el dOOF pueden tener signo opuesto,
porque el F1 pondera distinto cada etiqueta. De aqui en adelante el gate exige las dos
cosas: `dOOF` > +0.0376 **y** neto de celdas positivo en la misma configuracion.

## Decision

Ninguna de las dos reservas pasa. No se gasta ninguna submission. Las 3 se conservan para
una eventual revision del enunciado o de las reglas, no para buscar mecanismos. El
meta-resultado que si se documenta: en este regimen (N dev 1.483, p 27.431, test 1.154
filas) las mejoras de OOF no transfieren, el LB esta limitado por resolucion a ~0.0122, y
la unica mejora de las 22 submissions vino de un ajuste de un parametro ya medido
(umbrales), no de una tecnica nueva.

CERRAR EL PROYECTO
