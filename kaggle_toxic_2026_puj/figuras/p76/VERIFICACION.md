# P76: verificacion de las figuras

```text
metodo   las dos reliability se REGENERAN desde reports/p75_paired_oof.csv mas los parametros de
         Platt de reports/p76_calibracion.json, y se comparan con los PNG almacenados pixel a
         pixel. El titulo del modelo se dibuja como pixeles, luego la comparacion lo incluye.
resultado NINGUNA figura mal etiquetada
```

| figura | modelo regenerado | identica pixel a pixel |
|---|---|---|
| `p76_01_reliability_v03.png` | V03 | SI |
| `p76_02_reliability_cnn.png` | CNN | SI |

Las otras cinco se comprobaron por existencia, peso, no blancura y coherencia con
`reports/p76_calibracion.json`. Ademas se recalcularon los cuatro miembros desde el JSON y
reproducen su macro-F1 al decimal.

**La sospecha de intercambio V03/CNN era infundada.** El bucle de generacion usa la misma variable
`nombre` para el titulo y para el fichero, y ambos mapas son consistentes. La duda era legitima
porque el titulo no se puede leer de un PNG; se resuelto regenerando y comparando.

## Nota sobre la figura `p76_04_cv_mecanismos.png`

La comprobacion automatica de coherencia se formulo mal, no la figura. Se quiso verificar
`min < V03 < max` y la condicion da falso, porque **las ocho configuraciones quedan por debajo de
V03**: la mas alta es `C_V03raw_CNNcal_alpha0.05` con 0.7754366610 y la mas baja 0.7624993889. Es
decir, las ocho barras estan bajo la linea roja del campeon, que es exactamente lo que afirma
`reports/p76_calibracion.json`. La lectura correcta de la figura es "ninguna condicion alcanza al
campeon", no "la figura esta mal".
