# Decision de P75

**CASO 5: CNN_PERJUDICA**

```text
V03                            0.7772114525
CNN sola (diagnostica)         0.5838755350
alpha primario 0.1  delta -0.0052438938  folds 1/5
gate delta >= 0.003 y >= 4/5 folds y net positivo  ->  NO PASA
rescue 268  damage 813  net -545   (unidades tweet-etiqueta)
finalista                      None
```

## Los dos niveles que no se confunden

```text
Word2Vec + TextCNN produce informacion complementaria   -> solo si los errores muestran correccion especifica
Word2Vec + TextCNN mejora V03                           -> solo si la fusion demuestra transferencia en corrected_cv
```

Las probabilidades del CNN desplazan decisiones de forma netamente negativa. Se cierra la fusion y no se rescata automaticamente: nada de probar alpha mas pequeno, alpha por etiqueta ni stacking sin una hipotesis nueva.

## Estado

```text
V03 = FROZEN
P72 = CLOSED
P73 = COMPLEMENTARY
P74 = EVIDENCIA REPRODUCIBLE
P75 = EJECUTADA
TEXTCNN = WAITING (se cierra aqui salvo que P75 justifique una fase posterior)
test = 0
holdout = 0
leaderboard = 0
submission = 0
```

---

## ADENDA: validacion independiente del Kaggle Specialist

Anadida despues de la validacion. **No cambia ningun numero ni el veredicto.** Corrige el lenguaje y
dos razonamientos, y se registra en DECISIONS.md como D-258.

```text
1. el techo oraculo es casi vacuo: un techo positivo es casi seguro por construccion.
   NO esta probado que el fallo sea de direccion y no de magnitud. Lo sostenido es que, en el rango
   probado, la fusion se movio y el efecto agregado es negativo
2. el AUC no es ~0.5 en todo: HS 0.4872 (ligera inversion), AG 0.4973, TR 0.6045 (asociacion debil)
3. el rescate NO es despreciable: V03 tiene 757 errores, no 978, y el CNN corrige 268 = 35.4 por
   ciento de ellos. Hay complementariedad ESTRUCTURAL; lo que no hay es valor recuperable demostrado
4. la afirmacion de que "no se midio el modelo suelto" era falsa: el reporte si lo mide (0.5838755350).
   Lo defendible es que es peor EN ESTE PUNTO DE OPERACION, lo que no autoriza un juicio general
5. "no depende de ningun umbral" es impreciso: no puede explicarse por malscalibracion GLOBAL, pero
   si depende de los umbrales de V03 y de la recalibracion por pliegue
6. "alcanzables" sobrecuenta: la condicion es necesaria pero no suficiente. La evidencia dura son
   los 76 volteos absolutos y el delta negativo, no la razon
7. las slices se solapan (suman -1210 frente al net global -545): NO son doce pruebas independientes
8. los intervalos bootstrap son optimistas porque las unidades estan agrupadas tambien por tweet.
   "Perjudica" como significancia estricta solo se sostiene, y de forma fragil, en el alpha 0.15
```

### Lo que NO queda cerrado

**Sin finalista significa que la segunda particion NO SE EJECUTO**, luego la transferibilidad queda
**sin medir, no refutada**.

```text
CERRADO    la fusion lineal simple de V03 con este CNN, sin recalibrar y en el rango 0.05-0.15
ABIERTO    recalibracion por pliegue o etiqueta; combinadores selectivos o no lineales; alphas
           fuera del rango; estabilidad del CNN; y la segunda particion, que nunca se ejecuto
```
