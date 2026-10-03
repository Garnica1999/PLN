# P67 - Evidencia congelada

Generado por `experiments/p67_03_cierre_documental.py`. Fecha 2026-09-29T17:10:47+00:00.

Las cifras de este documento se **leen** de los informes de la fase, no se recalculan.
Recalcularlas crearia una segunda fuente de verdad para numeros que ya se produjeron y
verificaron una vez. La unica medicion nueva es el control de azar, y va marcada.

## La decision

```text
MIXED_CON_ATRIBUCION_PARCIAL
```

## Las cinco cifras que la sostienen

| evidencia | valor | lo que prueba | lo que NO prueba |
|---|---:|---|---|
| C-A calibracion escalar | `-0.0029731513` | que la calibracion NO explica el efecto: el modelo de calibracion mas  | nada a favor de MI_arbol. Es un control negativo, y su valor es negati |
| C-E AUC del residuo de MI_arbol | `0.579597` | que hay senal que no es movimiento de frontera: fuera de la banda no h | que esa senal sea exclusiva de MI_arbol, ni que sea explotable |
| C-E desviacion del residuo de MI_arbol | `0.833791` | que el residuo no es colineal con V03 | que la magnitud implique utilidad |
| C-E del candidato mas redundante | `0.219333` | el contraste que da sentido al anterior | nada por si sola |
| C-B atribucion | `4 de 7` | que parte del efecto es de la OPERACION y no de MI_arbol | que MI_arbol no aporte nada: es el mas alto de los siete |
| efecto defendible del blend de P66 | `0.0033068923` | que con 3 de 5 pliegues el blend crece mas que el residual | NO es adoptable: falla el gate de 4 de 5 pliegues, y no es comparable  |
| efecto del residual de P66 | `0.0026743167` | que pasa el gate de delta y el de 4 de 5 pliegues | que sea robusto: su IC95 por grupos incluye el cero |
| bootstrap del residual | `[-0.004431163444677563, 0.010210874472249717]` | que la incertidumbre domina el efecto | que el signo sea negativo. El punto esta por encima de cero, lo que no |
| control de azar con beta fijo | `senal real +0.0034274786 frente a azar -0.0036763898 en el mismo beta` | que la operacion DISTINGUE senal de ruido: la senal real es positiva e | que el efecto sea atribuible a MI_arbol, ni que sea robusto o explotab |
| equivalencia algebraica del mecanismo residual | `8.882e-16` | que el mecanismo residual de P66 es EXACTAMENTE un blend lineal de log | que el efecto sea falso. Solo dice que no es un mecanismo nuevo |

## Las cinco cifras, una por una

### C-A calibracion escalar

- **Medida**: delta del reentreno anidado de un escalar por etiqueta
- **Valor**: `-0.0029731513` (delta de corrected_cv)
- **Lo que prueba**: que la calibracion NO explica el efecto: el modelo de calibracion mas barato posible es peor que el umbral congelado
- **Lo que NO prueba**: nada a favor de MI_arbol. Es un control negativo, y su valor es negativo
- **Fuente**: `reports/p67_controles_decisivos.json`

### C-E AUC del residuo de MI_arbol

- **Medida**: AUC contra la etiqueta, fuera de la banda de margen
- **Valor**: `0.579597` (AUC)
- **Lo que prueba**: que hay senal que no es movimiento de frontera: fuera de la banda no hay margen que mover
- **Lo que NO prueba**: que esa senal sea exclusiva de MI_arbol, ni que sea explotable
- **Fuente**: `reports/p67_controles_decisivos.json`

### C-E desviacion del residuo de MI_arbol

- **Medida**: desviacion estandar del residuo
- **Valor**: `0.833791` (desviacion estandar)
- **Lo que prueba**: que el residuo no es colineal con V03
- **Lo que NO prueba**: que la magnitud implique utilidad
- **Fuente**: `reports/p67_controles_decisivos.json`

### C-E del candidato mas redundante

- **Medida**: desviacion estandar del residuo de huecos_1_2
- **Valor**: `0.219333` (desviacion estandar)
- **Lo que prueba**: el contraste que da sentido al anterior
- **Lo que NO prueba**: nada por si sola
- **Fuente**: `reports/p67_controles_decisivos.json`

### C-B atribucion

- **Medida**: candidatos con delta positivo bajo la misma operacion
- **Valor**: `4 de 7` (conteo)
- **Lo que prueba**: que parte del efecto es de la OPERACION y no de MI_arbol
- **Lo que NO prueba**: que MI_arbol no aporte nada: es el mas alto de los siete
- **Fuente**: `reports/p67_controles_decisivos.json`

### efecto defendible del blend de P66

- **Medida**: delta de MI_arbol|blend_prob
- **Valor**: `0.0033068923` (delta de corrected_cv)
- **Lo que prueba**: que con 3 de 5 pliegues el blend crece mas que el residual
- **Lo que NO prueba**: NO es adoptable: falla el gate de 4 de 5 pliegues, y no es comparable con el residual porque es OTRO mecanismo
- **Fuente**: `reports/p66_score_fusion.json`

### efecto del residual de P66

- **Medida**: delta de MI_arbol|residual
- **Valor**: `0.0026743167` (delta de corrected_cv)
- **Lo que prueba**: que pasa el gate de delta y el de 4 de 5 pliegues
- **Lo que NO prueba**: que sea robusto: su IC95 por grupos incluye el cero
- **Fuente**: `reports/p67_p66_reconciliation.json`

### bootstrap del residual

- **Medida**: IC95 por grupos de near-duplicados
- **Valor**: `[-0.004431163444677563, 0.010210874472249717]` (intervalo)
- **Lo que prueba**: que la incertidumbre domina el efecto
- **Lo que NO prueba**: que el signo sea negativo. El punto esta por encima de cero, lo que no esta es el intervalo
- **Fuente**: `reports/p66_complementarity.json`

### atribucion historica del blend

- **Medida**: brecha de atribucion de P66
- **Valor**: `0.0093216366` (delta de corrected_cv)
- **Lo que prueba**: nada: esta INVALIDADA como medida de informacion atribuida
- **Lo que NO prueba**: no se debe usar. Se conserva solo para trazabilidad, marcada como invalida
- **Fuente**: `reports/p67_p66_reconciliation.json`

### control de azar con beta fijo

- **Medida**: delta de la misma operacion con un score de ruido, escala igualada
- **Valor**: `senal real +0.0034274786 frente a azar -0.0036763898 en el mismo beta` (delta de corrected_cv)
- **Lo que prueba**: que la operacion DISTINGUE senal de ruido: la senal real es positiva en los seis betas y el azar negativo en los seis
- **Lo que NO prueba**: que el efecto sea atribuible a MI_arbol, ni que sea robusto o explotable
- **Fuente**: `reports/p67_interpretation.md`

### equivalencia algebraica del mecanismo residual

- **Medida**: maxima diferencia entre la forma residual y la forma blend
- **Valor**: `8.882e-16` (error numerico)
- **Lo que prueba**: que el mecanismo residual de P66 es EXACTAMENTE un blend lineal de logits mas un desplazamiento de umbral
- **Lo que NO prueba**: que el efecto sea falso. Solo dice que no es un mecanismo nuevo
- **Fuente**: `reports/p67_interpretation.md`

## El valor que NO se usa

```text
+0,0093216366   brecha de atribucion del blend de P66
estado          INVALIDADA como medida de informacion atribuida
```

Se conserva en el registro **solo** para trazabilidad, porque la regla del proyecto es
preservar el ORIGINAL junto al CORREGIDO y documentar la relacion, no borrar el
original.

### Por que estaba mal

Se calculaba como `delta real - delta barajado`. Pero el control barajado **no es un
cero de efecto**: es un **suelo de capacidad de la operacion**, es decir lo que cuesta
esa misma operacion cuando el candidato no lleva senal. Restar un suelo mide una
distancia al suelo, y no constituye una atribucion directa de la senal.

```text
ORIGINAL    +0,0093216366   brecha de atribucion del blend
CORREGIDO   no hay atribucion medible por esa via
RELACION    la cifra original es una distancia al suelo, no una atribucion
```

## El registro de reclasificaciones, con original y corregido

| original | clasificacion | corregido | relacion |
|---:|---|---|---|
| `+0,0033068923` | NON_COMPARABLE | (ninguno) | otro mecanismo, 3 de 5 pliegues, fuera del gate |
| `+0,0026743167` | FINAL_VALID_P66_RESULT | (ninguno) | el unico con delta y pliegues |
| `+0,0093216366` | INVALIDATED | no hay atribucion por esa via | distancia al suelo |
| `+0,0041133588` | CORRECTED | idem | idem, menos severa |
| `-0,0014390421` | ORIGINAL | (ninguno) | es el suelo del residual |
| `-0,0044311634` | ORIGINAL | (ninguno) | extremo inferior del IC95 |
| `+0,0102108745` | ORIGINAL | (ninguno) | extremo superior del IC95 |

Nada se sobrescribe: `reports/p67_p66_reconciliation.csv` guarda los siete valores con
su clasificacion, y los informes originales de P66 siguen intactos en el repositorio.

> V03 remains the best verified leaderboard result.

*Generado por `experiments/p67_03_cierre_documental.py`. Sin push.*