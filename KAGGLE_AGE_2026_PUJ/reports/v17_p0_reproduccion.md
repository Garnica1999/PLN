# V17 · P0' — Referencia reconciliada y A2 de V16 verificado

## 1. La discrepancia 0,6004 contra 0,6005

No son el mismo numero redondeado. Son **dos estimadores de la misma magnitud**:

- **Pooled** (macro-F1 de la confusion de todas las filas, que es `metricas_globales`): **0.6005017473**, que redondea a **0.6005**.
- **Media de los cinco folds**: **0.6004103057**, que redondea a **0.6004**.
- Diferencia: **-0.0000914416**, unas 2 veces la unidad de redondeo del ultimo digito. No se arregla redondeando.

**Por que difieren.** El macro-F1 de una confusion agregada no es la media de los macro-F1 de las partes que la componen, porque cada clase entra con su propio peso en cada subconjunto. Es la desigualdad de Jensen aplicada a la media por clases.

| Fold | macro-F1 |
|---|---|
| 1 | 0.60065636 |
| 2 | 0.60502888 |
| 3 | 0.60320352 |
| 4 | 0.60039292 |
| 5 | 0.59276984 |

**Valor canonico de V17: 0.6005 (pooled).** El motivo no es estetico: el umbral de efecto de D-042 se calibro contra los controles permutados de P1b, medidos en pooled; el IC95 del criterio 3 vive en pooled porque `bootstrap_diferencia` remuestrea la confusion conjunta; y las cinco fases modernas miden su referencia en pooled. Usar 0,6004 regalaria +0,0001 de delta y romperia el emparejamiento con V13, V14, V15 y V16.

**Regla operativa**: todo candidato de V17 se mide con `metricas_globales` (pooled). Si en un informe aparece 0,6004, es que se uso la media por fold y su delta no es comparable. Cuando se cite el historico hay que escribir el estimador.

## 2. A2 de V16 verificado en colderte desde los artefactos

| Arquitectura | macro-F1 recalculado | Publicado | Coincide | 18-24\|25-34 | 25-34\|35-49 |
|---|---|---|---|---|---|
| promedio_fijo | 0.6005 | 0.6005 | si | 0.4963 | 0.4974 |
| peso_aprendido | 0.6002 | 0.6002 | si | 0.4959 | 0.4972 |
| combinador_logistico | 0.6011 | 0.6011 | si | 0.5065 | 0.5131 |

**Delta de A2 sobre la referencia pooled: +0.0006** · IC95 pareado **[-0.0003, +0.0015]** · fronteras **+0.0102** y **+0.0157**.

### Limite de esta verificacion

Es **inerte**: relee las matrices OOF que V16 guardo y recalcula las metricas. Verifica que los artefactos y los informes de V16 sean coherentes entre si, que es lo que detecta un informe desalineado de su JSON. **No** verifica que el entrenamiento sea reproducible bit a bit, porque V16 no guardo las predicciones de su segunda particion de 4 folds; los numeros de esa particion se citan de `experiments/v16_arquitectura.json` y no se recalculan aqui.

**Veredicto: A2 se reproduce** sobre la evidencia inerte disponible.

## 3. Bandas de ruido y umbrales de V17

- Controles permutados de P1b (V13), en pooled: [0.6, 0.6003, 0.6]. Son la banda que calibro el umbral de D-042.
- Gate exploratorio de V17: delta >= +0,0015 con >= 4/5 folds positivos.
- Gate de adopcion: delta >= +0,003 con IC95 consistente, segunda particion positiva y mejora no concentrada en una sola fold.

