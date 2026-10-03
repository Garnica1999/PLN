# P1 - Dimension de la representacion, con la auditoria incorporada

Generado por `experiments/p72_02_dimension_downstream.py`. Fecha 2026-09-30.
Auditado por el Kaggle Specialist. Las cinco objeciones estan incorporadas.

## Los resultados

| dim | embedding-only | delta | mejor fusion | delta | rescue | damage | neto |
|---|---:|---:|---:|---:|---:|---:|---:|
| V03 | 0,7772114525 | - | 0,7772114525 | - | - | - | - |
| 100d | 0,594729 | -0,182482 | 0,776201 | -0,001011 | 2 | 5 | -3 |
| 200d | 0,597986 | -0,179225 | 0,776423 | -0,000788 | 2 | 5 | -3 |
| 300d | 0,594416 | -0,182796 | 0,776423 | -0,000788 | 2 | 5 | -3 |

## LA RESOLUCION DE LA METRICA, que la auditoria exigio calcular

```text
macro-F1 = media de 3 F1 por etiqueta
mover UNA celda de la etiqueta j cambia la macro en 2/(3*D_j)

HS: D=1326 -> 0,00050277 por celda
TR: D= 890 -> 0,00074906 por celda
AG: D=1127 -> 0,00059154 por celda
```

**La dispersion de la mejor fusion era `0,000222`, que es el 44 por ciento de UNA SOLA
CELDA** en la etiqueta mas facil.

Eso corta en las dos direcciones:

```text
'las tres dimensiones no difieren'  REFORZADO: la diferencia esta por debajo de la unidad
                                    de la metrica. No es que sean parecidas, es que la
                                    metrica no puede distinguirlas.

'200d es el punto'                  DESTRUIDO: elegirla porque su mejor fusion quedo
                                    0,000222 arriba es una lectura post-hoc. Lo
                                    parsimonioso seria 100d, no 200d.
```

## Las cinco objeciones de la auditoria, y que se hace con cada una

### 1. Sobreafirmacion: "la capacidad no es el limite"

**Aceptada.** No se sigue del diseno. Pooling, modelo lineal, escala de fusion y C quedan
fijos y pueden ser el cuello. La version defendible es:

> En el rango 100-300 y con el protocolo congelado, la dimension del embedding no produce
> mejora detectable en transferencia. Este diseno **no permite atribuir el cuello a la
> capacidad**.

### 2. La loss del skip-gram no mide transferencia

**Aceptada.** La loss mide ajuste al corpus de 2020, no transferencia a toxicidad de 2026. Y
**no es comparable entre dimensiones**: distinto numero de parametros, y arranca distinta
(`1,2765` frente a `1,2696`). Solo vale la lectura intra-corrida, que es la de que el modelo
converge.

### 3. Confundidor que no habia visto: dimension contra regularizacion efectiva

**Aceptada y es la mas importante.** El ancho del bloque denso cambia con la dimension, y con
`C` fijo la penalizacion L2 por caracteristica cambia. **La dimension queda confundida con la
regularizacion efectiva.** No tumba la conclusion negativa, pero impide atribuirla a
capacidad.

### 4. "La mejor fusion" es un maximo seleccionado

**Aceptada.** El numero reportado es el maximo de una rejilla de dos escalas sobre el mismo
dev: lleva la maldicion del ganador. La dispersion `0,000222` es la de maximos, no la del
pipeline. La curva completa esta en `twitter_word2vec_dimension_results.csv`.

### 5. El control de azar es el dato mas fuerte y el mas incomodo

**Aceptada.** Si a la escala que menos dano hace los embeddings reales rinden igual o peor
que el ruido, la lectura literal es que **este bloque denso no aporta senal util**. Y entonces
la dimension es irrelevante **por trivialidad**, no porque sobre capacidad.

## Lo que la auditoria pide y NO se ha ejecutado

```text
varianza de semilla      3-5 semillas por dimension, para poder afirmar equivalencia
curvas completas         no solo el mejor punto
azar en AMBOS regimenes  solo-embedding y fusion, y ademas vectores de inicializacion
                         aleatoria evaluados aguas abajo
comparacion emparejada   por pliegue, mas bootstrap sobre filas del dev
sensibilidad de umbral   +-0,02 alrededor de 0,44/0,51/0,46
checklist de 13 puntos   de la auditoria, sin ejecutar
```

**Sin esos controles no se puede afirmar equivalencia**, solo ausencia de diferencia
detectable. Se declara.

## La pregunta del encargo, respondida con la version defendible

```text
A. 200 dimensiones son suficientes y la dimension no es el cuello.
```

Con dos matices que la auditoria impone:

```text
no es que 200 sea el punto: es que 100, 200 y 300 son indistinguibles
no se puede decir que la capacidad NO sea el limite; se dice que la
  dimension no produce mejora detectable en este pipeline
```

B queda descartada (ninguna dimension mayor mejora). **El cuello esta en otro sitio**, y los
candidatos que el diseno NO descarta son el pooling, la escala de fusion y el modelo aguas
abajo.

## RIESGO RESIDUAL DE LEAKAGE, que debe declararse

El argumento **no es** que se midiera cero solapamiento. Es de **procedencia**:

```text
el corpus es de 2020 con periodo CERRADO, el test es de 2026
el solapamiento medido contra TRAIN es cero, exactas y normalizadas
el test NO se puede medir, por definicion de las reglas
```

Quedan tres escenarios residuales, improbables y **no demostrables**:

```text
el test incluyera retuits de textos de 2020 presentes en el corpus
la recoleccion del test solapara con la del corpus
la procedencia del test fuera desconocida
```

Se declaran como riesgo formal. Y el chequeo ejecutado es mas debil de lo que suena: solo
igualdad exacta y normalizada. **Falta un chequeo de near-duplicados** (Jaccard de shingles)
contra train y dev, que se puede correr sin tocar el test.

## COMPLIANCE: una ambiguedad que necesita confirmacion escrita

La regla dice: *se pueden usar modelos de vectorizacion, pero deben **entrenarse desde cero**
y hacer todo desde cero.*

Hay dos lecturas:

```text
(A) sin artefactos preentrenados, el modelo se entrena desde cero
(B) solo con datos de la competencia
```

**El diseno cumple (A) y no cumple (B).** La regla prohibe usar el test para entrenamiento,
no prohibe datos externos publicos, y "entrenarse desde cero" se refiere al modelo. Bajo
lectura literal: cumple.

**Pero el juez es el profesor, y esto necesita confirmacion escrita antes de publicar
cualquier conclusion apoyada en el embedding.** Si su intencion era (B), esta evidencia no
serviria para el entregable. Es una decision de gobernanza, no tecnica.

## Detalle del corpus y la tokenizacion

```text
fuente     mmaguero/covid19-spanishtweets-earlylateapril2020 (Kaggle)
licencia   CC BY-SA 4.0, atribucion obligatoria y share-alike si se distribuyen derivados
periodo    29 de marzo a 1 de mayo de 2020, CERRADO
tamano     361.994 documentos, 10.000.029 tokens
metadata   NO se usa: solo la columna `text`
```

Tokenizacion EMB-NORMAL-TW v1, la MISMA en las tres dimensiones: conserva el contenido del
hashtag donde v5 lo colapsaba, menciones a marcador con prefijo, alargamientos sin colapsar.

**Cobertura identica en las tres dimensiones: 68,6 por ciento.** El vocabulario lo fija
`min_count`, no la dimension. No se usa como criterio de superioridad.

> V03 remains the best verified leaderboard result.

*Sin push.*